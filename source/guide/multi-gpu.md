# Multi-GPU (CUDA)

$A^H A$ is a sum over coils, so one application can be spread over several GPUs of a node: each
GPU applies the operator for a share of the coils and the partial results are summed on the
caller's GPU (the *primary*). Whether an operator uses several GPUs is decided when it is created,
with `devices`; the same GPUs build the operator's constants and apply it. An application then only
broadcasts `x` and gathers the partial outputs. Both methods work this way:

* **Toeplitz**: every GPU holds a copy of `phi2_toep` (computed by all GPUs together, below) and of
  its coils' maps. Multi-GPU makes the operator faster but does not raise the largest problem size:
  the whole kernel must fit on every GPU.
* **NUFFT**: every GPU holds its own cufinufft plans, the Gram blocks and trajectory, and its coils'
  maps. No kernel, so the memory per GPU is far smaller; set-up is plan creation on every GPU.

```matlab
op   = mr_aha_op.create_op(SEs, phi2, traj, config, 'devices', 1:4);   % 1-based gpuDevice indices
AhAx = op * x;                                                         % or op.apply(x)
info = op.info();                                                      % devices, coils, timings
delete(op);
```

```python
with mr_aha.create_op(cfg, SEs, phi2, traj, devices=[0, 1, 2, 3]) as op:   # or devices="all"
    y = op(x)
    print(op.info)
```

```c
int devs[4] = {0, 1, 2, 3};
opt.devices = devs; opt.n_devices = 4;          /* or n_devices = MR_OP_ALL_DEVICES */
mr_op_create(&cfg, &in, &opt, &op);
```

* **Primary GPU.** Inputs and outputs live on the current GPU (MATLAB's `gpuDevice`, CuPy's
  current device, or `config.gpu_device_id`), which always takes part. The current device is
  unchanged on return. MATLAB uses the other GPUs only inside the library, so its "one GPU per
  session" rule is not violated.
* **How many GPUs.** With $C$ coils on $G$ GPUs the time is set by $\lceil C/G \rceil$ coils, so
  the smallest number of GPUs with the same $\lceil C/G \rceil$ is used (12 coils on 5 GPUs use
  4). Small problems (fewer than $2^{22}$ upsampled values per coil, e.g. the 2D benchmark) stay
  on the primary GPU; `MR_AHA_MGPU_MIN_WORK` changes the threshold. One GPU costs nothing extra
  over the single-GPU primitives (no worker streams, no copies of `x`).
* **Remainder split (Toeplitz).** When the coils do not divide over the GPUs, the remaining coils
  can be split across all GPUs *by rank*, with an all-to-all exchange of k-space slabs around the
  block-diagonal stage. `MR_AHA_MGPU_SPLIT=auto` (default) enables it only when a cost model, fed
  with a one-coil timing and the measured GPU-to-GPU bandwidth, predicts a gain of at least 5 %,
  which in practice needs NVLink; `on` / `off` force it.
* **Memory.** `share_primary` (default on; `MR_OP_SHARE_PRIMARY` in C) lets the primary GPU use the
  caller's `SEs` and mask in place instead of a second copy; they must then stay unchanged while
  the operator exists. The primary also holds one `x`-sized staging buffer per other GPU.
* **MATLAB lifetime.** The MEX file stays locked while an operator exists, so `clear mex` cannot
  unload it under a live operator. After `reset(gpuDevice)` or selecting another GPU, `apply`
  raises an error instead of computing; delete the operator and create a new one.

## Set-up: computing `phi2_toep` on all GPUs

Computing the kernel on one GPU and copying it to the others costs $T_\mathrm{calc}$ on that GPU
while the others are idle, plus copying the kernel from the primary to each of the other $G-1$
GPUs, all through the primary's own link. For 2D radial with L = 72 (an 8.6 GB kernel) on 4 L40S
over PCIe that set-up took 3.7 s, four times as long as 10 CG iterations. The Toeplitz operator
therefore computes the kernel on all its GPUs at once (`toep_dist`). On the same 4 GPUs the set-up
drops to 1.9 s, which is faster than `calc_toep` alone on one GPU (2.0 s). For a 160³ kooshball
(L = 12) it drops from 12.1 s to 4.3 s. See [the performance notes, §3.4](../background/performance.md) for
the mechanism and all measurements.

Each GPU computes a contiguous share of the components directly into its own copy of the kernel.
The GPUs then exchange their shares (an all-gather), chunk by chunk, overlapped with the
computation. Each GPU sends one copy at a time, to the others in rotated order, so every link
carries about one transfer at a time. Letting every GPU send to all the others at once was
several times slower on PCIe nodes whose GPUs sit on different NUMA nodes. Each GPU computes
about $T_\mathrm{calc}/G$ and receives $(G-1)/G$ of the kernel over its own link, instead of the
primary sending $(G-1)$ whole kernels. Every component is computed once, so the copies on all GPUs
are bit-identical.

* **`toep_dist`** (`'auto'` default, `'split'`, `'replicate'`, `'broadcast'`). `'split'` is the
  exchange above, `'replicate'` lets every GPU compute the whole kernel itself (no transfers),
  `'broadcast'` is the old path (the primary computes, then copies to the others). The
  environment variable `MR_AHA_TOEP_SPLIT` (same values; `off` = `broadcast`) overrides the
  argument; any other value is an error.
* **When splitting pays off.** Computing $1/G$ and exchanging is faster than every GPU computing
  the whole kernel itself when a GPU can receive one component faster than it can compute one.
  That needs only a few GB/s (about 4 GB/s for the 2D L = 72 kernel), well below PCIe
  peer-to-peer and far below NVLink.
* **Cost model (`'auto'`).** At creation every GPU computes and times the first chunk of its share
  (useful work in either case). All GPUs then copy 64 MB to every receiver *at the same time*, with
  the schedule of the exchange, to measure the bandwidth each gets when they share links. The
  library then picks the exchange (`toep_source = 'split'`) or lets every GPU compute the kernel
  itself (`'replicate'`, e.g. without peer access). `MR_AHA_TOEP_CHUNKS` (default 4) is the
  minimum number of chunks per GPU; 1 disables the overlap and is for benchmarking only.
* **Report.** `info` gives `method`, `toep_source`, `toep_devices`, `toep_ms`,
  `toep_component_ms`, `toep_gather_gbs` and `create_ms`; `kernel(k)` returns the kernel held on
  the `k`-th GPU.
* **Memory.** Every GPU that applies the operator holds a copy of the kernel plus, while
  computing, the transient buffers of `calc_toep`, sized within `mem_fraction` of the memory left
  after the copy. GPUs the operator does not use for coils (e.g. the 5th of 5 GPUs with 12 coils)
  still help compute the kernel.

`tests/check_op.py` and `tests/check_op_matlab.m` compare the operator on one and several GPUs,
with both methods, against the single-GPU primitives and the NUFFT reference.
The scripts of [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) time the set-up and the application of both
methods on any number of GPUs; measured scaling is in
[CUDA operator optimisation and multi-GPU](../benchmarks/cuda-multigpu.md).
