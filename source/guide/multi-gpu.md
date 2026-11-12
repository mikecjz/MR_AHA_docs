# Multi-GPU (CUDA)

$A^H A$ is a sum over coils, so one application can be spread over the GPUs of a node. Each
GPU applies the operator for a share of the coils, and the partial results are summed on the
caller's GPU (the *primary*).

Whether an operator is multi-GPU is decided when it is created, with `devices`. The same
GPUs build the operator's constants and apply it. An application then only broadcasts `x` and
gathers the partial outputs. Both methods work this way:

* **Toeplitz**: every GPU holds a copy of `phi2_toep`, computed by all GPUs together
  ([below](#set-up-computing-phi2_toep-on-all-gpus)), and its coils' maps. Multi-GPU makes the
  operator faster but do not raise the largest problem size: the whole kernel must fit on every
  GPU.
* **NUFFT**: every GPU holds its own cufinufft plans, the Gram blocks, the trajectory and its
  coils' maps. There is no kernel, so the memory per GPU is far smaller. Set-up is plan creation
  on every GPU.

```matlab
op   = mr_aha.create_op(SEs, phi2, traj, config, 'devices', 1:4);   % 1-based gpuDevice indices
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

## How the operator uses the GPUs

* **Primary GPU.** Inputs and outputs live on the current GPU (MATLAB's `gpuDevice`, CuPy's
  current device, or `config.gpu_device_id`), which always takes part. The current device is
  unchanged on return. MATLAB uses the other GPUs only inside the library, so its "one GPU per
  session" rule is not violated.
* **How many GPUs.** With $C$ coils on $G$ GPUs the time is set by $\lceil C/G \rceil$ coils, so
  the smallest number of GPUs with the same $\lceil C/G \rceil$ is used: 12 coils on 5 GPUs use
  4. Small problems (fewer than $2^{22}$ upsampled values per coil, e.g. the 2D benchmark) stay
  on the primary GPU; `MR_AHA_MGPU_MIN_WORK` changes the threshold. On one GPU the operator costs
  the same as the single-GPU primitives: it creates no worker streams and makes no copies of `x`.
* **Remainder split (Toeplitz).** When the coils do not divide over the GPUs, the remaining coils
  can be split across all GPUs *by rank*, with an all-to-all exchange of k-space slabs around the
  block-diagonal stage. With `MR_AHA_MGPU_SPLIT=auto` (default) a cost model decides: it times the
  operator, measures the GPU-to-GPU bandwidth, and enables the split when it predicts a gain of
  at least 5 %. In practice that needs NVLink. `on` and `off` force the choice.
* **Memory.** `share_primary` (default on; `MR_OP_SHARE_PRIMARY` in C) lets the primary GPU use the
  caller's `SEs` and mask in place instead of a second copy. They must then stay unchanged while
  the operator exists. The primary also holds one `x`-sized staging buffer per other GPU.
* **MATLAB lifetime.** The MEX file stays locked while an operator exists, so `clear mex` cannot
  unload it under a live operator. After `reset(gpuDevice)` or selecting another GPU, `apply`
  raises an error instead of computing. Delete the operator and create a new one.

## Set-up: computing `phi2_toep` on all GPUs

The Toeplitz operator computes its kernel on all its GPUs at once:

1. Each GPU computes a contiguous share of the $(L^2+L)/2$ components, directly into its own copy
   of the kernel.
2. The GPUs exchange their shares (an all-gather), chunk by chunk, overlapped with the
   computation.
3. Each GPU sends one copy at a time, to the other GPUs in rotated order, so every link carries
   about one transfer at a time.

Each GPU thus computes about $1/G$ of the kernel and receives the other $(G-1)/G$ over its own
link, so the set-up time does not grow with the number of GPUs. Every component is computed once,
so the copies on all GPUs are bit-identical.

Measured on 4 L40S over PCIe:

| problem | kernel | `calc_toep` on 1 GPU | operator set-up on 4 GPUs |
|---|---|---|---|
| 2D radial 320², L = 72 | 8.6 GB | 2.0 s | 1.9 s |
| 3D kooshball 160³, L = 12 | 20.5 GB | 7.9 s | 4.3 s |

The 4-GPU figure includes the kernel on every GPU.
[The performance notes](../background/performance.md#34-set-up-computing-phi2_toep-on-all-gpus)
explain the mechanism and give all measurements.

* **`toep_dist`** selects how every GPU gets the kernel. The environment variable
  `MR_AHA_TOEP_SPLIT` (same values; `off` = `broadcast`) overrides the argument; any other value
  is an error.

  | value | what happens |
  |---|---|
  | `'auto'` (default) | a cost model chooses between `'split'` and `'replicate'` |
  | `'split'` | the exchange above |
  | `'replicate'` | every GPU computes the whole kernel itself, with no transfers |
  | `'broadcast'` | the primary computes the kernel, then copies it to the others |

* **When `'split'` pays off.** It is faster than `'replicate'` when a GPU can receive one
  component faster than it can compute one. That needs only a few GB/s (about 4 GB/s for the 2D
  L = 72 kernel), well below PCIe peer-to-peer and far below NVLink.
* **Cost model (`'auto'`).** At creation every GPU computes and times the first chunk of its share,
  which is useful work in either case. All GPUs then copy 64 MB to every receiver *at the same
  time*, with the schedule of the exchange, to measure the bandwidth each gets when they share
  links. The library then picks `'split'` or, for example without peer access, `'replicate'`.
  `info` reports the choice as `toep_source`.
* **Chunks.** `MR_AHA_TOEP_CHUNKS` (default 4) is the minimum number of chunks per GPU. A value
  of 1 disables the overlap of computation and exchange; use it for benchmarking only.
* **Report.** `info` gives `method`, `toep_source`, `toep_devices`, `toep_ms`,
  `toep_component_ms`, `toep_gather_gbs` and `create_ms`; `kernel(k)` returns the kernel held on
  the `k`-th GPU.
* **Memory.** Every GPU that applies the operator holds a copy of the kernel plus, while
  computing, the transient buffers of `calc_toep`, sized within `mem_fraction` of the memory left
  after the copy. GPUs the operator does not use for coils (e.g. the 5th of 5 GPUs with 12 coils)
  still help compute the kernel.

## Tests and measurements

* `tests/check_op.py` and `tests/check_op_matlab.m` compare the single-GPU and multi-GPU operators,
  with both methods, against the single-GPU primitives and the NUFFT reference. See
  [Testing](../development/testing.md).
* The scripts of [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) time the
  set-up and the application of both methods on any number of GPUs.
* Measured scaling is in
  [CUDA operator optimisation and multi-GPU](../benchmarks/cuda-multigpu.md).
