# CUDA operator optimisation and multi-GPU (2026-11)

These runs postdate the tables of the [main study](results.md), which used the operator as it
was before. Two changes were made to the CUDA $A^HA$ (see
[Implementation notes](../background/implementation.md) and [Multi-GPU](../guide/multi-gpu.md);
the mechanisms are explained in
[How the CUDA operator was made faster](../background/performance.md)):

* **Block-diagonal stage.** Kernels specialised on $L$ (register vectors, each stored element of
  `phi2_toep` read once), an in-place masked path through an `int32` index map (this was the
  "faster masked execute path" recommended below), and coil groups of 4 that read `phi2_toep` once
  per group. The results are bitwise identical to the previous kernels.
* **Multi-GPU** (then `mr_aha.MultiGPUPlan` / `mr_aha_op`, now `create_op(..., devices=...)`):
  coils distributed over the GPUs of a node,
  with an optional rank split of the remaining coils over NVLink.

Inputs are constant arrays, timing is the median of 5 (single GPU) or 10 (multi-GPU) calls
([`bench_aha_backend.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/bench_aha_backend.py), [`bench_mgpu.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/bench_mgpu.py)). "Before" is
the same build with `MR_AHA_CUDA_GENERIC_BD=1`, i.e. the generic kernels one coil at a time, which
is the previous operator. The masked runs use a centred disc/ball with the threshold mask's kept
fraction.

## One GPU

| problem | GPU | before | after | gain |
|---|---|---|---|---|
| 2d | TITAN RTX | 2.8 ms | 2.1 ms | −25 % |
| 2d | H100 SXM | 1.2 ms | 0.6 ms | −50 % |
| stack | TITAN RTX | 1361 ms | 989 ms (groups of 2) | −27 % |
| stack | H100 SXM | 254.5 ms | 178.9 ms | −30 % |
| stack masked | TITAN RTX | 1388 ms | 955 ms (groups of 2) | −31 % |
| stack masked | H100 SXM | 260.7 ms | 171.2 ms | −34 % |
| 3d | TITAN RTX | 1906 ms | 1622 ms (groups of 1) | −15 % |
| 3d | H100 SXM | 362.6 ms | 254.6 ms | −30 % |
| 3d masked | TITAN RTX | out of memory | 1441 ms (groups of 1) | fits now |
| 3d masked | H100 SXM | 337.6 ms | 237.2 ms | −30 % |

* The block-diagonal stage was 45–47 % of the GPU time before (TITAN RTX, `MR_AHA_CUDA_PROFILE=1`)
  and is 20–25 % after. It now runs close to the DRAM bandwidth; the two FFTs are about 70 % of the
  time on the H100.
* The coil group is limited by memory: on the 24 GB TITAN RTX the stack kernel (14 GB) leaves room
  for groups of 2 within the default `mem_fraction = 0.5`, and the 20 GB 3D kernel for none. On the
  H100 (80 GB) groups of 4 fit everywhere; groups of 6 gain at most 2–3 % more.
* The masked path no longer needs the `[num_masked, L]` scratch buffer per coil (2 GB for 3D), which
  is what lets the masked 3D problem run on the TITAN RTX.

## Several GPUs (H100 SXM, NVLink)

Up to 3 GPUs of one node (`esplhpc-cp089`; more were not free), 12 coils, 369 GB/s measured
GPU-to-GPU:

| problem | 1 GPU | 2 GPUs | 3 GPUs |
|---|---|---|---|
| stack | 180.0 ms | 93.9 ms (×1.92) | 65.3 ms (×2.76) |
| stack masked | 171.3 ms | 90.7 ms (×1.89) | 62.8 ms (×2.73) |
| 3d | 252.9 ms | 130.4 ms (×1.94) | 88.2 ms (×2.87) |
| 3d masked | 237.1 ms | 122.7 ms (×1.93) | 82.9 ms (×2.86) |
| 2d | 0.5 ms | stays on one GPU (too small) | |

Together with the single-GPU changes, the stack operator goes from 254 ms (one H100, before) to
65 ms on three. The remainder split matters only when the coils do not divide over the GPUs, and
then only a little: with 7 coils on 3 GPUs it saves 4 % in 3D (65.7 → 63.0 ms) and is neutral for
the stack (47.1 vs 47.7 ms); with 8 coils on 3 GPUs it costs 12 % (51.6 → 57.9 ms). The
automatic mode (cost model timed at plan creation) keeps whole coils where the split loses and
picks it where it wins, except for the near-tie stack case. The local TITAN RTX workstation has no GPU peer
access (copies go through the host at ~4 GB/s), so the split is never chosen there.

These runs were made with [`hpc/run_mgpu_h100.sh`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/hpc/run_mgpu_h100.sh)
(`PART=test|single|bench|split|matlab`); the raw numbers are in
[`results/esplhpc-cp089/`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks/results/esplhpc-cp089) and
[`results/pact400-recon3/`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks/results/pact400-recon3) (`aha_cuda*.json`).
