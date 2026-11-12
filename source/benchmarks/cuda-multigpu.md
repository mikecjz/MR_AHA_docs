# CUDA operator optimisation and multi-GPU (2026-11)

This study measures two features of the CUDA $A^HA$ operator:

* **Block-diagonal stage.** Kernels specialised on $L$ (register vectors, each stored element of
  `phi2_toep` read once), an in-place masked kernel that works through an `int32` index map, and
  coil groups of 4 that read `phi2_toep` once per group. The results agree with the generic
  kernels to single-precision rounding.
* **Multi-GPU** (`create_op(..., devices=...)`). The coils are distributed over the GPUs of a
  node, with an optional rank split of the remaining coils over NVLink.

[Implementation notes](../background/implementation.md) and [Multi-GPU](../guide/multi-gpu.md)
describe both, and [How the CUDA operator gets its speed](../background/performance.md) explains
the mechanisms.

**Protocol.** Inputs are constant arrays. Timing is the median of 5 (single GPU) or 10
(multi-GPU) calls ([`bench_aha_backend.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/bench_aha_backend.py),
[`bench_mgpu.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/bench_mgpu.py)).
The masked runs use a centred disc/ball that keeps the same fraction of the grid as the masks of
the [main study](results.md).

**Reference.** "Generic" is the same build with `MR_AHA_CUDA_GENERIC_BD=1`: the generic kernels,
one coil at a time. The tables of the [main study](results.md) were measured with those kernels.

## One GPU

| problem | GPU | generic | default | gain |
|---|---|---|---|---|
| 2d | TITAN RTX | 2.8 ms | 2.1 ms | −25 % |
| 2d | H100 SXM | 1.2 ms | 0.6 ms | −50 % |
| stack | TITAN RTX | 1361 ms | 989 ms (groups of 2) | −27 % |
| stack | H100 SXM | 254.5 ms | 178.9 ms | −30 % |
| stack masked | TITAN RTX | 1388 ms | 955 ms (groups of 2) | −31 % |
| stack masked | H100 SXM | 260.7 ms | 171.2 ms | −34 % |
| 3d | TITAN RTX | 1906 ms | 1622 ms (groups of 1) | −15 % |
| 3d | H100 SXM | 362.6 ms | 254.6 ms | −30 % |
| 3d masked | TITAN RTX | does not fit | 1441 ms (groups of 1) | fits |
| 3d masked | H100 SXM | 337.6 ms | 237.2 ms | −30 % |

* **Share of the block-diagonal stage.** It is 45–47 % of the GPU time with the generic kernels
  and 20–25 % with the default ones (TITAN RTX, `MR_AHA_CUDA_PROFILE=1`). It then runs close to
  the DRAM bandwidth, and the two FFTs are about 70 % of the time on the H100.
* **Coil group.** The group is limited by memory. On the 24 GB TITAN RTX the stack kernel (14 GB)
  leaves room for groups of 2 within the default `mem_fraction = 0.5`, and the 20 GB 3D kernel
  for groups of 1. On the H100 (80 GB) groups of 4 fit everywhere and capture nearly all of the
  gain: groups of 6 add at most 2–3 %.
* **Masked operator.** The masked kernel works in place and needs no scratch buffer per coil.
  That is what lets the masked 3D problem run on the TITAN RTX.

## Multi-GPU (H100 SXM, NVLink)

Up to 3 H100 GPUs, 12 coils, 369 GB/s measured
GPU-to-GPU:

| problem | 1 GPU | 2 GPUs | 3 GPUs |
|---|---|---|---|
| stack | 180.0 ms | 93.9 ms (×1.92) | 65.3 ms (×2.76) |
| stack masked | 171.3 ms | 90.7 ms (×1.89) | 62.8 ms (×2.73) |
| 3d | 252.9 ms | 130.4 ms (×1.94) | 88.2 ms (×2.87) |
| 3d masked | 237.1 ms | 122.7 ms (×1.93) | 82.9 ms (×2.86) |
| 2d | 0.5 ms | stays on one GPU (too small) | |

Together with the default kernels, the stack operator goes from 254 ms (one H100, generic
kernels) to 65 ms on three H100s.

**Remainder split.** The split applies only when the coils do not divide over the GPUs. With 7
coils on 3 GPUs it saves 4 % in 3D (65.7 → 63.0 ms). The automatic mode, a cost model timed at
plan creation, enables the split there and keeps whole coils where they are faster, for example
with 8 coils on 3 GPUs. The TITAN RTX GPUs have no peer access (copies go
through the host at ~4 GB/s), so the automatic mode keeps whole coils there.

These runs were made with the batch script `run_mgpu_h100.sh`
(`PART=test|single|bench|split|matlab`); the raw numbers are in
`results/h100/` and `results/titan_rtx/` (`aha_cuda*.json`).
