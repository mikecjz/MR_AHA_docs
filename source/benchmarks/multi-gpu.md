# Multi GPU

`create_op(..., devices=...)` spreads the coils over the GPUs of a node, with an optional rank
split of the remaining coils over NVLink, and computes the Toeplitz kernel on all GPUs at once.
[Multi-GPU](../guide/multi-gpu.md) describes how to use it, and
[How the CUDA operator gets its speed](../background/performance.md) explains the mechanisms.

## Summary

* **One $A^H A$ scales almost linearly.** On three H100s the stack and 3D operators run 2.7–2.9×
  faster than on one: 65 ms and 83–88 ms.
* **Set-up gets faster too.** Building a 4-GPU operator, with the kernel on every GPU, takes less
  time than `calc_toep` on one GPU: 4.2 s against 7.8 s for the 3D kernel on four L40S, 2.4 s
  against 4.0 s on four H100s.
* **Reconstructions end sooner on more GPUs.** Set-up plus 10 CG iterations of the 3D problem
  takes 19.3 s on one L40S and 8.0 s on four; 6.7 s on one H100 and 3.1 s on four.

## Problems

The `2d72` (320², $L = 72$), `stack` (320² × 56, $L = 12$) and `3d` (160³, $L = 12$) problems of
[Single GPU](single-gpu.md#problems), 12 coils each, with synthetic inputs. Times are medians of
10 calls ($A^H A$) or 2 runs after a warm-up (set-up).

## Runtime

::::{tab-set}

:::{tab-item} L40S
:sync: l40s

Four NVIDIA L40S over PCIe: two pairs, each on its own PCIe switch, the pairs on different NUMA
nodes. Each GPU receives 14–16 GB/s from the others.

**Operator set-up** (kernel on every GPU)

| GPUs | `2d72` | `stack` | `3d` |
|---|---:|---:|---:|
| 1 (`calc_toep`) | 2.05 s | 2.41 s | 7.75 s |
| 2 | **1.46 s** | **2.15 s** | 5.36 s |
| 4 | 1.82 s | 2.40 s | **4.20 s** |

**Set-up and 10 CG iterations**

| GPUs | `2d72` | `stack` | `3d` |
|---|---:|---:|---:|
| 1 | 5.95 s | 10.80 s | 19.27 s |
| 2 | 3.07 s | 6.68 s | 11.26 s |
| 4 | **2.72 s** | **5.60 s** | **8.04 s** |

For the problems with many cheap kernel components (`2d72`, `stack`), set-up on 2 GPUs is faster
than on 4, but 4 GPUs still finish the reconstruction first.

:::

:::{tab-item} H100
:sync: h100

NVIDIA H100 80GB (SXM) with NVLink between every pair, 365–369 GB/s from GPU to GPU.

**One $A^H A$** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 3 GPUs |
|---|---:|---:|---:|
| `stack` | 180.0 ms | 93.9 ms (1.92×) | **65.3 ms** (2.76×) |
| `stack` masked | 171.3 ms | 90.7 ms (1.89×) | **62.8 ms** (2.73×) |
| `3d` | 252.9 ms | 130.4 ms (1.94×) | **88.2 ms** (2.87×) |
| `3d` masked | 237.1 ms | 122.7 ms (1.93×) | **82.9 ms** (2.86×) |

The small `2d` problem (0.5 ms) stays on one GPU.

**Operator set-up** (kernel on every GPU)

| GPUs | `2d72` | `stack` | `3d` |
|---|---:|---:|---:|
| 1 (`calc_toep`) | 2.16 s | 1.93 s | 3.97 s |
| 2 | **1.06 s** | **1.48 s** | 2.81 s |
| 4 | 1.32 s | 1.69 s | **2.36 s** |

**Set-up and 10 CG iterations**

| GPUs | `2d72` | `stack` | `3d` |
|---|---:|---:|---:|
| 1 | 5.68 s | 3.65 s | 6.68 s |
| 2 | 2.75 s | 2.46 s | 4.14 s |
| 4 | **2.17 s** | **2.25 s** | **3.09 s** |

**Remainder split.** When the coils do not divide evenly over the GPUs, the remaining coils can
be split by rank over NVLink. With 7 coils on 3 GPUs this saves 4 % in 3D (65.7 → 63.0 ms). The
automatic mode, a cost model timed at plan creation, enables the split there and keeps whole
coils where they are faster, for example with 8 coils on 3 GPUs.

:::

::::

GPUs without peer access, such as the TITAN RTX, copy through the host at about 4 GB/s; the
automatic mode then keeps whole coils on each GPU.

## Sources

The $A^H A$ scaling was measured with
[`bench_mgpu.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/bench_mgpu.py); the
raw numbers are in MR_AHA's history at commit
[`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks/results/esplhpc-cp089).
The set-up measurements, with every distribution mode of the kernel, are in
[How the CUDA operator gets its speed](../background/performance.md#34-set-up-computing-phi2_toep-on-all-gpus).
