# Multi GPU

`create_op(..., devices=...)` spreads the coils over the GPUs of a node, with an optional rank
split of the remaining coils over NVLink, and computes the Toeplitz kernel on all GPUs at once.
[Multi-GPU](../guide/multi-gpu.md) describes how to use it, and
[How the CUDA operator gets its speed](../background/performance.md) explains the mechanisms.

## Summary

* **With NVLink, one $A^H A$ scales almost linearly.** On four H100s the operator is 3.1–3.6×
  faster than on one: 11.7 ms (`2d72`), 53 ms (`stack`) and 70 ms (`3d`). 20 CG iterations of
  the 3D problem take 1.46 s instead of 5.10 s.
* **Over PCIe the gain is smaller.** Four L40S make one $A^H A$ 2.4–3.1× faster, and 20 CG
  iterations 2.2–3.0× faster.
* **Without peer access, two GPUs are the most that help.** On TITAN RTXs, two GPUs make the
  operator 1.3–1.5× faster; four are slower than two.
* **Set-up gets faster where the GPUs are well connected**: 2.35 s instead of 4.09 s for the 3D
  kernel on four H100s, 5.4 s instead of 8.5 s on two L40S. On TITAN RTXs only the 3D kernel is
  set up faster on more GPUs.

## Problems

The `2d72` (320², $L = 72$), `stack` (320² × 56, $L = 12$) and `3d` (160³, $L = 12$) problems of
[Single GPU](single-gpu.md#problems), 12 coils each. The small `2d` problem (0.5 ms on one GPU)
stays on one GPU.

**Timing.** The same as for [Single GPU](single-gpu.md#problems): set-up is the creation of the
operator, with the kernel on every GPU (median of 2 creations after a first one); one $A^H A$ is
the median of 10 calls after 2 warm-up calls; CG is 20 iterations from zero. The speed-up in
brackets is against the same operator on one GPU of the same node.

## Runtime

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

Four NVIDIA TITAN RTX, 24 GB, over PCIe: two GPUs on each of two PCIe switches, one NUMA node.
The TITAN RTX has no peer access, so the GPUs exchange data through host memory.

**Operator set-up**

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | **2.50 s** | 2.63 s | 2.86 s |
| `2d72` masked | **2.12 s** | 2.35 s | 2.85 s |
| `stack` | **3.05 s** | 3.49 s | 3.99 s |
| `stack` masked | **3.21 s** | 3.67 s | 4.56 s |
| `3d` | does not fit | does not fit | does not fit |
| `3d` masked | 18.91 s | 15.76 s | **11.40 s** |

**One $A^H A$** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 224 ms | **159 ms** (1.41×) | 170 ms (1.31×) |
| `2d72` masked | 199 ms | **146 ms** (1.37×) | 162 ms (1.23×) |
| `stack` | 1.01 s | **0.80 s** (1.27×) | 1.13 s (0.89×) |
| `stack` masked | 0.98 s | **0.78 s** (1.25×) | 1.12 s (0.87×) |
| `3d` | does not fit | does not fit | does not fit |
| `3d` masked | 1.38 s | **0.92 s** (1.51×) | 1.01 s (1.37×) |

**20 CG iterations** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 4.52 s | **3.23 s** (1.40×) | 3.44 s (1.31×) |
| `2d72` masked | 4.03 s | **2.95 s** (1.36×) | 3.28 s (1.23×) |
| `stack` | 20.54 s | **16.24 s** (1.26×) | 22.98 s (0.89×) |
| `stack` masked | 19.93 s | **15.97 s** (1.25×) | 22.78 s (0.87×) |
| `3d` | does not fit | does not fit | does not fit |
| `3d` masked | 27.90 s | **18.57 s** (1.50×) | 20.44 s (1.37×) |

The unmasked 3D kernel (20.5 GB) does not fit on a 24 GB GPU together with the operator's
buffers, also when the coils are spread over several GPUs, because every GPU holds the whole
kernel.

:::

:::{tab-item} L40S
:sync: l40s

Four NVIDIA L40S, 48 GB, over PCIe, as Slurm allocated them: GPUs 0 and 1 on one PCIe switch,
GPUs 2 and 3 each on its own NUMA node.

**Operator set-up**

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 2.16 s | **1.60 s** | 3.45 s |
| `2d72` masked | 2.22 s | **1.59 s** | 3.12 s |
| `stack` | 2.77 s | **2.27 s** | 4.00 s |
| `stack` masked | 2.64 s | **2.42 s** | 4.17 s |
| `3d` | 8.45 s | **5.42 s** | 9.91 s |
| `3d` masked | 7.34 s | **5.14 s** | 10.32 s |

**One $A^H A$** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 90.1 ms | 57.5 ms (1.57×) | **37.1 ms** (2.43×) |
| `2d72` masked | 85.7 ms | 54.4 ms (1.58×) | **35.6 ms** (2.41×) |
| `stack` | 753 ms | 436 ms (1.72×) | **302 ms** (2.50×) |
| `stack` masked | 744 ms | 430 ms (1.73×) | **300 ms** (2.48×) |
| `3d` | 1.06 s | 0.58 s (1.83×) | **0.35 s** (3.07×) |
| `3d` masked | 1.02 s | 0.55 s (1.84×) | **0.34 s** (3.03×) |

**20 CG iterations** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 1.83 s | 1.18 s (1.56×) | **0.82 s** (2.25×) |
| `2d72` masked | 1.74 s | 1.12 s (1.56×) | **0.74 s** (2.36×) |
| `stack` | 15.33 s | 9.04 s (1.70×) | **6.33 s** (2.42×) |
| `stack` masked | 15.18 s | 8.92 s (1.70×) | **6.39 s** (2.38×) |
| `3d` | 21.46 s | 11.81 s (1.82×) | **7.20 s** (2.98×) |
| `3d` masked | 20.58 s | 11.23 s (1.83×) | **6.94 s** (2.97×) |

On four GPUs, copying the kernel to the two GPUs across NUMA nodes costs more than computing it on
fewer GPUs, so set-up is fastest on two. The automatic distribution mode was the fastest of the
four modes on four GPUs (`3d`: 10.0 s; `replicate` 10.3 s, `broadcast` 11.5 s, `split` 16.8 s). On a node with two
pairs of GPUs, each pair on its own PCIe switch, four GPUs computed the 3D kernel in 4.2 s; see
[How the CUDA operator gets its speed](../background/performance.md#34-set-up-computing-phi2_toep-on-all-gpus).

:::

:::{tab-item} H100
:sync: h100

Four NVIDIA H100 80GB (SXM) with NVLink between every pair.

**Operator set-up**

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 1.94 s | **1.29 s** | 1.29 s |
| `2d72` masked | 2.08 s | **1.28 s** | 1.30 s |
| `stack` | 1.59 s | **1.45 s** | 1.52 s |
| `stack` masked | 1.97 s | 1.55 s | **1.53 s** |
| `3d` | 4.09 s | 2.79 s | **2.35 s** |
| `3d` masked | 4.82 s | 2.83 s | **2.37 s** |

**One $A^H A$** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 36.2 ms | 20.5 ms (1.77×) | **11.7 ms** (3.08×) |
| `2d72` masked | 32.9 ms | 18.7 ms (1.76×) | **10.7 ms** (3.08×) |
| `stack` | 178 ms | 93.9 ms (1.90×) | **53.3 ms** (3.34×) |
| `stack` masked | 170 ms | 90.4 ms (1.89×) | **50.9 ms** (3.35×) |
| `3d` | 252 ms | 130 ms (1.94×) | **70.3 ms** (3.59×) |
| `3d` masked | 232 ms | 120 ms (1.93×) | **64.3 ms** (3.60×) |

**20 CG iterations** (speed-up over one GPU in brackets)

| problem | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `2d72` | 733 ms | 420 ms (1.74×) | **245 ms** (2.99×) |
| `2d72` masked | 667 ms | 385 ms (1.73×) | **224 ms** (2.98×) |
| `stack` | 3.63 s | 1.95 s (1.86×) | **1.14 s** (3.19×) |
| `stack` masked | 3.48 s | 1.88 s (1.85×) | **1.09 s** (3.20×) |
| `3d` | 5.10 s | 2.65 s (1.92×) | **1.46 s** (3.50×) |
| `3d` masked | 4.69 s | 2.45 s (1.91×) | **1.34 s** (3.50×) |

**Remainder split.** When the coils do not divide evenly over the GPUs, the remaining coils can
be split by rank over NVLink. With 7 coils on 3 GPUs this saves 4 % in 3D (65.7 → 63.0 ms). The
automatic mode, a cost model timed at plan creation, enables the split there and keeps whole
coils where they are faster, for example with 8 coils on 3 GPUs.

:::

::::

## Sources

All tables were measured with `bench_aha.py` of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) and MR_AHA
[`dc61256`](https://github.com/mikecjz/MR_AHA/tree/dc61256), with the default (automatic)
distribution of the kernel:

```bash
python bench_aha.py --problem 2d72 stack 3d --methods toeplitz --mask both --gpus 1 2 4 \
    --cg 20 --setup-reps 3
```

The remainder-split numbers come from the earlier
[`bench_mgpu.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/bench_mgpu.py), with
raw numbers in MR_AHA's history at commit
[`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks/results/esplhpc-cp089).
The set-up measurements with every distribution mode of the kernel are in
[How the CUDA operator gets its speed](../background/performance.md#34-set-up-computing-phi2_toep-on-all-gpus).
