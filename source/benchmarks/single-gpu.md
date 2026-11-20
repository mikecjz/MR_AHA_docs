# Single GPU

One GPU, the Toeplitz operator of MR_AHA (`mr_aha.create_op`) against the NUFFT operator with
the same interface, which applies a type-2 and a type-1 NUFFT in every call (`aha_nufft_plan`,
or `create_op(..., method='nufft')`). The NUFFT numbers are those of the NUFFT library's default
settings.

## Summary

* **On the NVIDIA GPUs, Toeplitz is faster than NUFFT on every problem.** One $A^H A$ is 9–30×
  faster for the small 2D problem, 3.6–6.1× at rank 72, 2.1–4.7× for stack-of-stars and
  5.0–8.8× for the 3D kooshball. 20 CG iterations gain about as much, except on the small 2D
  problem (8–19×).
* **The NUFFT operator is ready at once; the Toeplitz operator first computes its kernel**, in
  1.6–4.7 s for the large problems on an H100 and up to 18.6 s (masked 3D on the TITAN RTX).
  The kernel pays for itself within 3 applications of $A^H A$ for the stack-of-stars and 3D
  problems and within 14 for the 2D ones, so set-up plus 20 CG iterations still finishes
  sooner with Toeplitz.
* **The fastest GPU is the H100**: 0.52 ms (`2d`), 33 ms (`2d72`), 0.17 s (`stack`) and 0.23 s
  (`3d`) per $A^H A$, and 4.7 s for 20 CG iterations in 3D.
* **The RTX 5090 is the fastest on the small `2d` problem** (0.32 ms) and close to the H100 at
  rank 72 (39 ms against 33 ms), but at about half the H100's memory bandwidth it takes about
  twice as long for the stack-of-stars (0.31 s) and 3D (0.43 s) problems, between the H100 and
  the L40S.
* **On the Apple M3 Ultra, Toeplitz is also faster than NUFFT (mlx-nufft) on every problem**:
  7–47× per $A^H A$, and 5.8× on the high-rank `2d72` problem (182 ms against 1.06 s), where the
  tiled block-diagonal kernels for $L > 32$ run it at about the speed of a TITAN RTX. The stack
  and 3D problems run at about the speed of an L40S (0.89 s for the stack, 1.10 s in 3D).
* **Both operators compute the same thing**: they agree to 1–4·10⁻⁶ per application, and the
  20-iteration CG solutions to 3–6·10⁻⁶ (see [Accuracy](#accuracy)).

## Problems

| name | size | coils | $L$ | samples |
|---|---|---:|---:|---|
| `2d` | 160² | 16 | 6 | 377 radial spokes × 160 |
| `2d72` | 320² | 12 | 72 | 754 radial spokes × 320 |
| `stack` | 320² × 56 | 12 | 12 | 280 spokes × 320, × 56 $k_z$ |
| `3d` | 160³ | 12 | 12 | 15680 kooshball spokes × 160 |

**Timing.** Wall time with device synchronization.

* **Operator set-up** is the creation of the operator: for Toeplitz the kernel (`calc_toep`) and,
  if masked, the mask; for NUFFT the plans. Median of 2 creations after a first one.
* **One $A^H A$** is the median of 10 calls after 2 warm-up calls.
* **20 CG iterations** is plain conjugate gradient from zero, exactly 20 iterations, without the
  set-up.

"Masked" is the masked Toeplitz operator (`is_masked = true`), which keeps 79–81 % of the
upsampled grid in 2D and stack-of-stars and 54 % in 3D. The speed-up is NUFFT against the faster
Toeplitz operator. The fastest time of each row is in bold.

## Runtime

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

NVIDIA TITAN RTX, 24 GB (Turing, sm_75), CUDA 12.6, cufinufft 2.5.1.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **1.71 ms** | 14.8 ms | 14.0 ms |
| `2d72` | **3.27 ms** | 2.49 s | 2.16 s |
| `stack` | **18.6 ms** | 3.12 s | 3.10 s |
| `3d` | **5.26 ms** | does not fit | 18.60 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 19.4 ms | 2.23 ms | **2.19 ms** | 8.9× |
| `2d72` | 706 ms | 225 ms | **199 ms** | 3.6× |
| `stack` | 2.58 s | 1.01 s | **0.98 s** | 2.6× |
| `3d` | 7.73 s | does not fit | **1.39 s** | 5.6× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 395 ms | 49.6 ms | **47.1 ms** | 8.4× |
| `2d72` | 14.24 s | 4.89 s | **4.02 s** | 3.5× |
| `stack` | 52.59 s | 20.54 s | **19.92 s** | 2.6× |
| `3d` | 156.6 s | does not fit | **28.00 s** | 5.6× |

The unmasked 3D kernel (20.5 GB) does not fit in 24 GB together with the operator's buffers; the
masked one does.

:::

:::{tab-item} L40S
:sync: l40s

NVIDIA L40S, 48 GB (Ada, sm_89), CUDA 12.9, cufinufft 2.5.1.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **131 ms** | 218 ms | 213 ms |
| `2d72` | **141 ms** | 2.18 s | 2.16 s |
| `stack` | **174 ms** | 2.84 s | 2.43 s |
| `3d` | **216 ms** | 8.45 s | 8.26 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 13.1 ms | 0.44 ms | **0.43 ms** | 30.2× |
| `2d72` | 436 ms | 90.2 ms | **85.8 ms** | 5.1× |
| `stack` | 1.62 s | 0.75 s | **0.74 s** | 2.2× |
| `3d` | 5.73 s | 1.06 s | **1.02 s** | 5.6× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 267 ms | **13.9 ms** | 14.2 ms | 19.2× |
| `2d72` | 8.75 s | 1.83 s | **1.74 s** | 5.0× |
| `stack` | 32.86 s | 15.35 s | **15.19 s** | 2.2× |
| `3d` | 115.9 s | 21.48 s | **20.52 s** | 5.6× |

On this cluster every NUFFT plan creation costs 0.12–0.2 s, which is most of the NUFFT set-up
and of the small `2d` Toeplitz set-up.

:::

:::{tab-item} H100
:sync: h100

NVIDIA H100 80GB (SXM, sm_90), CUDA 12.9, cufinufft 2.5.1.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **120 ms** | 190 ms | 189 ms |
| `2d72` | **128 ms** | 2.25 s | 2.35 s |
| `stack` | **156 ms** | 1.64 s | 1.66 s |
| `3d` | **191 ms** | 4.09 s | 4.65 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 6.61 ms | **0.52 ms** | 0.54 ms | 12.6× |
| `2d72` | 199 ms | 36.1 ms | **32.7 ms** | 6.1× |
| `stack` | 800 ms | 178 ms | **170 ms** | 4.7× |
| `3d` | 2.04 s | 0.25 s | **0.23 s** | 8.8× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 135 ms | **13.4 ms** | 13.6 ms | 10.1× |
| `2d72` | 3.99 s | 0.73 s | **0.66 s** | 6.0× |
| `stack` | 16.06 s | 3.63 s | **3.48 s** | 4.6× |
| `3d` | 40.95 s | 5.09 s | **4.68 s** | 8.7× |

On this cluster every NUFFT plan creation costs 0.12–0.2 s, which is most of the NUFFT set-up
and of the small `2d` Toeplitz set-up.

:::

:::{tab-item} RTX 5090
:sync: rtx5090

NVIDIA GeForce RTX 5090, 32 GB (Blackwell, sm_120), driver 595.99 (CUDA 13.2), CUDA toolkit
12.9, library built for `sm_120` only, cufinufft 2.6.0 (bundled with MR_AHA). The GPU also
drives the workstation's display.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **0.86 ms** | 4.17 ms | 4.18 ms |
| `2d72` | **1.23 ms** | 0.91 s | 0.95 s |
| `stack` | **1.24 ms** | 1.12 s | 1.04 s |
| `3d` | **1.77 ms** | 2.39 s | 2.49 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 5.08 ms | **0.315 ms** | 0.324 ms | 16.1× |
| `2d72` | 157 ms | 42.6 ms | **39.1 ms** | 4.0× |
| `stack` | 661 ms | 323 ms | **312 ms** | 2.1× |
| `3d` | 2.14 s | 0.54 s | **0.43 s** | 5.0× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 103 ms | **7.61 ms** | 7.80 ms | 13.5× |
| `2d72` | 3.14 s | 0.86 s | **0.79 s** | 4.0× |
| `stack` | 13.34 s | 6.59 s | **6.37 s** | 2.1× |
| `3d` | 42.88 s | 10.95 s | **8.69 s** | 4.9× |

The unmasked 3D operator fits in 32 GB (24.0 GB in use), unlike on the 24 GB TITAN RTX. A
repeated run gave the same $A^H A$ times to within 1 %.

:::

:::{tab-item} Apple M3 Ultra
:sync: m3ultra

Apple M3 Ultra, 512 GB unified memory, Metal backend. The NUFFT operator is MLX with
mlx-nufft; `calc_toep` uses mlx-nufft and finishes in the library.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **1.90 ms** | 11.1 ms | 9.69 ms |
| `2d72` | **2.89 ms** | 3.99 s | 3.93 s |
| `stack` | **2.49 ms** | 3.76 s | 3.79 s |
| `3d` | **26.0 ms** | 14.38 s | 14.04 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 78.7 ms | **1.68 ms** | 1.94 ms | 46.8× |
| `2d72` | 1.06 s | 214 ms | **182 ms** | 5.8× |
| `stack` | 6.29 s | 0.92 s | **0.89 s** | 7.0× |
| `3d` | 19.85 s | 1.18 s | **1.10 s** | 18.0× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 1.57 s | **0.038 s** | 0.043 s | 41.0× |
| `2d72` | 21.45 s | 5.04 s | **4.00 s** | 5.4× |
| `stack` | 126.7 s | **18.88 s** | 20.01 s | 6.7× |
| `3d` | 398.3 s | 25.60 s | **23.82 s** | 16.7× |

The `2d72` Toeplitz rows of $A^H A$ and CG use the tiled block-diagonal kernels for $L > 32$
(see [Apple silicon](../guide/apple-silicon.md#performance)). Before them, the masked operator
took 1.19 s per $A^H A$ and 24.19 s for 20 CG iterations, slower than NUFFT. The CG vector
updates run on the CPU. The `2d` problem is too small to occupy the GPU.

:::

::::

## Accuracy

The relative difference $\|a-b\|/\|b\|$ between the Toeplitz and the NUFFT operator, and between
their 20-iteration CG solutions, is the same on every GPU to the digits shown:

| problem | one $A^H A$ | CG solution | masked vs unmasked CG | CG error vs truth (every operator) |
|---|---:|---:|---:|---:|
| `2d` | 1.0·10⁻⁶ | 3.4·10⁻⁶ | 1.8·10⁻⁵ | 8.355·10⁻² |
| `stack` | 1.2·10⁻⁶ | 3.6·10⁻⁶ | 2.7·10⁻⁵ | – |
| `3d` | 3.8·10⁻⁶ | 1.7–2.5·10⁻⁴ | 2.3·10⁻⁵ | 2.483·10⁻¹ |

* **The 3D CG difference is a numerical floor, not a Toeplitz error.** The 3D problem is badly
  conditioned, so 20 CG iterations amplify single-precision rounding. Two runs of the same NUFFT
  operator with different cufinufft spreading methods differ by 3.6·10⁻⁶ per application and by
  3.5·10⁻⁴ after 20 iterations, more than Toeplitz and NUFFT do.
* **Metal and CUDA agree.** The Metal CG solutions differ from the CUDA ones by 5–10·10⁻⁶ in 2D
  and stack and by 5·10⁻⁵ in 3D. In 3D the single-precision mlx-nufft kernel gives an operator
  error of 1.4·10⁻⁵ (CUDA 1.9·10⁻⁶); `calc_toep(..., nufft='finufft')` brings it to 5·10⁻⁶ at
  four times the set-up time.
* **For sparse trajectories the mask costs accuracy.** On a two-arm spiral, a mask radius of 2
  keeps 4 % of the grid at an operator error of 3·10⁻², radius 8 keeps 14 % at 3·10⁻³; see
  [Neighborhood mask](neighbor-mask.md).

The runtime measurements give the same operator differences on the NVIDIA GPUs: 0.9–1.1·10⁻⁶ in
2D and stack-of-stars and 2.5·10⁻⁶ in 3D (1.1–1.6·10⁻⁶ and 1.9·10⁻⁵ on the M3 Ultra).

## Sources

The runtime tables were measured with `bench_aha.py` of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) and MR_AHA
[`dc61256`](https://github.com/mikecjz/MR_AHA/tree/dc61256), on the synthetic presets:

```bash
python bench_aha.py --problem 2d 2d72 stack 3d --methods toeplitz nufft_c --mask both \
    --cg 20 --setup-reps 3          # nufft_native instead of nufft_c on Apple silicon
```

The RTX 5090 tab was measured later with the same command and MR_AHA
[`da3e148`](https://github.com/mikecjz/MR_AHA/tree/da3e148), built with
`-DCMAKE_CUDA_ARCHITECTURES=120`.

The `2d72` Toeplitz rows of one $A^H A$ and 20 CG iterations on the M3 Ultra were measured again
with MR_AHA [`961f7ef`](https://github.com/mikecjz/MR_AHA/tree/961f7ef), which adds the tiled
Metal kernels (`--problem 2d72 --methods toeplitz --mask both --cg 20`). Its set-up and NUFFT
rows are from the run above.

The accuracy table comes from the original study, which used measured human 2D radial and
stack-of-stars data and a synthetic kooshball. Its scripts and raw numbers are in MR_AHA's
history at commit [`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks).
