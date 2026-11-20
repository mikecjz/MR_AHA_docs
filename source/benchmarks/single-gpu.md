# Single GPU

One GPU, the Toeplitz operator of MR_AHA (`mr_aha.create_op`) against the Python NUFFT
reference `mr_aha.aha_nufft_plan` (see [Operations](../guide/operations.md)), which applies a
type-2 and a type-1 NUFFT in every call. The NUFFT baseline is `aha_nufft_plan` on every
backend: on the NVIDIA GPUs it runs on CuPy with cufinufft's Python interface, on Apple silicon
on MLX with mlx-nufft. It does not use the library's own NUFFT operator
(`create_op(..., method='nufft')`). The NUFFT numbers are those of the NUFFT library's default
settings.

## Summary

* **On the NVIDIA GPUs, Toeplitz is faster than NUFFT on every problem.** One $A^H A$ is 10–48×
  faster for the small 2D problem, 3.5–11.5× at rank 72, 2.5–12.5× for stack-of-stars and
  3.3–13.4× for the 3D kooshball; on the larger problems the gain is largest on the H100
  (11.5–13.4×). 20 CG iterations gain about as much, except on the small 2D problem (9.5–36×).
* **The NUFFT operator is ready at once; the Toeplitz operator first computes its kernel**, in
  1.5–4.7 s for the large problems on an H100 and up to 18.9 s (masked 3D on the TITAN RTX).
  The kernel pays for itself within 3 applications of $A^H A$ for the stack-of-stars and 3D
  problems and within 10 for the 2D ones, so set-up plus 20 CG iterations still finishes
  sooner with Toeplitz.
* **The fastest GPU is the H100**: 0.52 ms (`2d`), 33 ms (`2d72`), 0.17 s (`stack`) and 0.23 s
  (`3d`) per $A^H A$, and 5.1 s for 20 CG iterations in 3D.
* **The RTX 5090 is the fastest on the small `2d` problem** (0.32 ms) and close to the H100 at
  rank 72 (39 ms against 33 ms), but at about half the H100's memory bandwidth it takes about
  twice as long for the stack-of-stars (0.31 s) and 3D (0.43 s) problems, between the H100 and
  the L40S.
* **On the Apple M3 Ultra, Toeplitz is also faster than NUFFT (mlx-nufft) on every problem**:
  7–45× per $A^H A$, and 5.8× on the high-rank `2d72` problem (182 ms against 1.05 s), where the
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

NVIDIA TITAN RTX, 24 GB (Turing, sm_75), CUDA 12.6, cufinufft 2.5.1. NUFFT: `aha_nufft_plan`
with CuPy 14.2 and cufinufft 2.5.1 (Python).

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **2.77 ms** | 14.3 ms | 13.3 ms |
| `2d72` | **3.03 ms** | 2.47 s | 2.14 s |
| `stack` | **4.50 ms** | 3.03 s | 3.07 s |
| `3d` | **17.2 ms** | does not fit | 18.91 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 21.8 ms | 2.22 ms | **2.11 ms** | 10.3× |
| `2d72` | 699 ms | 223 ms | **198 ms** | 3.5× |
| `stack` | 3.98 s | 1.01 s | **0.98 s** | 4.1× |
| `3d` | 10.72 s | does not fit | **1.38 s** | 7.8× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 443 ms | 49.2 ms | **46.8 ms** | 9.5× |
| `2d72` | 14.05 s | 4.51 s | **4.01 s** | 3.5× |
| `stack` | 79.87 s | 20.54 s | **19.90 s** | 4.0× |
| `3d` | 214.3 s | does not fit | **27.89 s** | 7.7× |

The unmasked 3D kernel (20.5 GB) does not fit in 24 GB together with the operator's buffers; the
masked one does.

:::

:::{tab-item} RTX 5090
:sync: rtx5090

NVIDIA GeForce RTX 5090, 32 GB (Blackwell, sm_120), driver 595.99 (CUDA 13.2), CUDA toolkit
12.9, library built for `sm_120` only, cufinufft 2.6.0 (bundled with MR_AHA) for the Toeplitz
kernel. NUFFT: `aha_nufft_plan` with CuPy 14.2 and cufinufft 2.5.1 (Python). The GPU also
drives the workstation's display.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **2.19 ms** | 3.97 ms | 3.61 ms |
| `2d72` | **2.57 ms** | 0.91 s | 0.96 s |
| `stack` | **2.82 ms** | 0.99 s | 1.18 s |
| `3d` | **5.50 ms** | 2.38 s | 2.49 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 13.7 ms | **0.316 ms** | 0.325 ms | 43.3× |
| `2d72` | 234 ms | 42.7 ms | **39.1 ms** | 6.0× |
| `stack` | 1.38 s | 323 ms | **313 ms** | 4.4× |
| `3d` | 1.44 s | 0.54 s | **0.43 s** | 3.3× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 271 ms | **7.57 ms** | 7.79 ms | 35.8× |
| `2d72` | 4.69 s | 0.86 s | **0.79 s** | 5.9× |
| `stack` | 27.58 s | 6.59 s | **6.38 s** | 4.3× |
| `3d` | 28.85 s | 10.95 s | **8.68 s** | 3.3× |

The unmasked 3D operator fits in 32 GB (23.9 GB in use), unlike on the 24 GB TITAN RTX.

:::

:::{tab-item} L40S
:sync: l40s

NVIDIA L40S, 48 GB (Ada, sm_89), CUDA 12.9, cufinufft 2.5.1. NUFFT: `aha_nufft_plan` with
CuPy 14.2 and cufinufft 2.5.1 (Python).

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **130 ms** | 211 ms | 212 ms |
| `2d72` | **138 ms** | 2.19 s | 2.21 s |
| `stack` | **138 ms** | 2.87 s | 2.48 s |
| `3d` | **216 ms** | 8.54 s | 7.34 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 20.7 ms | 0.435 ms | **0.431 ms** | 48.0× |
| `2d72` | 304 ms | 90.1 ms | **85.9 ms** | 3.5× |
| `stack` | 1.90 s | 0.75 s | **0.74 s** | 2.5× |
| `3d` | 3.87 s | 1.06 s | **1.02 s** | 3.8× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 417 ms | 13.51 ms | **13.46 ms** | 31.0× |
| `2d72` | 6.11 s | 1.83 s | **1.74 s** | 3.5× |
| `stack` | 38.30 s | 15.45 s | **15.16 s** | 2.5× |
| `3d` | 77.57 s | 21.44 s | **20.54 s** | 3.8× |

On this cluster every NUFFT plan creation costs 0.12–0.2 s, which is most of the NUFFT set-up
and of the small `2d` Toeplitz set-up.

:::

:::{tab-item} H100
:sync: h100

NVIDIA H100 80GB (SXM, sm_90), CUDA 12.9, cufinufft 2.5.1. NUFFT: `aha_nufft_plan` with
CuPy 14.2 and cufinufft 2.5.1 (Python).

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **126 ms** | 191 ms | 195 ms |
| `2d72` | **128 ms** | 2.25 s | 2.17 s |
| `stack` | **128 ms** | 1.49 s | 1.63 s |
| `3d` | **195 ms** | 4.07 s | 4.73 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 18.7 ms | **0.524 ms** | 0.533 ms | 35.7× |
| `2d72` | 378 ms | 36.2 ms | **32.9 ms** | 11.5× |
| `stack` | 2.14 s | 178 ms | **171 ms** | 12.5× |
| `3d` | 3.14 s | 0.25 s | **0.23 s** | 13.4× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 383 ms | **13.1 ms** | 13.3 ms | 29.2× |
| `2d72` | 7.57 s | 0.73 s | **0.67 s** | 11.3× |
| `stack` | 42.81 s | 3.63 s | **3.48 s** | 12.3× |
| `3d` | 62.81 s | **5.13 s** | 7.13 s | 12.2× |

On this cluster every NUFFT plan creation costs 0.12–0.2 s, which is most of the NUFFT set-up
and of the small `2d` Toeplitz set-up.

:::

:::{tab-item} Apple M3 Ultra
:sync: m3ultra

Apple M3 Ultra, 512 GB unified memory, Metal backend. NUFFT: `aha_nufft_plan` on MLX with
mlx-nufft 0.3.3; `calc_toep` uses mlx-nufft and finishes in the library.

**Operator set-up**

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | **1.87 ms** | 10.6 ms | 10.0 ms |
| `2d72` | **2.82 ms** | 3.94 s | 3.94 s |
| `stack` | **2.48 ms** | 3.74 s | 3.62 s |
| `3d` | **24.8 ms** | 14.32 s | 14.13 s |

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 79.7 ms | **1.75 ms** | 2.03 ms | 45.5× |
| `2d72` | 1.05 s | 214 ms | **182 ms** | 5.8× |
| `stack` | 6.21 s | 0.92 s | **0.89 s** | 7.0× |
| `3d` | 19.90 s | 1.18 s | **1.10 s** | 18.1× |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 1.58 s | **0.040 s** | 0.045 s | 39.4× |
| `2d72` | 21.19 s | 5.03 s | **3.94 s** | 5.4× |
| `stack` | 127.3 s | **18.82 s** | 20.04 s | 6.8× |
| `3d` | 400.5 s | 25.56 s | **23.83 s** | 16.8× |

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

The previous runtime measurements, which used the library's NUFFT operator on the NVIDIA GPUs,
gave the same operator differences: 0.9–1.1·10⁻⁶ in 2D and stack-of-stars and
2.5·10⁻⁶ in 3D on the NVIDIA GPUs, and 1.1–1.6·10⁻⁶ and 1.9·10⁻⁵ on the M3 Ultra.

## Sources

The runtime tables were measured with `bench_aha.py` of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks)
[`762a4c0`](https://github.com/mikecjz/MR_AHA_benchmarks/tree/762a4c0) and MR_AHA
[`da3e148`](https://github.com/mikecjz/MR_AHA/tree/da3e148) on every GPU, on the synthetic
presets. The NUFFT operator is `nufft_native` (`aha_nufft_plan`), with cufinufft 2.5.1 (Python)
on the NVIDIA GPUs and mlx-nufft 0.3.3 on the M3 Ultra:

```bash
python bench_aha.py --problem 2d 2d72 stack 3d --methods toeplitz nufft_native --mask both \
    --cg 20 --setup-reps 3
```

Each tab comes from a single run, so its Toeplitz and NUFFT rows were measured together. The
NVIDIA runs also timed the library's own NUFFT operator (`--methods toeplitz nufft_c
nufft_native`), which is not in the tables. It is faster than `aha_nufft_plan` on most problems
(up to 2.7×, on the H100 `stack`), but slower in 3D on the RTX 5090 and the L40S and at rank 72
on the L40S. The RTX 5090 library was built with `-DCMAKE_CUDA_ARCHITECTURES=120`.

The accuracy table comes from the original study, which used measured human 2D radial and
stack-of-stars data and a synthetic kooshball. Its scripts and raw numbers are in MR_AHA's
history at commit [`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks).
