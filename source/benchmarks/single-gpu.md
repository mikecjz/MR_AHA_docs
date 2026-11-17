# Single GPU

One GPU, the Toeplitz operator of MR_AHA (`mr_aha.create_op`) against the NUFFT operator with
the same interface, which applies a type-2 and a type-1 NUFFT in every call (`aha_nufft_plan`,
or `create_op(..., method='nufft')`). The NUFFT numbers are those of the NUFFT library's default
settings.

## Summary

* **Toeplitz is faster than NUFFT on every problem and every GPU.** One $A^H A$ is 12–59× faster
  for the small 2D problem, 5–6× faster at rank 72, 3.9–7.3× faster for stack-of-stars and 3.7–18×
  faster for the 3D kooshball.
* **The fastest GPU is the H100**: 0.53 ms (2D), 0.17 s (stack) and 0.23 s (3D) per $A^H A$.
* **The Apple M3 Ultra runs the large problems at about the speed of an L40S** (0.84 s for the
  stack, 1.10 s in 3D), and its 512 GB of unified memory hold any of the kernels.
* **Both operators compute the same thing**: they agree to 1–4·10⁻⁶ per application, and the
  20-iteration CG solutions to 3–6·10⁻⁶ (see [Accuracy](#accuracy)).

## Problems

| name | size | coils | $L$ | samples |
|---|---|---:|---:|---|
| `2d` | 160² | 16 | 6 | 377 radial spokes × 160 |
| `2d72` | 320² | 12 | 72 | 754 radial spokes × 320 |
| `stack` | 320² × 56 | 12 | 12 | 280 spokes × 320, × 56 $k_z$ |
| `3d` | 160³ | 12 | 12 | 15680 kooshball spokes × 160 |

**Timing.** Wall time with device synchronisation, after a warm-up call; the median of 5 to 20
calls. "Masked" is the masked Toeplitz operator (`is_masked = true`), which keeps 54–83 % of the
upsampled grid. CG is plain conjugate gradient from zero, exactly 20 iterations.

## Runtime

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

NVIDIA TITAN RTX, 24 GB (Turing, sm_75), CUDA 12.6, cufinufft 2.5.1.

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 35.6 ms | **2.1 ms** | 2.1 ms | **17×** |
| `stack` | 5.23 s | 0.99 s | **0.96 s** | **5.5×** |
| `3d` | 11.85 s | 1.62 s | **1.44 s** | **8.2×** |

**20 CG iterations** (one-off kernel computation `calc_toep` in brackets)

| problem | Toeplitz | Toeplitz masked |
|---|---:|---:|
| `2d` | **0.05 s** (+0.03 s) | 0.06 s (+0.02 s) |
| `stack` | **27.6 s** (+2.9 s) | 27.9 s (+3.2 s) |
| `3d` | does not fit | **35.1 s** (+20.3 s) |

The CG runs were made before the current CUDA kernels; with them, one $A^H A$ takes 15–31 % less
time (the first table), and the unmasked 3D kernel fits in 24 GB.

:::

:::{tab-item} L40S
:sync: l40s

NVIDIA L40S, 48 GB (Ada, sm_89), CUDA 12.9, cufinufft 2.5.1.

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 25.4 ms | 0.44 ms | **0.43 ms** | **59×** |
| `2d72` | 434 ms | 90 ms | **86 ms** | **5.0×** |
| `stack` | 2.88 s | 0.75 s | **0.74 s** | **3.9×** |
| `3d` | 3.79 s | 1.06 s | **1.02 s** | **3.7×** |

**20 CG iterations** (`calc_toep` in brackets)

| problem | Toeplitz | Toeplitz masked |
|---|---:|---:|
| `2d` | **0.02 s** (+0.17 s) | 0.02 s (+0.28 s) |
| `stack` | **19.1 s** (+2.9 s) | 20.3 s (+2.7 s) |
| `3d` | **26.8 s** (+7.6 s) | 27.4 s (+7.1 s) |

The CG runs were made before the current CUDA kernels, which take 20–35 % less time per
$A^H A$ (the first table).

On this system every first-time plan or allocation costs about 0.1–0.15 s, which is most of the
small `2d` set-up times.

:::

:::{tab-item} H100
:sync: h100

NVIDIA H100 80GB (SXM, sm_90), CUDA 12.9.

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 6.62 ms | **0.53 ms** | 0.54 ms | **12×** |
| `2d72` | 201 ms | 36 ms | **33 ms** | **6.1×** |
| `stack` | – | 178 ms | **170 ms** | |
| `stack` with 48 slices | 685 ms | **154 ms** | – | **4.4×** |
| `3d` | 2052 ms | 253 ms | **232 ms** | **8.8×** |

**20 CG iterations**

| problem | NUFFT | Toeplitz | Toeplitz speed-up |
|---|---:|---:|---:|
| `2d72` | 4.03 s | **0.73 s** | 5.5× |
| `stack` with 48 slices | 13.8 s | **3.13 s** | 4.4× |
| `3d` | 41.0 s | **5.10 s** | 8.0× |

Creating the masked operator (mask and kernel) takes 0.21 s (`2d`), 2.0 s (`2d72`), 1.75 s
(`stack`) and 4.8 s (`3d`).

:::

:::{tab-item} Apple M3 Ultra
:sync: m3ultra

Apple M3 Ultra, 512 GB unified memory, Metal backend. The NUFFT operator is MLX with
mlx-nufft; `calc_toep` uses mlx-nufft and finishes in the library.

**One $A^H A$**

| problem | NUFFT | Toeplitz | Toeplitz masked | Toeplitz speed-up |
|---|---:|---:|---:|---:|
| `2d` | 78.9 ms | **1.7 ms** | 2.0 ms | **46×** |
| `stack` | 6.15 s | **0.84 s** | 0.89 s | **7.3×** |
| `3d` | 19.9 s | 1.17 s | **1.10 s** | **18×** |

**20 CG iterations** (`calc_toep` in brackets)

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---:|---:|---:|
| `2d` | 1.57 s | **0.04 s** (+0.03 s) | 0.05 s (+0.01 s) |
| `stack` | 127.4 s | **20.1 s** (+4.9 s) | 21.6 s (+3.6 s) |
| `3d` | 398.5 s | 25.4 s (+15.2 s) | **24.6 s** (+14.0 s) |

The CG vector updates run on the CPU here. The `2d` problem is too small to occupy the GPU.

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
  [Neighbourhood mask](neighbour-mask.md).

## Sources

The NUFFT times on the TITAN RTX and the L40S, and the CG times and all numbers on the M3 Ultra,
come from the original study, which used measured human 2D radial and stack-of-stars data and a
synthetic kooshball. Its scripts and raw numbers are in MR_AHA's history at commit
[`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks). The other numbers were
measured with the current CUDA kernels on the synthetic presets of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks), which have the same sizes.
