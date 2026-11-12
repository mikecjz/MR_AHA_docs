# Main study: Toeplitz vs. NUFFT $A^HA$

This study benchmarks the speed and accuracy of the Toeplitz-embedded operator of MR_AHA
(`mr_aha.create_op` in Python) against the NUFFT operator, which applies type-2 and type-1
cufinufft transforms in every call (`mr_aha.aha_nufft_plan` in Python, `aha_nufft_plan` / `AhA_cufinufft` in MATLAB;
inside the library: `create_op(..., method='nufft')`).

> **Reading the Toeplitz numbers.** Two things are better in the current library than in these
> tables:
>
> * **Speed.** These runs used the generic block-diagonal kernels, one coil at a time. The
>   default kernels are 15–50 % faster on one GPU; see
>   [CUDA operator optimisation and multi-GPU](cuda-multigpu.md).
> * **Masks.** The masks of this study kept 83 % (2D), 81 % (stack) and 64 % (3D) of the upsampled
>   grid. `calc_mask` with its default `mask_radius = 2` keeps 80 %, 79 % and 54 % on the same
>   trajectories; see [Neighbourhood mask](neighbour-mask.md). The kernel size is proportional to
>   the kept fraction.

Everything was run on two machines:

| machine | GPU | software |
|---|---|---|
| local workstation `pact400-recon3` | NVIDIA TITAN RTX, 24 GB (Turing, sm_75) | CUDA 12.6 (nvcc), CuPy 14.2, cufinufft 2.5.1 (pip) for Python, cufinufft 2.6-dev (built from the `finufft/` submodule) for MATLAB R2025b |
| HPC node `esplhpc-cp075` | NVIDIA L40S, 48 GB (Ada, sm_89) | CUDA 12.9 module, CuPy 14.2, cufinufft 2.5.1 (pip) for Python, the cufinufft MEX bundled in the `matlab_biri_v1.6` Singularity image for MATLAB R2025b |

Full per-machine tables, including every intermediate number, are in
[`results/pact400-recon3/summary.md`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/results/pact400-recon3/summary.md) and
[`results/esplhpc-cp075/summary.md`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/results/esplhpc-cp075/summary.md). The raw numbers are in the
JSON files next to them.

## Summary

* **Toeplitz is faster than the best-tuned NUFFT operator in every case.** One $A^HA$ application
  is **8–20×** faster in 2D, **1.8–2.3×** faster for the stack-of-stars data, and **2.5×** (L40S)
  to **5.1×** (TITAN RTX) faster for the 3D kooshball. The 20-iteration CG reconstructions speed up
  by the same factors, and they still win comfortably once the one-off `calc_toep` time is included.
* **Both forms compute the same operator.** On image-like inputs the Toeplitz and NUFFT operators
  agree to 1–4·10⁻⁶. The 20-iteration CG solutions agree to 3.4–3.6·10⁻⁶ in 2D and stack. The
  MATLAB and Python NUFFT solutions agree with each other to 2.7–3.0·10⁻⁶.
* **The 3D CG difference of 1.5–2.5·10⁻⁴ is a numerical floor, not a Toeplitz error.** The 3D
  problem is badly conditioned (15680 spokes for a 160³ matrix), so CG amplifies single-precision
  rounding. Two runs of the *same* NUFFT operator that differ only in cufinufft's spreading method
  differ by 3.6·10⁻⁶ per application and by **3.5·10⁻⁴** after 20 CG iterations. That is more than
  the Toeplitz-vs-NUFFT difference.
* **MATLAB and Python NUFFT run at about the same speed.** MATLAB is 5–15 % faster on stack and 3D;
  its cufinufft build is newer, since the Python wheel is 2.5.1. Python is 12–17 % faster in 2D.
  Selecting cufinufft's `gpu_method` per transform type matters more than the language: it is worth
  1.4–1.9× in 2D and stack. The default settings are always reported alongside the tuned ones.
* **Masking saves memory.** The masks of this study kept 83 % (2D), 81 % (stack) and 64 % (3D) of
  the upsampled grid. The kernel shrinks by 17–36 %, which is what lets the 3D problem run on the
  24 GB TITAN at all: unmasked it needs 19 GB of kernel. The CG solution moves by only 2·10⁻⁵.
  In these runs a masked application takes slightly longer than an unmasked one. With the default
  in-place masked kernel it is 3–11 % faster; see [CUDA operator optimisation](cuda-multigpu.md).
* **For a sparse trajectory the mask saves far more, at a price in accuracy.** With a two-arm
  spiral on a 256² matrix, a radius of 2 keeps only 4 % of the grid and a radius of 8 keeps 14 %.
  The operator error is then 3·10⁻² and 3·10⁻³, not 10⁻⁶. See
  [Neighbourhood mask](neighbour-mask.md).

## Problems

| name | data | size | coils | L | samples | CG right-hand side |
|---|---|---|---|---|---|---|
| `2d` | `data/human_C-35_2D_radial.mat` | 160² | 16 | 6 | 377 spokes × 160 | stored `AhAx_ref` = $A^HA\,x$ (CPU finufft), so the truth is the stored `x` |
| `stack` | `data/AhAx_test_toeplitz.mat` | 320² × 56 | 12 | 12 | 280 spokes × 320, × 56 kz | stored `Ahb` |
| `3d` | `data/kooshball_3d.mat`, synthetic ([`make_kooshball.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/make_kooshball.py)) | 160³ | 12 | 12 | 15680 spokes × 160 | $A^HA\,x_\text{true}$ for a random-ellipsoid `x_true` |

The synthetic 3D problem uses golden-means kooshball directions, smooth complex Gaussian coil
maps, and a smooth complex orthonormal temporal basis $U$ (the rank × time basis is $U^T$). Each
spoke is acquired at one time point, so its Gram block is the rank-one $u_t\,\overline{u_t}^T$
(see [Gram blocks](../guide/conventions.md#gram-blocks)).

**Gram-block convention.** All problems follow the library's convention: `phi2` is
$\Phi\Phi^H$ of a rank × time basis and multiplies the rank row vector from the right. The `2d`
file predates this convention (it holds the transposed blocks), so the loaders swap its two $L$
axes; so do they for a `kooshball_3d.mat` written before the change (no `gram_convention`
variable). The stack `Phi2` is `Phi_rt * Phi_rt'` and is used as is.

**Stack data convention.** `Phi2` is stored as `[L, L, Ntraj, Nz]`. Its trace over kz peaks at
indices 0 and 55 and is smallest around 28, so kz is in *natural* FFT order (DC first). The loaders
permute it to `[Ntraj, Nz, L, L]` and `fftshift` it along kz into the centred order the library
expects. Both operators use the same convention, so the comparison is unaffected either way, but
the reconstruction is only physically right with the shift.

## Method

* **NUFFT operator.** Per coil: sensitivity → type-2 cufinufft (sign −1) → one $L\times L$ block
  per readout → type-1 cufinufft (sign +1) → conjugate sensitivity. All `L` ranks (× `Nz` kz planes
  for the stack) go through one batched plan, and the plans and point sorting are built once. The
  settings are `eps = 1e-6`, `upsampfac = 2`, single precision.
  * For the stack, the centred z-FFT is a plain FFT applied to the image after modulating it by
    $e^{2\pi i c n/N_z}$, folded into the coil map. The $k$-side phases cancel because the Gram
    blocks are diagonal in kz, so no `fftshift` copies are made.
  * The block-diagonal product uses `pagemtimes` in MATLAB (4 ms vs 16 ms for a broadcast-and-sum
    on the 3D data) and `einsum` in CuPy.
* **Tuning.** "Tuned" picks the fastest (type-1, type-2) `gpu_method` pair out of {1, 2}² for each
  dataset and machine.
* **Toeplitz operator.** The kernel is computed once (`calc_toep`) and then applied repeatedly,
  which is what `mr_aha.create_op` does. "Masked" uses the masks described in the note at the top
  of this page.
* **Timing.** Wall time with device synchronisation, after a warm-up call. The median of 20 (2D)
  or 5 (stack, 3D) calls is reported.
* **CG.** Plain conjugate gradient from $u_0 = 0$, exactly 20 iterations, no stopping test. The
  algorithm is identical in Python (`common.cg`) and MATLAB (`bench_matlab.m`). All solutions are
  compared pairwise as $\|a-b\|/\|b\|$.

## One $A^HA$ application (median)

| problem | GPU | NUFFT Py default / tuned | NUFFT MATLAB default / tuned | Toeplitz | Toeplitz masked | Toeplitz vs best NUFFT |
|---|---|---|---|---|---|---|
| 2d | TITAN RTX | 35.6 / 19.0 ms | 42.8 / 23.0 ms | **2.3 ms** | 2.6 ms | **8.2×** |
| 2d | L40S | 25.4 / 13.4 ms | 29.6 / 15.3 ms | **0.67 ms** | 0.79 ms | **19.9×** |
| stack | TITAN RTX | 5.23 / 3.64 s | 4.36 / 3.16 s | **1.36 s** | 1.38 s | **2.3×** |
| stack | L40S | 2.88 / 1.84 s | 2.55 / 1.71 s | **0.94 s** | 1.00 s | **1.8×** |
| 3d | TITAN RTX | 11.85 / 9.45 s | 10.48 / 8.97 s | does not fit (19 GB kernel) | **1.74 s** | **5.1×** |
| 3d | L40S | 3.79 / 3.79 s | 3.94 / 3.39 s | **1.33 s** | 1.36 s | **2.5×** |

One-off pre-computation (`calc_toep`):

| problem | TITAN RTX | L40S | kernel size, unmasked → masked |
|---|---|---|---|
| 2d | 0.03 s | 0.17 s | 16 → 14 MB |
| stack | 2.9 s | 2.9 s | 13.3 → 10.8 GB |
| 3d | 20.3 s (masked only) | 7.6 s | 19.0 → 12.1 GB |

On the HPC node every first-time plan or allocation carries about 0.1–0.15 s of overhead, which
dominates its small 2D pre-computation times. Local plan creation takes milliseconds.

## CG, 20 iterations from zero

| problem | GPU | NUFFT Py | NUFFT MATLAB | Toeplitz (+ `calc_toep`) | Toeplitz masked (+ mask + `calc_toep`) |
|---|---|---|---|---|---|
| 2d | TITAN RTX | 0.40 s | 0.50 s | **0.05 s** (+0.03) | 0.06 s (+0.02) |
| 2d | L40S | 0.31 s | 0.51 s | **0.02 s** (+0.17) | 0.02 s (+0.28) |
| stack | TITAN RTX | 74.1 s | 64.2 s | **27.6 s** (+2.9) | 27.9 s (+3.2) |
| stack | L40S | 37.1 s | 34.4 s | **19.1 s** (+2.9) | 20.3 s (+2.7) |
| 3d | TITAN RTX | 184.5 s | 175.3 s | does not fit | **35.1 s** (+20.3) |
| 3d | L40S | 76.0 s | 68.6 s | **26.8 s** (+7.6) | 27.4 s (+7.1) |

Relative differences between the CG solutions; both machines agree to the digits shown:

| problem | Toeplitz vs NUFFT (Py) | Toeplitz vs NUFFT (MATLAB) | NUFFT Py vs MATLAB | masked vs unmasked Toeplitz | error vs truth (all operators) |
|---|---|---|---|---|---|
| 2d | 3.4·10⁻⁶ | 3.4–3.5·10⁻⁶ | 2.7–2.8·10⁻⁶ | 1.8·10⁻⁵ | 8.355·10⁻² |
| stack | 3.6·10⁻⁶ | 3.6·10⁻⁶ | 3.0·10⁻⁶ | 2.7·10⁻⁵ | – |
| 3d | 1.7·10⁻⁴ (TITAN, masked), 2.5·10⁻⁴ (L40S) | 2.5·10⁻⁴ (L40S) | 1.7·10⁻⁵ (TITAN), 7.7·10⁻⁵ (L40S) | 2.3·10⁻⁵ | 2.483·10⁻¹ |

The final relative residuals agree to three or more digits in every case (for example
1.768·10⁻² for all four stack solutions). The errors against the truth are identical, so the
choice of operator has no visible effect on reconstruction quality.

**The 3D floor.** At the operator level the 3D Toeplitz and NUFFT differ by 3.8·10⁻⁶, the same
order as in 2D. The CG difference is about 50× larger because the 20 iterations of a poorly
conditioned problem amplify any 10⁻⁶-level perturbation. To check this, CG was rerun with the NUFFT
operator using spreading methods (2,2) instead of (1,2). That solution differs from the (1,2)
solution by **3.5·10⁻⁴**, although the two operators differ by only 3.6·10⁻⁶. Raising the NUFFT
accuracy to its single-precision limit (`eps` = 1.2·10⁻⁷) moves the operator by another 2·10⁻⁶. The
Toeplitz solution therefore lies within the spread of the NUFFT solutions themselves.


## Apple silicon (Metal backend)

The same benchmark on a Mac Studio (Apple M3 Ultra, 512 GB unified memory) with the Metal backend
is in [`results/AW650523HJY/summary.md`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/results/AW650523HJY/summary.md). In short:

* **The Toeplitz operator is faster than on the L40S** for the large problems: one $A^HA$ takes
  0.84 s (stack) and 1.17 s (3D) against 0.94 s and 1.33 s, and the unmasked 3D kernel (20 GB)
  fits in unified memory. 2D is too small to matter (1.7 ms).
* **Same reconstructions.** The 20-iteration CG solutions agree with the CUDA ones to 5-10e-6 in
  2D and stack and 5e-5 in 3D (within its CG floor), and match the ground truth equally well.
* **Pre-computation is slower** (mlx-nufft in Python): `calc_toep` 4.9 s (stack) and 15 s (3D)
  against 2.9 s and 7.6 s on the L40S. In 3D the single-precision mlx-nufft kernel gives an
  operator error of 1.4e-5 (CUDA 1.9e-6); `nufft='finufft'` reduces it to 5e-6 at 4x the time.
* **The NUFFT reference is 3-5x slower** than tuned cufinufft; it serves only for validation.

## Recommendations

* **Use the Toeplitz operator for iterative reconstruction.** It wins at every size tested, even
  counting `calc_toep`. Keep the NUFFT operators (`AhA_cufinufft`, `aha_nufft_plan`) as an independent
  reference.
* **Use `is_masked = true` when memory is tight.** In 3D the kernel is the largest allocation.
  `calc_mask` with its default radius 2 cuts it by almost half (54 % of the grid kept) at the
  accuracy of the unmasked operator. The masked operator is also 3–11 % faster than the unmasked
  one, because its kernel works in place on the padded grid; see
  [CUDA operator optimisation](cuda-multigpu.md).
* **Sparse trajectories (few spiral arms, few shots): pick `mask_radius` for the accuracy you
  need.** On the two-arm spiral, radius 2 keeps 4 % of the grid at 3·10⁻² operator error and
  radius 8 keeps 14 % at 3·10⁻³; see [Neighbourhood mask](neighbour-mask.md). Also pass a small
  `mem_fraction` so that `calc_toep`'s transient memory stays close to the masked kernel's size.
* **When using the NUFFT path, tune `gpu_method`.** Pass `type1_opts` / `type2_opts`; on these
  problems (1, 2) or (1, 1) was best except for 3D on the L40S, where (2, 2) won.
