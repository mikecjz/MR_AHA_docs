# Comparison with BART and MRISubspaceRecon.jl

This study compares MR_AHA's low-rank normal operator $A^H A$ with two other GPU implementations
of the same operator:

* [BART](https://codeberg.org/mrirecon/bart), the Berkeley Advanced Reconstruction Toolbox (C and
  CUDA, with MATLAB and Python wrappers);
* [MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl) (Julia,
  CUDA.jl).

All three evaluate $A^H A$ with Toeplitz embedding: an FFT onto a 2× oversampled grid, an
$L \times L$ multiplication at every grid point, and an inverse FFT. They differ in how they store
and apply the kernel, in how they handle coils and stack-of-stars trajectories, and in what they
cost to set up.

## Summary

* **BART is 2.1–5.6× slower per $A^H A$ and needs 2–10× the GPU memory.** On a 24 GB GPU it
  fits only the smallest of the four standard problems, and fits reduced stack and 3D problems
  only in its low-memory mode. A 20-iteration CG reconstruction with `bart pics` is 1.9–5.5×
  slower than the same iterations with MR_AHA, and 3–9× slower end to end, because every call
  through BART's Python or MATLAB interface starts a new process, exchanges files and recomputes
  the kernel.
* **MRISubspaceRecon.jl is close to MR_AHA on 2D and 3D problems.** Its $A^H A$ is 1.05–1.9×
  slower than MR_AHA's masked Toeplitz operator on a TITAN RTX, an L40S and an H100, and its
  set-up is up to 15× slower. One problem is the exception: with $L = 72$ on the H100 it is 2.2×
  faster.
* **Stack-of-stars is where the difference is largest.** Neither library has a dedicated
  stack-of-stars operator, so both treat a stack as a 3D problem oversampled along $z$ as well.
  MR_AHA's stack operator is 2.6–2.8× faster than MRISubspaceRecon.jl's and 5.5× faster than
  BART's. MRISubspaceRecon.jl also cannot run the standard stack problem at all (see
  [Limits of MRISubspaceRecon.jl](#limits-of-mrisubspacereconjl)).
* **All three compute the same operator.** BART's full-kernel operator agrees with MR_AHA to
  about $10^{-5}$. MRISubspaceRecon.jl agrees to $2$–$5 \cdot 10^{-3}$, the error of its kernel
  mask, which cannot be switched off.

## What was compared

| Library | Version | Operator |
|---|---|---|
| MR_AHA | [`711003d`](https://github.com/mikecjz/MR_AHA/tree/711003d) | `mr_aha.create_op`, method `"toeplitz"` (unmasked and masked) |
| BART | v1.0.01 ([`381039c`](https://codeberg.org/mrirecon/bart/src/commit/381039c4)), built with CUDA | the normal operator of `bart pics` (`sense_nc_init()`), timed with a small C program linked against BART's libraries, and `bart pics` itself through `bart.py` |
| MRISubspaceRecon.jl | v0.10.1 ([`1e14dff`](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/tree/1e14dff48634ce8bb8222177e7bfd335a8ccf03d)), Julia 1.11.7, CUDA.jl 6.5.1 | `NFFTNormalOp` on `CuArray`s |

| GPU | Used for |
|---|---|
| NVIDIA TITAN RTX, 24 GB (sm_75), CUDA 12.6 | BART and MRISubspaceRecon.jl |
| NVIDIA L40S, 48 GB (sm_89), CUDA 12.9 | MRISubspaceRecon.jl |
| NVIDIA H100 80GB HBM3 (sm_90), CUDA 12.9 | MRISubspaceRecon.jl |

Every run used one GPU. MRISubspaceRecon.jl runs on one GPU only; BART was not tried on several.

### Problems

The problems are the synthetic presets of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks): golden-angle radial (2D and
stack-of-stars) and golden-means kooshball (3D) trajectories, random coil maps and random
coefficients, and a random Hermitian $L \times L$ Gram block per readout.

| Name | Matrix | $L$ | Coils | Samples |
|---|---|---:|---:|---|
| `2d` | 160² | 6 | 16 | 377 spokes × 160 |
| `2d72` | 320² | 72 | 12 | 754 spokes × 320 |
| `stack` | 320² × 56 | 12 | 12 | 280 spokes × 320 × 56 $k_z$ |
| `3d` | 160³ | 12 | 12 | 15680 spokes × 160 |

BART does not fit the full `2d72`, `stack` and `3d` presets in 24 GB, so its tables use reduced
versions: `2d72` with $L = 24$, a 160² × 28 stack with 140 spokes, and a 96³ kooshball with 5645
spokes.

### From Gram blocks to a temporal basis

MR_AHA takes one Gram block $\Phi_2$ per readout. BART and MRISubspaceRecon.jl take a temporal
basis instead: one row of $L$ coefficients per time frame, and one trajectory per time frame. The
benchmarks convert one into the other exactly. Each Gram block is eigen-decomposed,
$\Phi_2 = \sum_k u_k u_k^H$, and each readout becomes $r$ time frames that share its trajectory,
with basis rows $u_k$ ($r = 2$ for the synthetic blocks). The conversion is not timed. It
doubles the number of non-uniform points from which the other two libraries compute their
kernel, so it adds to their set-up time but not to $A^H A$.

## BART

All BART numbers are from the TITAN RTX. Times are the best median of 2–3 runs; set-up includes
the kernel (BART calls it the point-spread function, PSF).

### One $A^H A$

| Problem | MR_AHA Toeplitz | BART, full PSF | BART, upper-triangular PSF | BART, upper-triangular, low memory |
|---|---:|---:|---:|---:|
| `2d` | **2.20 ms** (0.05 GB) | 4.70 ms (0.32 GB) | 5.14 ms (0.32 GB) | 5.22 ms (0.17 GB) |
| `2d72`, $L = 24$ | **44 ms** (1.3 GB) | 148 ms (12.3 GB) | 156 ms (7.1 GB) | 155 ms (3.9 GB) |
| stack 160² × 28 | **133 ms** (2.8 GB) | out of memory | out of memory | 751 ms (10.9 GB) |
| 3D 96³ | **387 ms** (6.8 GB) | out of memory | out of memory | 829 ms (13.1 GB) |
| full `2d72`, `stack`, `3d` | 673 ms / 1012 ms / out of memory | out of memory | out of memory | out of memory |

* BART's upper-triangular PSF saves memory, not time.
* The machine was shared with other users. On the larger problems, run-to-run times varied by
  up to 2.5× for both libraries; the table gives the best runs.

### 20 CG iterations

`bart pics -g` was run through BART's own Python interface (`bart.py`) with 1 and 21
iterations; the difference is the time of 20 iterations. BART's CG has no stopping tolerance by
default, so every run does all its iterations. MR_AHA ran 20 plain CG iterations on
$A^H A\,u = b$ with the same operator as above. "End to end" is, for MR_AHA, copying the inputs to
the GPU, creating the operator and the 20 iterations, and for BART the whole `pics` call: process
start, file I/O, PSF, scaling, adjoint and the 20 iterations.

| Problem | MR_AHA Toeplitz CG | BART CG | Ratio | MR_AHA end to end | BART end to end | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `2d` | **0.055 s** | 0.105 s (full PSF) | 1.9× | **0.10 s** | 0.91 s | 9× |
| `2d72`, $L = 24$ | **0.88 s** | 3.03 s (full PSF) | 3.4× | **1.26 s** | 6.64 s | 5.3× |
| stack 160² × 28 | **2.47 s** | 13.7 s (low memory) | 5.5× | **3.44 s** | 27.3 s | 7.9× |
| 3D 96³ | **6.32 s** | 16.9 s (low memory) | 2.7× | **10.5 s** | 32.8 s | 3.1× |

* The CG ratio follows the $A^H A$ ratio, since both libraries spend almost all their CG time
  in $A^H A$.
* The end-to-end ratio is larger. BART's MATLAB and Python interfaces run the `bart` executable
  and exchange `.cfl` files, so an operator cannot stay in GPU memory between calls, and every
  call recomputes the PSF.
* The `2d` BART CG time is the difference of two calls of about 1 s each and varies by about
  ±30 % between runs.

### Accuracy

After the best complex scalar (BART normalises its FFT differently), BART's full-PSF operator
agrees with MR_AHA's Toeplitz operator to $7 \cdot 10^{-6}$–$1.6 \cdot 10^{-5}$.

```{note}
In BART v1.0.01 the upper-triangular PSF multiplies by the complex conjugate of the Gram blocks
that the full PSF applies.
```

## MRISubspaceRecon.jl

"Masked" is MR_AHA's masked Toeplitz operator, the closest match: MRISubspaceRecon.jl always
masks its kernel. Times are medians of 10 applications after 2 warm-up runs; set-up is the median
of the warm creations (the first creation in Julia also compiles, about 1.7 s for `2d`).

### One $A^H A$

$A^H A$ in ms, set-up in s in brackets.

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

| Problem | MRISubspaceRecon.jl | MR_AHA masked | MR_AHA Toeplitz |
|---|---:|---:|---:|
| `2d` | 3.05 (0.03) | **2.16** (0.01) | 2.22 |
| `2d72` | 802 (5.68) | **583** (2.16) | 674 |
| `stack` | fails | **980** (3.25) | 1011 |
| `stack` with 32 slices | 1489 (31.7) | **560** (2.06) | 577 |
| `3d` | 1685 (36.1) | **1378** (19.4) | 1603 |

:::

:::{tab-item} L40S
:sync: l40s

| Problem | MRISubspaceRecon.jl | MR_AHA masked | MR_AHA Toeplitz |
|---|---:|---:|---:|
| `2d` | 0.82 (0.02) | **0.43** (0.21) | 0.43 |
| `2d72` | 362 (4.07) | **288** (2.31) | 309 |
| `stack` | fails | **744** (2.59) | 752 |
| `stack` with 32 slices | 1140 (19.3) | **412** (1.64) | 417 |
| `3d` | 1348 (21.4) | **1016** (7.34) | 1063 |

:::

:::{tab-item} H100
:sync: h100

| Problem | MRISubspaceRecon.jl | MR_AHA masked | MR_AHA Toeplitz |
|---|---:|---:|---:|
| `2d` | 0.89 (0.02) | **0.54** (0.21) | 0.53 |
| `2d72` | **76** (4.65) | 165 (2.11) | 335 |
| `stack` | fails | **170** (1.84) | 178 |
| `stack` with 32 slices | 272 (5.26) | **96** (1.05) | 100 |
| `3d` | 317 (6.15) | **231** (4.75) | 252 |

:::

::::

* On 2D and 3D problems MRISubspaceRecon.jl is 1.2–1.9× slower than MR_AHA's masked operator
  (1.05–1.4× slower than the unmasked one on the TITAN RTX).
* On the stack it is 2.6–2.8× slower: it has no stack-of-stars operator (see below).
* `2d72` on the H100 is the exception, where MRISubspaceRecon.jl is 2.2× faster. MR_AHA's
  Toeplitz operator gains almost nothing from the L40S to the H100 on this problem
  ($L = 72$), while every other problem runs 3–4× faster on the H100. $L = 72$ is above the largest rank (30) for which MR_AHA has
  a block-diagonal kernel specialised on $L$, so this problem runs MR_AHA's generic block kernel.
* MRISubspaceRecon.jl's set-up is 1.3–15× slower, except on the small `2d` problem on the L40S
  and the H100, where MR_AHA's set-up takes 0.21 s against 0.02 s.
* Its kernel memory is close to MR_AHA's masked kernel (`3d`: 18.5 GB, against 15.1 GB masked
  and 24.0 GB unmasked for MR_AHA on the TITAN RTX).

### Accuracy

The relative difference to an exact reference $A^H A$, applied without a Toeplitz kernel, is the
same on all three GPUs:

| Problem | MR_AHA Toeplitz | MR_AHA masked | MRISubspaceRecon.jl |
|---|---:|---:|---:|
| `2d` | 9.4e-7 | 3.0e-3 | 1.9e-3 |
| stack 64² × 16, $L = 4$, 4 coils | 1.2e-6 | 1.1e-2 | 4.6e-3 |
| 3D 48³, $L = 4$, 4 coils | 1.4e-6 | 1.0e-2 | 3.4e-3 |

The best-fit scalar is 0.99999 in every case, so both libraries compute the same unnormalised
operator with the same sign and Gram-block conventions. The remaining difference is the mask.
MRISubspaceRecon.jl keeps the grid points within its Gaussian spreading kernel's reach of a
sample; MR_AHA's default mask (`calc_mask`, `mask_radius = 2`) keeps a similar fraction chosen
differently (2D: 82 % against 81 %, 3D: 57 % against 54 % of the grid) and is less accurate on
these problems. The random input is a worst case for any mask.

(limits-of-mrisubspacereconjl)=
### Limits of MRISubspaceRecon.jl

The following describes MRISubspaceRecon.jl at commit
[`1e14dff`](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/tree/1e14dff48634ce8bb8222177e7bfd335a8ccf03d)
(v0.10.1).

* **No dedicated stack-of-stars support.** A stack must be given as a 3D trajectory, with
  $k_z$ as a third coordinate. The operator then oversamples $z$ by 2 like $x$ and $y$: the FFTs
  run on a grid with twice the slices, and the kernel is twice as large. MR_AHA's `"stack"`
  trajectory type keeps $M_z = N_z$ and needs no FFT along $z$ for the kernel. This is why its
  stack operator is 2.6–2.8× faster.
* **Grid sizes must have only the prime factors 2, 3 and 5.** Every dimension of the 2×
  oversampled grid must be of the form $2^a 3^b 5^c$ (`nextprod((2, 3, 5), n)` in
  `calculate_kmask_indcs`), a requirement of NonuniformFFTs.jl, which it uses.
  The standard `stack` preset has 56 slices, so its oversampled grid has 112 = 2⁴ · 7 slices and
  the operator cannot be created. With 32 slices it runs. Image sizes such as 56, 112 or 224
  cannot be used without padding the images first.
* **The kernel is always masked.** The operator keeps only the grid points that the trajectory
  reaches through a Gaussian spreading kernel. There is no option to switch this off, so the
  operator is an approximation ($2$–$5 \cdot 10^{-3}$ on these problems).
* **One GPU.** The CUDA code never selects a device; there is no multi-GPU mode.
* **Slower set-up.** The kernel is computed one triangular entry at a time.

## Reproduce

The scripts are in [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks):
`bart/` and `julia/`, each with a README that describes the set-up, every option and the
results.

::::{tab-set}

:::{tab-item} BART
:sync: bart

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks
bart/build_bart.sh                      # BART with CUDA; needs GCC 12 or newer
python bart/bench_bart.py --problem 2d --check
python bart/bench_bart.py --problem 2d --methods bart_pics_cg mr_aha_toeplitz --cg 20
```

:::

:::{tab-item} MRISubspaceRecon.jl
:sync: julia

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python save_problem.py 2d data/2d.mat
julia -t 8 --project=. bench_aha.jl --data data/2d.mat
```

:::

::::
