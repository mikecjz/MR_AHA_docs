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

* **MR_AHA is the fastest of the three on every problem and GPU measured.**
* **BART is 2.1–7.8× slower per $A^H A$ and needs 1.2–3.2× the GPU memory on the larger
  problems, in its most frugal mode.** Its two other modes ask for single blocks of tens to hundreds of GB on the stack and
  3D problems. It cannot hold the standard stack problem on any of the three GPUs, not even in
  80 GB. A 20-iteration CG reconstruction with `bart pics` is 2.6–7.7× slower than the same
  iterations with MR_AHA, and 2.8–11× slower end to end, because every call through BART's
  Python or MATLAB interface starts a new process, exchanges files and recomputes the kernel.
* **MRISubspaceRecon.jl is 1.2–4.2× slower than MR_AHA's masked operator on 2D and 3D
  problems**, and its set-up is up to 17× slower.
* **Stack-of-stars is where the difference is largest.** Neither library has a dedicated
  stack-of-stars operator, so both treat a stack as a 3D problem oversampled along $z$ as well.
  MR_AHA's stack operator is 2.7–2.8× faster than MRISubspaceRecon.jl's and 5.6–6.3× faster than
  BART's. MRISubspaceRecon.jl also cannot run the standard stack problem at all (see
  [Limits of MRISubspaceRecon.jl](#limits-of-mrisubspacereconjl)).
* **High rank used to be the exception and no longer is.** At $L = 72$ MR_AHA is 2.3–4.2× faster
  than MRISubspaceRecon.jl and 7.7–7.8× faster than BART, with its
  [tiled kernels for ranks above 30](../background/performance.md#25-ranks-above-30-tiled-kernels).
  Before them, MRISubspaceRecon.jl was 2.2× and BART 1.2× faster than MR_AHA on the H100.
* **MRISubspaceRecon.jl computes the same operator up to its kernel mask.** It agrees with
  MR_AHA to $2$–$5 \cdot 10^{-3}$, the error of the mask, which cannot be switched off.

## What was compared

| Library | Version | Operator |
|---|---|---|
| MR_AHA | [`c60db28`](https://github.com/mikecjz/MR_AHA/tree/c60db28) | `mr_aha.create_op`, method `"toeplitz"` (unmasked and masked) |
| BART | v1.0.01 ([`381039c`](https://codeberg.org/mrirecon/bart/src/commit/381039c4)), built with CUDA | the normal operator of `bart pics` (`sense_nc_init()`), timed with a small C program linked against BART's libraries, and `bart pics` itself through `bart.py` |
| MRISubspaceRecon.jl | v0.10.1 ([`1e14dff`](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/tree/1e14dff48634ce8bb8222177e7bfd335a8ccf03d)), Julia 1.11.7, CUDA.jl 6.5.1 | `NFFTNormalOp` on `CuArray`s |

| GPU | BART | MRISubspaceRecon.jl |
|---|---|---|
| NVIDIA TITAN RTX, 24 GB (sm_75), CUDA 12.6 | compiled on the machine | measured on 2026-11-12, not repeated for this MR_AHA version |
| NVIDIA L40S, 48 GB (45 GB usable; sm_89), CUDA 12.9 | from a container image | same job and node as MR_AHA |
| NVIDIA H100 80GB HBM3 (sm_90), CUDA 12.9 | from a container image | same job and node as MR_AHA |

Every run used one GPU. MRISubspaceRecon.jl runs on one GPU only; BART was not tried on several.

The cluster with the L40S and the H100 has no compiler that BART accepts (it needs GCC 12 or
newer). BART therefore runs there from an Apptainer image, built from a
[fork](https://github.com/mikecjz/bart) that adds a `Dockerfile` to BART v1.0.01 and leaves its
C sources unchanged ([`mikecjz/bart`](https://hub.docker.com/r/mikecjz/bart) on Docker Hub, CUDA
12.6 runtime). MR_AHA always runs natively. The operator times are measured inside the process
and do not contain the start of the container; BART's end-to-end CG times do.

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

BART cannot hold all of these on any of the three GPUs. Its problems are therefore fitted to
each GPU: a preset that BART cannot hold even in its most frugal mode is shrunk step by step
until it fits (the stack loses slices, the kooshball matrix size, `2d72` ranks). Each GPU thus
gets the largest problem of the ladder that BART holds on it (BART's memory in brackets):

| Preset | TITAN RTX (24 GB) | L40S (45 GB) | H100 (80 GB) |
|---|---|---|---|
| `2d72` (320², L=72) | **L=64** (22.8 GB) | as it is (28.4 GB) | as it is (28.4 GB) |
| `stack` (320²×56, L=12) | **256²×28** (17.2 GB) | **320²×32** (30.7 GB) | **320²×48** (46.0 GB) |
| `3d` (160³, L=12) | **128³** (19.6 GB) | **144³** (27.9 GB) | as it is (38.3 GB) |

### From Gram blocks to a temporal basis

MR_AHA takes one Gram block $\Phi_2$ per readout. BART and MRISubspaceRecon.jl take a temporal
basis instead: one row of $L$ coefficients per time frame, and one trajectory per time frame. The
benchmarks convert one into the other exactly. Each Gram block is eigen-decomposed,
$\Phi_2 = \sum_k u_k u_k^H$, and each readout becomes $r$ time frames that share its trajectory,
with basis rows $u_k$ ($r = 2$ for the synthetic blocks). The conversion is not timed. It
doubles the number of non-uniform points from which the other two libraries compute their
kernel, so it adds to their set-up time but not to $A^H A$.

## BART

On the TITAN RTX, times are the faster of two runs on a shared machine. Set-up includes the
kernel (BART calls it the point-spread function, PSF). Memory is what a method holds on the GPU
after its warm-up.

BART's Toeplitz embedding splits the 2× oversampled grid into $2^d$ shifted grids of the image
size: 4 in 2D, 8 in 3D. By default all of them are in GPU memory at once. **Low-memory mode**
(`-U` in `bart pics`) processes them one after another instead, which needs less memory at
little cost in time here. The **upper-triangular PSF** stores only the upper triangle of each
Hermitian $L \times L$ kernel block.

```{note}
In BART v1.0.01 the upper-triangular PSF multiplies by the complex conjugate of the Gram blocks.
```

### One $A^H A$

On the problems fitted to each GPU.

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | Best BART / MR_AHA | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d` 160², L=6 | **2.16 ms** (0.05 GB) | 4.75 ms (0.32 GB) | 5.13 ms (0.32 GB) | 5.23 ms (0.17 GB) | 2.2× | 4.7 ms | 20 ms |
| `2d72` 320², L=64 | **175 ms** (7.2 GB) | out of memory | out of memory | 985 ms (22.8 GB) | 5.6× | 988 ms | 627 ms |
| `stack` 256²×28 | **302 ms** (7.2 GB) | out of memory | out of memory | 1714 ms (17.2 GB) | 5.7× | 1717 ms | 828 ms |
| `3d` 128³ | **740 ms** (16.1 GB) | out of memory | out of memory | 1913 ms (19.6 GB) | 2.6× | 1920 ms | 3795 ms |

:::

:::{tab-item} L40S
:sync: l40s

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | Best BART / MR_AHA | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d` 160², L=6 | **0.44 ms** (0.05 GB) | 1.52 ms (0.36 GB) | 2.42 ms (0.36 GB) | 1.24 ms (0.21 GB) | 2.8× | 5.1 ms | 13 ms |
| `2d72` 320², L=72 | **90 ms** (9.0 GB) | out of memory | out of memory | 705 ms (28.4 GB) | 7.8× | 710 ms | 434 ms |
| `stack` 320²×32 | **417 ms** (12.9 GB) | out of memory | out of memory | 2521 ms (30.7 GB) | 6.0× | 2537 ms | 913 ms |
| `3d` 144³ | **792 ms** (23.0 GB) | out of memory | out of memory | 2332 ms (27.9 GB) | 2.9× | 2339 ms | 4210 ms |

:::

:::{tab-item} H100
:sync: h100

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | Best BART / MR_AHA | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d` 160², L=6 | **0.53 ms** (0.05 GB) | 1.31 ms (0.38 GB) | 2.26 ms (0.39 GB) | 1.64 ms (0.23 GB) | 2.5× | 1.8 ms | 6.62 ms |
| `2d72` 320², L=72 | **36 ms** (9.0 GB) | out of memory | out of memory | 277 ms (28.4 GB) | 7.7× | 278 ms | 201 ms |
| `stack` 320²×48 | **154 ms** (19.3 GB) | out of memory | out of memory | 912 ms (46.0 GB) | 5.9× | 919 ms | 685 ms |
| `3d` 160³ | **253 ms** (31.5 GB) | out of memory | out of memory | 762 ms (38.3 GB) | 3.0× | 767 ms | 2052 ms |

:::

::::

* MR_AHA is 2.1–7.8× faster than BART's fastest mode, and holds less memory.
* On the large problems BART runs only with the upper-triangular PSF in low-memory mode.
* `bart pics` per CG iteration agrees with the operator time, so the C program measures what
  BART users get. On the tiny `2d` problem the per-iteration time through the container is not
  reliable, because the start of the container varies by more than the iterations take.

### The same problems on all three GPUs

The problems that BART holds on 24 GB, as a comparison between GPUs: MR_AHA Toeplitz against
BART's fastest mode, with the ratio in brackets.

| Problem | TITAN RTX | L40S | H100 |
|---|---:|---:|---:|
| `2d` 160², L=6 | 2.16 ms / 4.75 ms (2.2×) | 0.44 ms / 1.24 ms (2.8×) | 0.53 ms / 1.31 ms (2.5×) |
| `2d72` 320², L=24 | 43 ms / 150 ms (3.5×) | 24 ms / 94 ms (4.0×) | 9.00 ms / 25 ms (2.8×) |
| `stack` 160²×28 | 121 ms / 671 ms (5.6×) | 90 ms / 557 ms (6.2×) | 22 ms / 138 ms (6.3×) |
| `3d` 96³ | 313 ms / 829 ms (2.6×) | 232 ms / 684 ms (3.0×) | 50 ms / 169 ms (3.4×) |

### Out of memory

Without fitting, BART fails on these presets. Each failed entry is the first allocation BART
could not get on that GPU, from its own message, with nothing else on the GPU. A smaller GPU
fails on an earlier, smaller request, so the sizes differ between GPUs:

| Preset | GPU | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory |
|---|---|---:|---:|---:|---:|
| `2d72` | TITAN RTX | 229 ms (9.0 GB) | fails on a 18 GB request | fails on a 32 GB request | fails on a 8 GB request |
| `2d72` | L40S | 90 ms (9.0 GB) | fails on a 63 GB request | fails on a 32 GB request | 705 ms (28.4 GB) |
| `2d72` | H100 | 36 ms (9.0 GB) | fails on a 63 GB request | fails on a 32 GB request | 278 ms (28.4 GB) |
| `stack` | TITAN RTX | 1011 ms (18.5 GB) | fails on a 49 GB request | fails on a 26 GB request | fails on a 26 GB request |
| `stack` | L40S | 753 ms (22.6 GB) | fails on a 49 GB request | fails on a 213 GB request | fails on a 26 GB request |
| `stack` | H100 | 178 ms (22.6 GB) | fails on a 393 GB request | fails on a 213 GB request | fails on a 26 GB request |
| `3d` | TITAN RTX | out of memory | fails on a 35 GB request | fails on a 152 GB request | fails on a 19 GB request |
| `3d` | L40S | 1062 ms (31.5 GB) | fails on a 281 GB request | fails on a 152 GB request | fails on a 19 GB request |
| `3d` | H100 | 253 ms (31.5 GB) | fails on a 281 GB request | fails on a 152 GB request | 761 ms (38.3 GB) |

BART's full PSF and its upper-triangular PSF without low-memory mode ask for single blocks of
tens to hundreds of GB on the stack and 3D presets. MR_AHA's unmasked `3d` operator does not
fit the 24 GB GPU next to the other arrays of this comparison; its masked one does (see below).

### 20 CG iterations

`bart pics -g` was run through BART's own Python interface (`bart.py`) with 1 and 21 iterations;
the difference is the time of 20 iterations. BART's CG has no stopping tolerance by default, so
every run does all its iterations. BART's mode is the first that fits of: full PSF, low memory,
upper-triangular PSF with low memory. MR_AHA ran 20 plain CG iterations on $A^H A\,u = b$ with
the same operator as above. "End to end" is, for MR_AHA, copying the inputs to the GPU, creating
the operator and the 20 iterations, and for BART the whole `pics` call: process start, file I/O,
PSF, scaling, adjoint and the 20 iterations.

| GPU | Problem | MR_AHA Toeplitz CG | BART `pics` CG | Ratio | MR_AHA end to end | BART end to end | Ratio | MR_AHA NUFFT CG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TITAN RTX | `2d72` 320², L=64 | **3.54 s** | 19.7 s (upper-triag, low memory) | 5.6× | **5.33 s** | 29.5 s | 5.5× | 13.1 s |
| TITAN RTX | `stack` 256²×28 | **6.16 s** | 34.7 s (upper-triag, low memory) | 5.6× | **7.84 s** | 62.0 s | 7.9× | 16.7 s |
| TITAN RTX | `3d` 128³ | **15.0 s** | 38.6 s (upper-triag, low memory) | 2.6× | **24.1 s** | 68.2 s | 2.8× | 76.7 s |
| L40S | `2d72` 320², L=72 | **1.83 s** | 14.1 s (upper-triag, low memory) | 7.7× | **3.97 s** | 23.3 s | 5.9× | 8.71 s |
| L40S | `stack` 320²×32 | **8.52 s** | 51.1 s (upper-triag, low memory) | 6.0× | **10.3 s** | 87.5 s | 8.5× | 18.6 s |
| L40S | `3d` 144³ | **16.0 s** | 47.2 s (upper-triag, low memory) | 2.9× | **22.3 s** | 78.8 s | 3.5× | 83.8 s |
| H100 | `2d72` 320², L=72 | **0.73 s** | 3.93 s (low memory) | 5.4× | **2.57 s** | 12.3 s | 4.8× | 4.03 s |
| H100 | `stack` 320²×48 | **3.13 s** | 18.3 s (upper-triag, low memory) | 5.8× | **5.27 s** | 58.9 s | 11.2× | 13.8 s |
| H100 | `3d` 160³ | **5.10 s** | 15.3 s (upper-triag, low memory) | 3.0× | **9.60 s** | 48.0 s | 5.0× | 41.0 s |

* The CG ratio follows the $A^H A$ ratio, since both libraries spend almost all their CG time
  in $A^H A$.
* The end-to-end ratio is larger. BART's MATLAB and Python interfaces run the `bart` executable
  and exchange `.cfl` files, so an operator cannot stay in GPU memory between calls, and every
  call recomputes the PSF. On the L40S and the H100 every call also starts the container.

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
| `2d` | 3.05 (0.03) | **2.09** (0.01) | 2.14 |
| `2d72` | 802 (5.68) | **202** (2.17) | 230 |
| `stack` | fails | **979** (3.05) | 1012 |
| `stack` with 32 slices | 1489 (31.70) | **555** (1.85) | 574 |
| `3d` | 1685 (36.10) | **1387** (18.94) | 1603 |

:::

:::{tab-item} L40S
:sync: l40s

| Problem | MRISubspaceRecon.jl | MR_AHA masked | MR_AHA Toeplitz |
|---|---:|---:|---:|
| `2d` | 0.82 (0.02) | **0.43** (0.28) | 0.43 |
| `2d72` | 362 (4.10) | **86** (2.31) | 90 |
| `stack` | fails | **744** (2.48) | 752 |
| `stack` with 32 slices | 1142 (19.38) | **412** (1.67) | 417 |
| `3d` | 1347 (21.44) | **1016** (7.37) | 1064 |

:::

:::{tab-item} H100
:sync: h100

| Problem | MRISubspaceRecon.jl | MR_AHA masked | MR_AHA Toeplitz |
|---|---:|---:|---:|
| `2d` | 0.88 (0.02) | **0.54** (0.21) | 0.53 |
| `2d72` | 76 (4.66) | **33** (2.02) | 36 |
| `stack` | fails | **170** (1.75) | 178 |
| `stack` with 32 slices | 272 (5.27) | **96** (1.07) | 100 |
| `3d` | 317 (6.15) | **232** (4.78) | 252 |

:::

::::

* On 2D and 3D problems MRISubspaceRecon.jl is 1.2–4.2× slower than MR_AHA's masked operator
  (1.05–4.0× slower than the unmasked one).
* On the stack it is 2.7–2.8× slower: it has no stack-of-stars operator (see below).
* `2d72` shows the largest gap on 2D and 3D problems: MR_AHA is 4.0×, 4.2× and 2.3× faster on
  the TITAN RTX, the L40S and the H100. $L = 72$ is above the largest rank (30) for which
  MR_AHA keeps the rank vectors in registers; such ranks run its
  [tiled kernels](../background/performance.md#25-ranks-above-30-tiled-kernels). Before those
  kernels MR_AHA's masked operator took 583, 288 and 165 ms, and MRISubspaceRecon.jl was 2.2×
  faster on the H100.
* MRISubspaceRecon.jl's set-up is 1.3–17× slower, except on the small `2d` problem on the L40S
  and the H100, where MR_AHA's set-up takes 0.2 to 0.3 s against 0.02 s.
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
python bart/bench_bart.py --problem 2d72 stack 3d --fit --check
python bart/bench_bart.py --problem 2d72 stack 3d --fit --methods bart_pics_cg mr_aha_toeplitz --cg 20
```

:::

:::{tab-item} BART from a container image
:sync: bart-sif

For machines without GCC 12, such as many clusters. BART comes from an Apptainer / Singularity
image; no root rights are needed.

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks
module load apptainer                   # or singularity
bart/build_bart.sh --sif                # downloads the image to bart/bart.sif
python bart/bench_bart.py --sif bart/bart.sif --problem 2d72 stack 3d --fit --check

# or everything in one Slurm job (MR_AHA for the GPU, the image, all runs):
mkdir -p bart/hpc/logs
sbatch --gres=gpu:h100:1 --export=ALL,ARCH=90 bart/hpc/bench_bart.sbatch
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
