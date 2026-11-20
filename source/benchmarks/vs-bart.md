# MR_AHA vs BART

This study compares MR_AHA's low-rank normal operator $A^H A$ with the one in
[BART](https://codeberg.org/mrirecon/bart), the Berkeley Advanced Reconstruction Toolbox (C and
CUDA, with MATLAB and Python wrappers). Both evaluate $A^H A$ with Toeplitz embedding: an FFT onto
a 2× oversampled grid, an $L \times L$ multiplication at every grid point, and an inverse FFT. They
differ in how they store and apply the kernel, in how they handle coils and stack-of-stars
trajectories, and in how a reconstruction calls them.

The same study against MRISubspaceRecon.jl is on [MR_AHA vs MRISubspaceRecon.jl](vs-mrisubspacerecon.md).

## Summary

* **MR_AHA is faster on every problem and GPU measured.** BART is 2.2–7.8× slower per $A^H A$ and
  needs 1.2–3.2× the GPU memory on the larger problems, in its most frugal mode. It cannot hold
  the standard stack problem on any of the three GPUs, not even in 80 GB.
* **20 CG iterations** with `bart pics` are 2.6–7.7× slower than the same iterations with MR_AHA,
  and 2.8–11× slower end to end, because every call through BART's Python or MATLAB interface
  starts a new process, exchanges files and recomputes the kernel.
* **Stack-of-stars is where the difference is largest.** BART has no stack-of-stars operator and
  treats a stack as a 3D problem oversampled along $z$ as well. MR_AHA's stack operator is
  5.7–6.0× faster.
* **High rank.** At $L = 72$ MR_AHA is 7.7–7.8× faster, with its
  [tiled kernels for ranks above 30](../background/performance.md#23-ranks-above-30-tiled-kernels).
  Before them, BART was 1.2× faster than MR_AHA on the H100.
* **BART's full PSF computes the same operator** as MR_AHA, to $10^{-5}$ after a real scale.
  Its upper-triangular PSF is wrong for a complex temporal basis (see [Accuracy](#accuracy)).

## What was compared

| Library | Version | Operator |
|---|---|---|
| MR_AHA | [`c60db28`](https://github.com/mikecjz/MR_AHA/tree/c60db28) | `mr_aha.create_op`, method `"toeplitz"` |
| BART | v1.0.01 ([`381039c`](https://codeberg.org/mrirecon/bart/src/commit/381039c4)), built with CUDA | the normal operator of `bart pics` (`sense_nc_init()`), timed with a small C program linked against BART's libraries, and `bart pics` itself through `bart.py` |

| GPU | BART |
|---|---|
| NVIDIA TITAN RTX, 24 GB (sm_75), CUDA 12.6 | compiled on the machine |
| NVIDIA L40S, 48 GB (45 GB usable; sm_89), CUDA 12.9 | from a container image |
| NVIDIA H100 80GB HBM3 (sm_90), CUDA 12.9 | from a container image |

Every run used one GPU. BART was not tried on several.

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
gets the largest problem of the ladder that BART holds on it (BART's memory in parentheses):

| Preset | TITAN RTX (24 GB) | L40S (45 GB) | H100 (80 GB) |
|---|---|---|---|
| `2d72` (320², L=72) | **L=64** (22.8 GB) | as it is (28.4 GB) | as it is (28.4 GB) |
| `stack` (320²×56, L=12) | **256²×28** (17.2 GB) | **320²×32** (30.7 GB) | **320²×48** (46.0 GB) |
| `3d` (160³, L=12) | **128³** (19.6 GB) | **144³** (27.9 GB) | as it is (38.3 GB) |

### From Gram blocks to a temporal basis

MR_AHA takes one Gram block $\Phi_2$ per readout. BART takes a temporal basis instead: one row of
$L$ coefficients per time frame, and one trajectory per time frame. The benchmark converts one
into the other exactly. Each Gram block is eigen-decomposed, $\Phi_2 = \sum_k u_k u_k^H$, and each
readout becomes $r$ time frames that share its trajectory, with basis rows $u_k$ ($r = 2$ for the
synthetic blocks). The conversion is not timed. It doubles the number of non-uniform points from
which BART computes its kernel, so it adds to its set-up time but not to $A^H A$.

## How it was run

On the TITAN RTX, times are the faster of two runs on a shared machine. Set-up includes the
kernel (BART calls it the point-spread function, PSF). Memory is what a method holds on the GPU
after its warm-up.

BART's Toeplitz embedding splits the 2× oversampled grid into $2^d$ shifted grids of the image
size: 4 in 2D, 8 in 3D. By default all of them are in GPU memory at once. **Low-memory mode**
(`-U` in `bart pics`) processes them one after another instead, which needs less memory at
little cost in time here. The **upper-triangular PSF** stores only the upper triangle of each
Hermitian $L \times L$ kernel block.

`bart pics -g` was run through BART's own Python interface (`bart.py`) with 1 and 21 iterations;
the difference is the time of 20 iterations. BART's CG has no stopping tolerance by default, so
every run does all its iterations. BART's mode is the first that fits of: full PSF, low memory,
upper-triangular PSF with low memory. MR_AHA ran 20 plain CG iterations on $A^H A\,u = b$ with
the same operator. "End to end" is, for MR_AHA, copying the inputs to the GPU, creating the
operator and the 20 iterations, and for BART the whole `pics` call: process start, file I/O, PSF,
scaling, adjoint and the 20 iterations.

## One $A^H A$

On the problems fitted to each GPU. GPU memory in parentheses.

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

* MR_AHA is 2.2–7.8× faster than BART's fastest mode, and holds less memory.
* On the large problems BART runs only with the upper-triangular PSF in low-memory mode.
* `bart pics` per CG iteration agrees with the operator time, so the C program measures what
  BART users get. On the tiny `2d` problem the per-iteration time through the container is not
  reliable, because the start of the container varies by more than the iterations take.

## 20 CG iterations

On the problems fitted to each GPU.

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

| Problem | MR_AHA Toeplitz CG | BART `pics` CG | Ratio | MR_AHA end to end | BART end to end | Ratio | MR_AHA NUFFT CG |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d72` 320², L=64 | **3.54 s** | 19.7 s (upper-triag, low memory) | 5.6× | **5.33 s** | 29.5 s | 5.5× | 13.1 s |
| `stack` 256²×28 | **6.16 s** | 34.7 s (upper-triag, low memory) | 5.6× | **7.84 s** | 62.0 s | 7.9× | 16.7 s |
| `3d` 128³ | **15.0 s** | 38.6 s (upper-triag, low memory) | 2.6× | **24.1 s** | 68.2 s | 2.8× | 76.7 s |

:::

:::{tab-item} L40S
:sync: l40s

| Problem | MR_AHA Toeplitz CG | BART `pics` CG | Ratio | MR_AHA end to end | BART end to end | Ratio | MR_AHA NUFFT CG |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d72` 320², L=72 | **1.83 s** | 14.1 s (upper-triag, low memory) | 7.7× | **3.97 s** | 23.3 s | 5.9× | 8.71 s |
| `stack` 320²×32 | **8.52 s** | 51.1 s (upper-triag, low memory) | 6.0× | **10.3 s** | 87.5 s | 8.5× | 18.6 s |
| `3d` 144³ | **16.0 s** | 47.2 s (upper-triag, low memory) | 2.9× | **22.3 s** | 78.8 s | 3.5× | 83.8 s |

:::

:::{tab-item} H100
:sync: h100

| Problem | MR_AHA Toeplitz CG | BART `pics` CG | Ratio | MR_AHA end to end | BART end to end | Ratio | MR_AHA NUFFT CG |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d72` 320², L=72 | **0.73 s** | 3.93 s (low memory) | 5.4× | **2.57 s** | 12.3 s | 4.8× | 4.03 s |
| `stack` 320²×48 | **3.13 s** | 18.3 s (upper-triag, low memory) | 5.8× | **5.27 s** | 58.9 s | 11.2× | 13.8 s |
| `3d` 160³ | **5.10 s** | 15.3 s (upper-triag, low memory) | 3.0× | **9.60 s** | 48.0 s | 5.0× | 41.0 s |

:::

::::

* The CG ratio follows the $A^H A$ ratio, since both libraries spend almost all their CG time
  in $A^H A$.
* The end-to-end ratio is larger. BART's MATLAB and Python interfaces run the `bart` executable
  and exchange `.cfl` files, so an operator cannot stay in GPU memory between calls, and every
  call recomputes the PSF. On the L40S and the H100 every call also starts the container.

## Accuracy

The relative difference to MR_AHA's Toeplitz operator, after the best-fit real scalar (BART
normalizes its FFTs differently), is the same on all three GPUs. MR_AHA's Toeplitz operator is
itself within $1.4 \cdot 10^{-6}$ of an exact $A^H A$ applied without a Toeplitz kernel.

| Problem | BART full PSF |
|---|---:|
| `2d` | 7e-6 |
| `2d72`, $L = 24$ | 1.6e-5 |

BART's full PSF computes the same operator as MR_AHA. Its upper-triangular PSF does not: in
v1.0.01 it multiplies by the complex conjugate of the Gram blocks, so with the complex Gram blocks
of these problems it is off by 0.2–0.9. With a real temporal basis, the usual case in BART, it
agrees to about $10^{-5}$. The bug does not change its run time. For comparison, MR_AHA's own
NUFFT operator agrees with its Toeplitz operator to $8 \cdot 10^{-7}$–$2 \cdot 10^{-6}$.

## Limits of BART

The following describes BART v1.0.01. [Why BART's $A^H A$ is slower](why-bart-slower.md) has the
profiles behind these points.

* **No stack-of-stars support.** A stack must be given as a 3D trajectory. The operator then
  oversamples $z$ by 2 like $x$ and $y$, so its PSF is twice MR_AHA's.
* **All coils at once.** BART's working grids hold every coil's $L$ images at the same time;
  MR_AHA loops over the coils with one coil's images in flight. This is most of BART's extra
  memory.
* **Full or upper-triangular PSF only.** The full PSF does not fit the large problems; the
  upper-triangular PSF saves memory but not time, and is wrong for a complex basis.
* **No in-memory operator from MATLAB or Python.** Every call starts the `bart` executable,
  exchanges files and recomputes the PSF.

## Reproduce

The scripts are in the `bart/` folder of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks), with a README that describes
the set-up, every option and the results.

::::{tab-set}

:::{tab-item} Native build
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

:::{tab-item} From a container image
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

::::
