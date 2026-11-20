# MR_AHA vs MRISubspaceRecon.jl

This study compares MR_AHA's low-rank normal operator $A^H A$ with the one in
[MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl) (Julia,
CUDA.jl). Both evaluate $A^H A$ with Toeplitz embedding: an FFT onto a 2× oversampled grid, an
$L \times L$ multiplication at every grid point, and an inverse FFT. They differ in how they store
and apply the kernel, in how they handle stack-of-stars trajectories, and in what they cost to set
up.

The same study against BART is on [MR_AHA vs BART](vs-bart.md).

## Summary

* **MR_AHA is faster on every problem and GPU measured.** MRISubspaceRecon.jl is 1.2–4.2× slower
  per $A^H A$ than MR_AHA's masked operator on 2D and 3D problems, and its set-up is up to 17×
  slower.
* **20 CG iterations** with MRISubspaceRecon.jl are 1.2–4.2× slower than with MR_AHA's masked
  operator on 2D and 3D problems, and 2.7–2.8× slower on the stack. Both libraries spend almost all
  their CG time in $A^H A$.
* **Stack-of-stars is where the difference is largest.** MRISubspaceRecon.jl has no
  stack-of-stars operator and treats a stack as a 3D problem oversampled along $z$ as well.
  MR_AHA's stack operator is 2.7–2.8× faster. MRISubspaceRecon.jl also cannot run the standard
  stack problem at all (see [Limits of MRISubspaceRecon.jl](#limits-of-mrisubspacereconjl)).
* **High rank.** At $L = 72$ MR_AHA is 2.3–4.2× faster, with its
  [tiled kernels for ranks above 30](../background/performance.md#23-ranks-above-30-tiled-kernels).
  Before them, MRISubspaceRecon.jl was 2.2× faster than MR_AHA on the H100.
* **MRISubspaceRecon.jl computes the same operator up to its kernel mask.** It agrees with an
  exact $A^H A$ to $2$–$5 \cdot 10^{-3}$, the error of the mask, which cannot be switched off.

## What was compared

| Library | Version | Operator |
|---|---|---|
| MR_AHA | [`c60db28`](https://github.com/mikecjz/MR_AHA/tree/c60db28) | `mr_aha.create_op`, method `"toeplitz"` (unmasked and masked) |
| MRISubspaceRecon.jl | v0.10.1 ([`1e14dff`](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/tree/1e14dff48634ce8bb8222177e7bfd335a8ccf03d)), Julia 1.11.7, CUDA.jl 6.5.1 | `NFFTNormalOp` on `CuArray`s |

| GPU | MRISubspaceRecon.jl |
|---|---|
| NVIDIA TITAN RTX, 24 GB (sm_75), CUDA 12.6 | measured on 2026-11-12, not repeated for this MR_AHA version |
| NVIDIA L40S, 48 GB (45 GB usable; sm_89), CUDA 12.9 | same job and node as MR_AHA |
| NVIDIA H100 80GB HBM3 (sm_90), CUDA 12.9 | same job and node as MR_AHA |

Every run used one GPU. MRISubspaceRecon.jl runs on one GPU only.

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

MRISubspaceRecon.jl cannot create the operator of the standard `stack` preset (see
[Limits of MRISubspaceRecon.jl](#limits-of-mrisubspacereconjl)). The tables therefore add the same
stack with 32 slices instead of 56, which it accepts.

### From Gram blocks to a temporal basis

MR_AHA takes one Gram block $\Phi_2$ per readout. MRISubspaceRecon.jl takes a temporal basis
instead: one row of $L$ coefficients per time frame, and one trajectory per time frame. The
benchmark converts one into the other exactly. Each Gram block is eigen-decomposed,
$\Phi_2 = \sum_k u_k u_k^H$, and each readout becomes $r$ time frames that share its trajectory,
with basis rows $u_k$ ($r = 2$ for the synthetic blocks). The conversion is not timed. It doubles
the number of non-uniform points from which MRISubspaceRecon.jl computes its kernel, so it adds to
its set-up time but not to $A^H A$.

## How it was run

"Masked" is MR_AHA's masked Toeplitz operator, the closest match: MRISubspaceRecon.jl always
masks its kernel. Times are medians of 10 applications after 2 warm-up runs; set-up is the median
of the warm creations (the first creation in Julia also compiles, about 1.7 s for `2d`).

For CG, each library runs 20 plain CG iterations on $A^H A\,u = x$ from $u = 0$, without a
stopping test, after one untimed iteration. Both use the same algorithm (`cg_time` of the
benchmarks' `python/bench_aha.py`, which the [BART comparison](vs-bart.md#20-cg-iterations) also
uses): one $A^H A$, two inner products and three vector updates per iteration. The CG runs used
MR_AHA [`da3e148`](https://github.com/mikecjz/MR_AHA/tree/da3e148), whose CUDA code is the same as
`c60db28`'s. On the L40S and the H100 both libraries ran in the same job; on the TITAN RTX the same
commands ran by hand on one GPU.

## One $A^H A$

In ms, set-up in s in parentheses.

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

| Problem | MR_AHA masked | MR_AHA Toeplitz | MRISubspaceRecon.jl | MRISubspaceRecon.jl / MR_AHA masked |
|---|---:|---:|---:|---:|
| `2d` | **2.09** (0.01) | 2.14 | 3.05 (0.03) | 1.5× |
| `2d72` | **202** (2.17) | 230 | 802 (5.68) | 4.0× |
| `stack` | **979** (3.05) | 1012 | fails | – |
| `stack` with 32 slices | **555** (1.85) | 574 | 1489 (31.70) | 2.7× |
| `3d` | **1387** (18.94) | 1603 | 1685 (36.10) | 1.2× |

:::

:::{tab-item} L40S
:sync: l40s

| Problem | MR_AHA masked | MR_AHA Toeplitz | MRISubspaceRecon.jl | MRISubspaceRecon.jl / MR_AHA masked |
|---|---:|---:|---:|---:|
| `2d` | **0.43** (0.28) | 0.43 | 0.82 (0.02) | 1.9× |
| `2d72` | **86** (2.31) | 90 | 362 (4.10) | 4.2× |
| `stack` | **744** (2.48) | 752 | fails | – |
| `stack` with 32 slices | **412** (1.67) | 417 | 1142 (19.38) | 2.8× |
| `3d` | **1016** (7.37) | 1064 | 1347 (21.44) | 1.3× |

:::

:::{tab-item} H100
:sync: h100

| Problem | MR_AHA masked | MR_AHA Toeplitz | MRISubspaceRecon.jl | MRISubspaceRecon.jl / MR_AHA masked |
|---|---:|---:|---:|---:|
| `2d` | **0.54** (0.21) | 0.53 | 0.88 (0.02) | 1.6× |
| `2d72` | **33** (2.02) | 36 | 76 (4.66) | 2.3× |
| `stack` | **170** (1.75) | 178 | fails | – |
| `stack` with 32 slices | **96** (1.07) | 100 | 272 (5.27) | 2.8× |
| `3d` | **232** (4.78) | 252 | 317 (6.15) | 1.4× |

:::

::::

* On 2D and 3D problems MRISubspaceRecon.jl is 1.2–4.2× slower than MR_AHA's masked operator
  (1.05–4.0× slower than the unmasked one).
* On the stack it is 2.7–2.8× slower: it has no stack-of-stars operator (see
  [Limits of MRISubspaceRecon.jl](#limits-of-mrisubspacereconjl)).
* `2d72` shows the largest gap on 2D and 3D problems: MR_AHA is 4.0×, 4.2× and 2.3× faster on
  the TITAN RTX, the L40S and the H100. $L = 72$ is above the largest rank (30) for which
  MR_AHA keeps the rank vectors in registers; such ranks run its
  [tiled kernels](../background/performance.md#23-ranks-above-30-tiled-kernels). Before those
  kernels MR_AHA's masked operator took 583, 288 and 165 ms, and MRISubspaceRecon.jl was 2.2×
  faster on the H100.
* MRISubspaceRecon.jl's set-up is 1.3–17× slower, except on the small `2d` problem on the L40S
  and the H100, where MR_AHA's set-up takes 0.2 to 0.3 s against 0.02 s.
* Its kernel memory is close to MR_AHA's masked kernel (`3d`: 18.5 GB, against 15.1 GB masked
  and 24.0 GB unmasked for MR_AHA on the TITAN RTX).

## 20 CG iterations

In s, on the same problems as one $A^H A$.

::::{tab-set}

:::{tab-item} TITAN RTX
:sync: titanrtx

| Problem | MR_AHA masked | MR_AHA Toeplitz | MRISubspaceRecon.jl | MRISubspaceRecon.jl / MR_AHA masked | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|
| `2d` | **0.050** | 0.051 | 0.065 | 1.3× | 0.43 |
| `2d72` | **4.30** | 4.81 | 16.1 | 3.7× | 15.9 |
| `stack` | **20.3** | 20.9 | fails (2-3-5 check) | – | 58.5 |
| `stack` with 32 slices | **11.4** | 11.8 | 31.0 | 2.7× | 33.1 |
| `3d` | **28.5** | out of memory | 34.1 | 1.2× | 175 |

MR_AHA's unmasked `3d` operator (24.0 GB) did not fit on the 24 GB TITAN RTX next to the
other arrays of the CG run; its masked one (15.1 GB) did.

:::

:::{tab-item} L40S
:sync: l40s

| Problem | MR_AHA masked | MR_AHA Toeplitz | MRISubspaceRecon.jl | MRISubspaceRecon.jl / MR_AHA masked | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|
| `2d` | **0.014** | 0.014 | 0.018 | 1.3× | 0.27 |
| `2d72` | **1.75** | 1.83 | 7.27 | 4.2× | 8.75 |
| `stack` | **15.2** | 15.4 | fails (2-3-5 check) | – | 32.4 |
| `stack` with 32 slices | **8.41** | 8.50 | 23.0 | 2.7× | 18.5 |
| `3d` | **20.6** | 21.5 | 27.1 | 1.3× | 115 |

:::

:::{tab-item} H100
:sync: h100

| Problem | MR_AHA masked | MR_AHA Toeplitz | MRISubspaceRecon.jl | MRISubspaceRecon.jl / MR_AHA masked | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|
| `2d` | **0.013** | 0.013 | 0.019 | 1.5× | 0.13 |
| `2d72` | **0.67** | 0.73 | 1.54 | 2.3× | 3.96 |
| `stack` | **3.48** | 3.63 | fails (2-3-5 check) | – | 16.0 |
| `stack` with 32 slices | **1.97** | 2.06 | 5.47 | 2.8× | 9.23 |
| `3d` | **4.72** | 5.13 | 6.44 | 1.4× | 40.4 |

:::

::::

* Apart from the tiny `2d` problem, CG time is 20 $A^H A$ applications plus 1–3 % in both
  libraries, so the ratios follow those of one $A^H A$.
* MRISubspaceRecon.jl cannot run the standard `stack` problem at all (see
  [Limits of MRISubspaceRecon.jl](#limits-of-mrisubspacereconjl)).

## Accuracy

The relative difference to an exact reference $A^H A$, applied without a Toeplitz kernel, is the
same on all three GPUs:

| Problem | MR_AHA Toeplitz | MR_AHA masked | MRISubspaceRecon.jl |
|---|---:|---:|---:|
| `2d` | 9.4e-7 | 3.0e-3 | 1.9e-3 |
| stack 64² × 16, $L = 4$, 4 coils | 1.2e-6 | 1.1e-2 | 4.6e-3 |
| 3D 48³, $L = 4$, 4 coils | 1.4e-6 | 1.0e-2 | 3.4e-3 |

The best-fit scalar is 0.99999 in every case, so both libraries compute the same unnormalized
operator with the same sign and Gram-block conventions. The remaining difference is the mask.
MRISubspaceRecon.jl keeps the grid points within its Gaussian spreading kernel's reach of a
sample; MR_AHA's default mask (`calc_mask`, `mask_radius = 2`) keeps a similar fraction chosen
differently (2D: 82 % against 81 %, 3D: 57 % against 54 % of the grid) and is less accurate on
these problems. The random input is a worst case for any mask.

(limits-of-mrisubspacereconjl)=
## Limits of MRISubspaceRecon.jl

The following describes MRISubspaceRecon.jl at commit
[`1e14dff`](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/tree/1e14dff48634ce8bb8222177e7bfd335a8ccf03d)
(v0.10.1).

* **No stack-of-stars support.** A stack must be given as a 3D trajectory, with
  $k_z$ as a third coordinate. The operator then oversamples $z$ by 2 like $x$ and $y$: the FFTs
  run on a grid with twice the slices, and the kernel is twice as large. MR_AHA's `"stack"`
  trajectory type keeps $M_z = N_z$ and needs no FFT along $z$ for the kernel. This is why its
  stack operator is 2.7–2.8× faster.
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

The scripts are in the `julia/` folder of
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks), with a README that describes
the set-up, every option and the results. The CG option `--cg` comes from
[MR_AHA_benchmarks#2](https://github.com/mikecjz/MR_AHA_benchmarks/pull/2).

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python save_problem.py 2d data/2d.mat
julia -t 8 --project=. bench_aha.jl --data data/2d.mat
julia -t 8 --project=. bench_aha.jl --data data/2d.mat --cg 20   # adds 20 CG iterations
```
