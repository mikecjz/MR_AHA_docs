---
orphan: true
myst:
  html_meta:
    robots: noindex, nofollow
---

# Why BART's $A^H A$ is slower

BART v1.0.01 evaluates the low-rank normal operator the same way MR_AHA's Toeplitz method does:
an FFT onto a 2× oversampled grid, an $L \times L$ multiplication by the kernel (BART calls it the
point-spread function, PSF) at every grid point, and an inverse FFT. Yet one $A^H A$ is 2.1–5.6×
slower than MR_AHA's, needs 2–10× the GPU memory, and a reconstruction through BART's Python or
MATLAB interface is 3–9× slower end to end. This page explains where that time and memory go.

All measurements are from one NVIDIA TITAN RTX (24 GB, 672 GB/s), BART at
[v1.0.01](https://codeberg.org/mrirecon/bart/src/commit/381039c4) built with CUDA, and MR_AHA at
[`711003d`](https://github.com/mikecjz/MR_AHA/tree/711003d), on the problems of the
[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) `bart/` benchmark.

## Where the time goes

Nsight Systems kernel times for one $A^H A$, averaged over 21 applications, BART with its default
full PSF:

| Stage | BART, `2d` | MR_AHA, `2d` | BART, `2d72` with $L = 24$ | MR_AHA, `2d72` with $L = 24$ |
|---|---:|---:|---:|---:|
| Kernel (PSF) multiplication | **2.47 ms** | 0.53 ms | **122.6 ms** | 24.2 ms |
| FFT and inverse FFT | 1.20 ms | 1.31 ms | 14.5 ms | 15.9 ms |
| Coil maps, phases, padding, coil sum | 0.84 ms | 0.35 ms | 11.7 ms | 3.8 ms |
| **Whole $A^H A$** | **4.7 ms** | **2.2 ms** | **148 ms** | **43 ms** |

`2d` is 160², $L = 6$, 16 coils. `2d72` is 320², $L$ reduced from 72 to 24 so that BART fits, 12
coils.

The FFTs cost the same in both. Almost all of the difference is the kernel multiplication: 83 % of
BART's operator on `2d72`, and 5× slower than MR_AHA's.

## The kernel multiplication

### BART: a generic strided multiply–add

BART writes the multiplication as a general tensor contraction, `md_ztenmul()`
(`toeplitz_mult()` in `src/noncart/nufft.c`), which on the GPU runs as `kern_zfmac_strides`
(`src/num/gpukrnls.cu`). That kernel works for any array layout. Each thread takes one grid point
and loops over up to three further dimensions (here the coils and the two rank indices), and every
term is a separate read–modify–write in GPU memory:

```c
dst[i + o_offset] = cuCaddf(dst[i + o_offset],
                            cuCmulf(src1[i + i1_offset], src2[i + i2_offset]));
```

Every complex multiply–add therefore reads three numbers and writes one: up to 32 bytes of memory
traffic for 8 floating-point operations. For `2d72` with $L = 24$ there are
$640^2 \times 12 \text{ coils} \times 24^2 = 2.8 \cdot 10^9$ multiply–adds, which is up to 91 GB
of traffic per $A^H A$. At the TITAN RTX's 672 GB/s that takes 135 ms; the measured time is
123 ms (some accesses hit the cache). The multiplication is bound by memory bandwidth, and the
amount of memory it touches grows with $L^2$ times the number of coils.

BART also stores and reads the **full** $L \times L$ PSF by default, although the matrix at every
grid point is Hermitian: twice the entries that are needed.

### MR_AHA: one thread per grid point, the vector in registers

MR_AHA's block-diagonal kernel (`mr_k_bd_dense_t<L>`, `src/cuda/mr_kernels_bd.cu`) is compiled
separately for every rank from 1 to 30 (larger ranks use a generic block kernel). One thread
handles one grid point. It loads that point's $L$
coefficients of a coil into registers once, walks the packed upper triangle of the kernel once,
applying each entry $a$ to both $y_c \mathrel{+}= a\,v_r$ and $y_r \mathrel{+}= \bar a\,v_c$, and
writes the $L$ results once. For $L \le 16$ it processes two coils per walk of the triangle, so
the kernel is read once for both.

Per grid point and coil this moves $(L^2 + L)/2$ kernel entries and $2L$ coefficients, about
2.8 kB for $L = 24$, against up to 18 kB for BART. For `2d72` that is about 14 GB per $A^H A$,
or 20 ms at full bandwidth. The measured time is 24 ms.

### BART's upper-triangular PSF does not help

With `--nufft-conf upper-triag-psf` BART stores only $(L^2 + L)/2$ blocks, but it applies them
with one generic kernel launch per block and per conjugation (`md_ztenmul_upper_triag()`): about
2,300 launches of 55–60 µs each for one $A^H A$ of `2d72` with $L = 24$, 134 ms in total. It halves
the PSF memory and is 5–10 % slower than the full PSF.

In v1.0.01 this path also applies the complex conjugate of the Gram blocks, so it is wrong for a
complex temporal basis and right only for a real one (see
[Comparison with BART and MRISubspaceRecon.jl](other-libraries.md)).

## Memory: all coils at once

BART's Toeplitz operator processes every coil in one pass. Its working grids hold
$N_\text{coils} \times L$ images of the oversampled size: for `2d72` with $L = 24$, 0.94 GB per
grid, next to a full PSF of 1.9 GB. With the FFT work space and the temporaries of the tensor
operations, the operator takes 12.3 GB, against 1.3 GB for MR_AHA, which keeps one coil (or a
small group of coils) in flight at a time and stores the triangular kernel (0.98 GB).

BART also splits the 2× oversampled grid into $2^d$ shifted grids of the image size
("decomposed" Toeplitz embedding): 4 in 2D, 8 in 3D. Without `--lowmem` all of them are in memory
at once. On the 24 GB GPU this is what makes the 3D and stack problems fail:

| Problem | MR_AHA | BART, full PSF | BART, low memory |
|---|---:|---:|---:|
| stack 160² × 28, $L = 12$ | 2.8 GB, 133 ms | out of memory | 10.9 GB, 751 ms |
| 3D 96³, $L = 12$ | 6.8 GB, 387 ms | out of memory | 13.1 GB, 829 ms |

Low-memory mode (`-U` in `pics`) processes the $2^d$ shifted grids one after another, so each
FFT and each multiplication runs on a smaller batch, and still holds all coils of one shifted
grid at once. The full-size `2d72`, `stack` and `3d` presets do not fit in 24 GB in any mode.

## No stack-of-stars operator

BART has no stack-of-stars mode. A stack is passed as a 3D trajectory, and the operator oversamples
$z$ by 2 like $x$ and $y$: twice the slices in every FFT and in the PSF, and twice the PSF memory.
MR_AHA's `"stack"` trajectory type keeps $M_z = N_z$ (the $k_z$ positions are already on the
Cartesian grid). Together with the memory limit above, this is why the stack shows the largest gap,
5.6× per $A^H A$ and 5.5× for 20 CG iterations.

## The interface: a new process for every call

BART's MATLAB (`bart.m`) and Python (`bart.py`) interfaces do not call a library. They write every
input array as a `.cfl` file, start the `bart` executable, and read the result back from a file.
Nothing stays in GPU memory between calls, so every `bart pics` call:

* starts a process and initialises CUDA;
* writes and reads all arrays through the file system;
* recomputes the PSF from the trajectory and the basis;
* computes the scaling and the adjoint of the data, then iterates.

For the `2d` problem this costs about 0.8 s of a 0.91 s call with 20 CG iterations. On the stack
and 3D problems it is as long as the iterations themselves. This is why a 20-iteration CG
reconstruction is 3–9× slower end to end but only 1.9–5.5× slower in the iterations.

MR_AHA's operator is an object that is created once and applied as often as needed, with its
kernel kept on the GPU.

## Reproduce the profile

From a clone of MR_AHA_benchmarks with BART built (`bart/build_bart.sh`), write the BART inputs
of a problem with `bart/bench_bart.py` (its `write_inputs()` function), then profile the timing
helper and MR_AHA's benchmark:

```bash
nsys profile -t cuda -o bart_2d72 bart/bart_aha -g --reps 20 --warmup 1 traj basis maps x
nsys profile -t cuda -o mraha_2d72 python python/bench_aha.py --problem 2d72 --L 24 \
    --methods toeplitz --reps 20 --warmup 1
nsys stats -r cuda_gpu_kern_sum bart_2d72.nsys-rep
```
