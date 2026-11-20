# Benchmarks

One $A^H A$ of MR_AHA's Toeplitz operator (the faster of masked and unmasked), against the
Python NUFFT reference `aha_nufft_plan` on the same GPU (cufinufft's Python interface on the
NVIDIA GPUs, mlx-nufft on Apple silicon):

| problem | TITAN RTX | RTX 5090 | L40S | H100 | Apple M3 Ultra |
|---|---:|---:|---:|---:|---:|
| 2D radial 160², $L = 6$, 16 coils | 2.11 ms (10×) | 0.32 ms (43×) | 0.43 ms (48×) | 0.52 ms (36×) | 1.75 ms (45×) |
| 2D radial 320², $L = 72$, 12 coils | 198 ms (3.5×) | 39 ms (6.0×) | 86 ms (3.5×) | 33 ms (12×) | 182 ms (5.8×) |
| stack-of-stars 320² × 56, $L = 12$, 12 coils | 0.98 s (4.1×) | 0.31 s (4.4×) | 0.74 s (2.5×) | 0.17 s (13×) | 0.89 s (7.0×) |
| 3D kooshball 160³, $L = 12$, 12 coils | 1.38 s (7.8×) | 0.43 s (3.3×) | 1.02 s (3.8×) | 0.23 s (13×) | 1.10 s (18×) |

The speed-up over `aha_nufft_plan` on the same GPU is in parentheses. On four H100s the stack and
3D operators take 51 ms and 64 ms.

| Section | What it measures |
|---|---|
| [Single GPU](single-gpu.md) | Toeplitz against NUFFT: operator set-up, one $A^H A$ and 20 CG iterations, and their accuracy, on a TITAN RTX, an RTX 5090, an L40S, an H100 and an Apple M3 Ultra |
| [Multi GPU](multi-gpu.md) | operator set-up, one $A^H A$ and 20 CG iterations on 1, 2 and 4 GPUs, on TITAN RTX, L40S and H100 nodes |
| [vs BART](vs-bart.md) | MR_AHA against BART: one $A^H A$, 20 CG iterations and accuracy, on a TITAN RTX, an L40S and an H100 |
| [vs MRISubspaceRecon.jl](vs-mrisubspacerecon.md) | MR_AHA against MRISubspaceRecon.jl: one $A^H A$, 20 CG iterations and accuracy, on a TITAN RTX, an L40S and an H100 |

## Measure it yourself

[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) holds one script per language
that builds a test problem, creates the operator with each method on one GPU or multi-GPU, and
prints the set-up time and the time of one $A^H A$:

::::{tab-set}

:::{tab-item} Python
:sync: python

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks/python
python bench_aha.py --problem stack --gpus 1 2 4
```

:::

:::{tab-item} MATLAB
:sync: matlab

```matlab
cd MR_AHA_benchmarks/matlab
bench_aha('problem', 'stack', 'gpus', [1 2 4])
```

:::

::::

| Method | What runs | GPUs |
|---|---|---|
| `toeplitz` | the library's Toeplitz operator | 1 or several |
| `nufft_c` | the library's own NUFFT operator, `create_op(..., method='nufft')` (cufinufft's C API inside the library). Not used in the [Single GPU](single-gpu.md) tables | 1 or several |
| `nufft_native` | the NUFFT reference `aha_nufft_plan`, written with cufinufft's Python or MATLAB interface (mlx-nufft on Apple silicon). The NUFFT baseline of the [Single GPU](single-gpu.md) tables | 1 |

The problems are synthetic presets with adjustable sizes, or your own data file. The
repository's README lists every option.

```{note}
The older measurements were run with scripts that lived in the `benchmarks/` folder of MR_AHA, which
needed data files that are not public. The folder was removed when the benchmarks moved to
their own repository; the published part of it is still in MR_AHA's history, at commit
[`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks), and the pages link
there for scripts and raw numbers. Some pages name scripts that were never published. For new
measurements use MR_AHA_benchmarks.
```
