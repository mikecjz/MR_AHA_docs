# Benchmarks

One $A^H A$ of MR_AHA's Toeplitz operator (the faster of masked and unmasked), against the NUFFT
operator with the same interface:

| problem | TITAN RTX | L40S | H100 | Apple M3 Ultra |
|---|---:|---:|---:|---:|
| 2D radial 160², $L = 6$, 16 coils | 2.19 ms (8.9×) | 0.43 ms (30×) | 0.52 ms (13×) | 1.68 ms (47×) |
| 2D radial 320², $L = 72$, 12 coils | 199 ms (3.6×) | 86 ms (5.1×) | 33 ms (6.1×) | 182 ms (5.8×) |
| stack-of-stars 320² × 56, $L = 12$, 12 coils | 0.98 s (2.6×) | 0.74 s (2.2×) | 0.17 s (4.7×) | 0.89 s (7.0×) |
| 3D kooshball 160³, $L = 12$, 12 coils | 1.39 s (5.6×) | 1.02 s (5.6×) | 0.23 s (8.8×) | 1.10 s (18×) |

The speed-up over the NUFFT operator on the same GPU is in parentheses. On four H100s the stack and
3D operators take 51 ms and 64 ms.

| Section | What it measures |
|---|---|
| [Single GPU](single-gpu.md) | Toeplitz against NUFFT: operator set-up, one $A^H A$ and 20 CG iterations, and their accuracy, on a TITAN RTX, an L40S, an H100 and an Apple M3 Ultra |
| [Multi GPU](multi-gpu.md) | operator set-up, one $A^H A$ and 20 CG iterations on 1, 2 and 4 GPUs, on TITAN RTX, L40S and H100 nodes |
| [Julia and BART](other-libraries.md) | MR_AHA against BART and MRISubspaceRecon.jl: one $A^H A$, 20 CG iterations, memory and accuracy, on a TITAN RTX, an L40S and an H100 |

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
| `nufft_c` | the library's NUFFT operator (cufinufft's C API inside the library) | 1 or several |
| `nufft_native` | the same NUFFT operator written with cufinufft's Python or MATLAB interface | 1 |

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
