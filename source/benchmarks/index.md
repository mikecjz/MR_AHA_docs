# Benchmarks

## Measure it yourself

[MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks) holds one script per language
that builds a test problem, creates the operator with each method on one GPU or multi-GPU, and
prints the set-up time and the time of one $A^H A$:

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks/python
python bench_aha.py --problem stack --gpus 1 2 4
```

```matlab
cd MR_AHA_benchmarks/matlab
bench_aha('problem', 'stack', 'gpus', [1 2 4])
```

| Method | What runs | GPUs |
|---|---|---|
| `toeplitz` | the library's Toeplitz operator | 1 or several |
| `nufft_c` | the library's NUFFT operator (cufinufft's C API inside the library) | 1 or several |
| `nufft_native` | the same NUFFT operator written with cufinufft's Python or MATLAB interface | 1 |

The problems are synthetic presets with adjustable sizes, or your own data file. The
repository's README lists every option.

## Published studies

The pages of this section record the measurements made while the library was developed.

| Study | Question |
|---|---|
| [Main study](results.md) | Toeplitz against NUFFT, one $A^H A$ and 20 CG iterations, on a TITAN RTX, an L40S and an Apple M3 Ultra |
| [Neighbourhood mask](neighbour-mask.md) | how much of the grid the mask keeps and how accurate the masked operator is, for radii 1 to 8 |
| [CUDA optimisation and multi-GPU](cuda-multigpu.md) | the default CUDA kernels against the generic ones, and multi-GPU scaling on H100s |
| [NUFFT upsampling factor](upsampfac.md) | experimental NUFFT settings that compute the Toeplitz kernel faster |
| [VkFFT on Metal](vkfft.md) | whether VkFFT's Metal backend is accurate and fast enough |

```{note}
These studies were run with scripts that lived in the `benchmarks/` folder of MR_AHA, which
needed data files that are not public. The folder was removed when the benchmarks moved to
their own repository; the published part of it is still in MR_AHA's history, at commit
[`b4fb00c`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks), and the pages link
there for scripts and raw numbers. Some pages name scripts that were never published. For new
measurements use MR_AHA_benchmarks.
```
