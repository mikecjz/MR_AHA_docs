# VkFFT-Metal feasibility check for the Apple-silicon backend

The Toeplitz operator spends its FFT time on one forward and one inverse batched C2C transform per
coil, on the zero-padded grid `[Mx, My, (Mz), L]` (column-major, batch `L`, unnormalised in both
directions, exactly as `cufftExecC2C` in `src/cuda/mr_aha.cu`). This study checks that VkFFT's Metal backend
computes that transform accurately and fast on the Mac Studio, at the sizes of the three
benchmark problems of the [main study](results.md).

The files named below are in [`benchmarks/vkfft/`](https://github.com/mikecjz/MR_AHA/tree/b4fb00c/benchmarks/vkfft) of MR_AHA at commit
`b4fb00c`, the last one that contained the benchmark folder.

| file | purpose |
|---|---|
| `vkfft_bench.cpp`, `build.sh` | VkFFT (Metal backend) on a shared `MTL::Buffer`: plan time, round-trip error, GPU time (command-buffer timestamps) of forward, inverse and forward+inverse |
| `check_vkfft.py` | runs `vkfft_bench` on fixed-seed data, compares its forward output against a double-precision SciPy FFT, and times MLX's `mx.fft` on the same data |
| `cufft_bench.py` | the same transform with cuFFT through CuPy, for the NVIDIA baseline |
| `results_AW650523HJY.json` | Mac Studio results (VkFFT default `coalescedMemory`) |
| `../results/pact400-recon3/cufft_bench.json` | TITAN RTX cuFFT results |

VkFFT and metal-cpp come from the `third_party/VkFFT` submodule (the `mikecjz/VkFFT` fork, see
[below](#the-vkfft-fork)). The results table was measured with upstream v1.3.4; the fork gives
the same numbers.

```bash
# on the Mac, from the repository root
benchmarks/vkfft/build.sh
.venv/bin/python benchmarks/vkfft/check_vkfft.py            # all cases
benchmarks/vkfft/vkfft_bench --dims 640 640 56 --batch 12 --coalesced 32
# on a CUDA machine
python benchmarks/vkfft/cufft_bench.py
```

## Results (2026-11-05)

Mac Studio: Apple M3 Ultra, 512 GB, macOS 26.7.2, VkFFT v1.3.4, MLX 0.31.2.
NVIDIA baseline: TITAN RTX (`pact400-recon3`), cuFFT 11.3 via CuPy 14.2. Times are medians of 20
runs after warm-up, in ms. VkFFT and cuFFT times are GPU times; MLX times are wall-clock times of
`mx.eval`, so they include dispatch overhead.

| case | grid × L | size | cuFFT TITAN RTX fwd+inv | VkFFT M3 Ultra fwd+inv (default / `coalescedMemory=32`) | MLX `mx.fft` fwd+inv |
|---|---|---|---|---|---|
| 2d | 320² × 6 | 4.9 MB | 0.083 | 0.12 / – | 0.41 |
| stack | 640² × 56 × 12 | 2.2 GB | 54.1 | 51.0 / **37.9** | 73.6 |
| 3d | 320³ × 12 | 3.1 GB | 78.2 | 53.9 / 54.0 | 104.9 |

The library uses `coalescedMemory = 32`, the fastest setting here and the value mlx-nufft also
measured best on Apple GPUs.

Accuracy of the forward transform against a double-precision SciPy FFT (relative L2 error) and the
round-trip error `||ifft(fft(x))/N − x|| / ||x||`:

| case | VkFFT forward | VkFFT round trip | MLX forward | MLX round trip | cuFFT round trip |
|---|---|---|---|---|---|
| 2d | 2.9e-7 | 3.1e-7 | 4.6e-7 | 6.6e-7 | 3.8e-7 |
| stack | 3.5e-7 | 3.7e-7 | 5.6e-7 | 6.9e-7 | 4.0e-7 |
| 3d | 3.8e-7 | 3.8e-7 | 6.1e-7 | 8.7e-7 | 4.9e-7 |

VkFFT plan creation (it generates and compiles its Metal kernels at run time) takes 2 ms (2d),
240 ms (stack) and 305 ms (3d). A plan is built once per operator (`create_op`).

**Conclusion.** VkFFT-Metal matches single-precision cuFFT accuracy and is 1.4–1.5× faster than
cuFFT on the TITAN RTX for the two large grids, and about 1.9× faster than MLX's FFT there. It
is the FFT of the Metal backend. The 2D grid is too small to matter (0.1 ms per coil).

## The VkFFT fork

The `third_party/VkFFT` submodule points to the fork
[mikecjz/VkFFT](https://github.com/mikecjz/VkFFT), branch `mr_aha`. It is upstream `develop`
(`e687182`, two bug-fix releases after v1.3.4) plus two commits for the Metal backend:

1. **Autorelease pools** (`bd406e8`, authored by nweston; the Metal commit of upstream PR
   [#227](https://github.com/DTolm/VkFFT/pull/227), which fixes issue
   [DTolm/VkFFT#226](https://github.com/DTolm/VkFFT/issues/226)). VkFFT leaves the objects it
   obtains autoreleased (`NS::String::string`, `MTL::CommandQueue::commandBuffer`, the encoders)
   to the autorelease pool. Plans then survive the draining of the pool, which MATLAB and Python
   do. Every call into VkFFT must run inside an autorelease pool; the library provides one around
   each API entry point.
2. **Push constants** (`2f9fb8f`). The fork passes each dispatch's push constants with
   `setBytes`, which copies them at encode time. Several dispatches can then share one command
   buffer, each with its own values.

`vkfft_bench --replan N` exercises the life cycle of a plan (create, run, destroy, drain) N
times. With the fork (`coalescedMemory=32`), `--replan 50` (2d) and `--replan 3` (stack, 3d)
pass. The forward error against SciPy is 2.5e-7 / 2.9e-7 / 3.2e-7 and forward + inverse takes
0.12 / 38.1 / 54.1 ms (2d / stack / 3d), the same as v1.3.4.

The fork can be dropped once upstream merges PR #227.

## Size limit

VkFFT's Metal backend indexes with 32 bits, so one transform covers fewer than 2^31 complex
values (2^34 bytes). The library rejects larger FFT batches (`src/metal/mr_metal_fft.cpp`), and
`calc_toep` chunks its components to stay below the limit.
