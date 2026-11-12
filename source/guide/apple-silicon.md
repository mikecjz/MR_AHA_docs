# Apple silicon (Metal backend)

The Metal backend runs the same operators on the GPU of Apple-silicon Macs (developed and tested on
an M3 Ultra). It is selected automatically on macOS (`MR_AHA_BACKEND=METAL`).

**How it maps onto the CUDA backend.**

| | CUDA | Metal |
|---|---|---|
| kernels | `src/cuda/*.cu` | `src/metal/mr_kernels.metal` (same indexing, ported kernel by kernel) |
| FFT | cuFFT | [VkFFT](https://github.com/DTolm/VkFFT)'s Metal backend (`third_party/VkFFT`) |
| type-1 NUFFT (`calc_toep`) | cufinufft, inside the library | [mlx-nufft](https://github.com/martinlachaine/mlx-nufft), in Python (`third_party/mlx-nufft`) |
| arrays | device pointers (gpuArray, CuPy) | host pointers in unified memory (MATLAB arrays, NumPy) |
| MATLAB | gpuArray MEX gateways | host-array MEX gateways (`matlab/metal/`) + MATLAB's Python interface |

There is no C/C++ NUFFT for Apple GPUs, so the Toeplitz pre-computation is split in two:
`mr_aha.calc_toep` (Python) evaluates the type-1 NUFFT of each triangular component with mlx-nufft,
then the library's `mr_calc_toep_from_psf` performs the inverse FFT, scaling and mask gather on the
GPU. The C function
`mr_calc_toep` returns `MR_ERR_UNSUPPORTED` on Metal with directions; the operator itself
(`mr_aha_plan_*`, `mr_aha_toeplitz`) and the mask (`mr_calc_mask`, which needs no NUFFT) are
complete in C. `mr_calc_toep_from_psf` also exists in the CUDA backend.

**Requirements.** macOS 13 or newer on Apple silicon, CMake ≥ 3.24, Apple clang (Xcode or its
Command Line Tools), Python 3.11/3.12 with `numpy`, `mlx==0.31.2` and mlx-nufft. Optional: Xcode's
Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), which lets CMake precompile the
shaders; without it they are compiled by macOS's built-in Metal compiler on first use (~0.15 s,
cached by macOS afterwards).

```bash
git submodule update --init third_party/VkFFT third_party/mlx-nufft
cmake -S . -B build -DMR_AHA_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2024b.app
cmake --build build -j

uv venv --python 3.12 .venv                     # 3.12: newest Python MATLAB R2024b's pyenv accepts
uv pip install -p .venv/bin/python numpy scipy "mlx==0.31.2" -e third_party/mlx-nufft
```

Python: `import mr_aha` with `python/` on the path (or `pip install -e python`); arrays are NumPy.
MATLAB: put `matlab/` and `build/matlab/` on the path; `mr_python` loads `.venv/bin/python`
in-process (or the interpreter named by `MR_AHA_PYTHON`) the first time a NUFFT is needed.

**Validation.** `tests/check_reference.py` and `tests/check_reference_matlab.m` (via
`tests/run_matlab_tests.m`) compare every public function with CUDA reference outputs
(`tests/make_reference.py`): 2D, 3D and stack-of-stars, odd sizes and a prime FFT length,
L = 1, 3, 4, both triangulations, unmasked and masked. The operator matches CUDA to 2-3e-7; the
operator built by the mlx-nufft `calc_toep` matches the stored one to 0.8-1.9e-6; the mask is
identical.

**VkFFT fork.** `third_party/VkFFT` points to [mikecjz/VkFFT](https://github.com/mikecjz/VkFFT)
(branch `mr_aha`): upstream `develop` plus two Metal fixes, an autorelease double free
(DTolm/VkFFT#226) and push constants that were overwritten when several dispatches shared a
command buffer. Details in [`benchmarks/vkfft/README.md`](../benchmarks/vkfft.md).

**Size limit.** One VkFFT transform on Metal must cover fewer than 2^31 complex values (VkFFT's
32-bit indexing; its 64-bit mode does not compile on Metal, and launch offsets cannot get around
it). The operator therefore requires `Mx·My·Mz·L < 2^31`, e.g. `L ≤ 65` for a 320^3 grid, and
returns `MR_ERR_UNSUPPORTED` beyond that; `calc_toep` chunks its components to stay below it.

**Accuracy of `calc_toep` on Metal.** mlx-nufft is a single-precision NUFFT, like the cufinufft
used on CUDA, and both are accurate to about 1e-6 to 3e-5 on large 3D grids, depending on the
input. On the 3D kooshball benchmark (smooth Gram blocks, a strongly peaked point-spread
function) the Metal operator differs from the NUFFT reference by 1.4e-5 per application, against
1.9e-6 for CUDA. Where that matters, `mr_aha.calc_toep(..., nufft='finufft')` computes the
point-spread functions with CPU FINUFFT in double precision: 5e-6, at about 4x the pre-compute
time (67 s instead of 15 s on an M3 Ultra). 2D and stack-of-stars are at CUDA's level (1e-6)
with the default.

**Performance tuning.** The block-diagonal stage, which is bound by reading `phi2_toep`, uses
kernels specialised on L (L ≤ 32; generic ones otherwise) and processes the coils in groups
(default 4, `MR_AHA_METAL_COIL_GROUP`), two at a time from registers, so `phi2_toep` is read far
fewer times than once per coil. Together they cut the GPU time of one A^H A by 16-23 % on the
benchmark problems; profile with `MR_AHA_METAL_PROFILE=1`.

**Environment variables** (testing and diagnostics): `MR_AHA_METAL_FROM_SOURCE=1` compiles the
embedded shader source even when a precompiled library is embedded; `MR_AHA_METAL_FORCE_COPY=1`
copies caller arrays instead of wrapping them in place; `MR_AHA_METAL_COIL_GROUP=<n>` sets the
coil group size; `MR_AHA_METAL_GENERIC_BD=1` uses the
generic block-diagonal kernels instead of the ones specialised on L; `MR_AHA_METAL_PROFILE=1`
prints the GPU time of each operator stage; `MR_AHA_PYTHON` selects MATLAB's Python.

**Distributing a prebuilt library.** No Xcode is needed at run time (shaders are precompiled or
compiled by macOS itself). A downloaded, unsigned `.dylib` / `.mexmaca64` is quarantined by
Gatekeeper: clear it with `xattr -dr com.apple.quarantine <folder>` or sign and notarise the binaries.
