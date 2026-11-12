# Installation

Requirements (CUDA backend):

* CUDA toolkit 11.x or 12.x;
* CMake ≥ 3.24;
* a C++17 compiler;
* the bundled FINUFFT submodule (`third_party/finufft`), which is stock upstream FINUFFT.

For Apple silicon see [Apple silicon](../guide/apple-silicon.md).

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA.git
cd MR_AHA

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75           \
      -DMR_AHA_BUILD_PYTHON=ON               \
      -DMR_AHA_BUILD_MATLAB=ON               \
      -DMR_AHA_BUILD_EXAMPLES=ON
cmake --build build -j
```

| Option | Default | Produces |
| ------ | ------- | -------- |
| `MR_AHA_BACKEND` | `METAL` on macOS, else `CUDA` | which GPU backend the library is built for |
| — | | `build/libmr_aha.a` — the standalone library |
| `MR_AHA_BUILD_PYTHON` | `ON` | `build/lib/libmr_aha_c.so` (`.dylib` on macOS) — for the ctypes binding |
| `MR_AHA_BUILD_MATLAB` | `OFF` | `build/matlab/*.mexa64` / `*.mexmaca64` — needs `MATLAB_ROOT`; the CUDA gateways also need the Parallel Computing Toolbox |
| `MR_AHA_METAL_PRECOMPILE` | `ON` | Metal only: precompile the shaders when `xcrun metal` works (they are always embedded as source too) |
| `MR_AHA_BUILD_EXAMPLES` | `OFF` | `build/mr_example_aha` |
| `MR_AHA_BUILD_CUFINUFFT_MATLAB` | `OFF` | `build/finufft/matlab/cufinufft.mexa64` + `.m` wrappers — finufft's own GPU MATLAB interface, needed by `AhA_cufinufft.m` |

Set `MATLAB_ROOT` (or `-DMatlab_ROOT_DIR=...`) to your `matlabroot` before configuring with
`MR_AHA_BUILD_MATLAB=ON`.
