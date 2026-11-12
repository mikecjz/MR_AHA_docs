# MR_AHA

```{image} _static/banner-light.svg
:class: only-light
:alt: "MR-A^H A: GPU forward normal operator for non-Cartesian MRI"
```

```{image} _static/banner-dark.svg
:class: only-dark
:alt: "MR-A^H A: GPU forward normal operator for non-Cartesian MRI"
```

MR_AHA evaluates the forward normal operator $A^H A$ on the GPU for **non-Cartesian MRI with
low-rank (subspace / MR-Multitasking) modelling**, via a Toeplitz embedding: after a one-off
pre-computation, one application costs two FFTs per coil instead of two NUFFTs. A NUFFT-based
operator with the same interface is included as the reference.

The numerics live in a standalone C++ library with a pure C surface. MATLAB, Python and plain
C/C++ all call the same code. The library is built for one GPU backend: **CUDA** (NVIDIA GPUs,
one or several) or **Metal** (Apple silicon).

Everything runs on the GPU. With CUDA all array arguments are device pointers (`gpuArray` in
MATLAB, `cupy.ndarray` in Python) and the library never copies bulk data through the host. With
Metal they are ordinary host arrays (MATLAB arrays, `numpy.ndarray`), which the Apple GPU reads
in place through unified memory.

## Quick start

Build the library ([Installation](getting-started/install.md)), then create the operator once
from the sensitivity maps `SEs`, the Gram blocks `phi2` and the trajectory `traj`, and apply it
as often as needed, for example inside conjugate gradient.

::::{tab-set-code}

```python
import mr_aha

cfg = mr_aha.make_config("stack", (320, 320, 56), L=16, n_coils=12)

with mr_aha.create_op(cfg, SEs, phi2, traj) as op:   # Toeplitz kernel computed here
    for _ in range(n_iter):
        AhAx = op(x)
```

```matlab
config = mr_default_config('stack', [320 320 56], 16);

op = mr_aha.create_op(SEs, phi2, traj, config);   % Toeplitz kernel computed here
for it = 1:n_iter
    AhAx = op * x;
end
delete(op);                                          % frees the GPU memory
```

::::

## Where to go next

| | |
|---|---|
| [Installation](getting-started/install.md) | requirements, CMake options |
| [Python](getting-started/python.md), [MATLAB](getting-started/matlab.md), [C / C++](getting-started/c.md) | using the operator from each language |
| [The operations](guide/operations.md) | `create_op`, `calc_toep`, `AhA_Toeplitz`, `calc_mask` and the NUFFT reference |
| [Conventions](guide/conventions.md) | array shapes, `config`, the Gram-block convention, triangular packing |
| [Masking](guide/masking.md) | the masked operator and how the mask is chosen |
| [Multi-GPU](guide/multi-gpu.md) | several GPUs, how the kernel is distributed |
| [Apple silicon](guide/apple-silicon.md) | the Metal backend |
| [Why Toeplitz?](background/toeplitz.md) | the method, the [implementation notes](background/implementation.md), and [how the CUDA operator gets its speed](background/performance.md) |
| [Benchmarks](benchmarks/index.md) | measured speed and accuracy, and scripts to measure your own |
| API reference | [Python](api/python/mr_aha/index), [C](api/c.md), [MATLAB](api/matlab/index.md) |

## Repositories

* [MR_AHA](https://github.com/mikecjz/MR_AHA): the library.
* [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks): Python and MATLAB scripts
  that time the Toeplitz and NUFFT operators on one or several GPUs.
* [MR_AHA_docs](https://github.com/mikecjz/MR_AHA_docs): the sources of this site.

```{toctree}
:caption: Getting started
:hidden:

getting-started/install
getting-started/python
getting-started/matlab
getting-started/c
```

```{toctree}
:caption: User guide
:hidden:

guide/operations
guide/conventions
guide/masking
Multi-GPU <guide/multi-gpu>
Apple silicon <guide/apple-silicon>
```

```{toctree}
:caption: Background
:hidden:

Why Toeplitz? <background/toeplitz>
background/implementation
Performance notes <background/performance>
```

```{toctree}
:caption: Benchmarks
:hidden:

Overview <benchmarks/index>
Main study <benchmarks/results>
Neighbourhood mask <benchmarks/neighbour-mask>
CUDA optimisation and multi-GPU <benchmarks/cuda-multigpu>
NUFFT upsampling factor <benchmarks/upsampfac>
VkFFT on Metal <benchmarks/vkfft>
```

```{toctree}
:caption: API reference
:hidden:

Python <api/python/mr_aha/index>
C <api/c>
MATLAB <api/matlab/index>
```

```{toctree}
:caption: Development
:hidden:

development/testing
VS Code <development/vscode>
development/contributing
development/licensing
```
