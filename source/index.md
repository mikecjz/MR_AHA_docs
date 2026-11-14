# MR_AHA

```{image} _static/banner-light.svg
:class: only-light
:alt: "MR-A^H A: GPU forward normal operator for non-Cartesian MRI"
```

```{image} _static/banner-dark.svg
:class: only-dark
:alt: "MR-A^H A: GPU forward normal operator for non-Cartesian MRI"
```

MR_AHA is a highly optimized GPU library for evaluating the forward normal operator $A^H A$ for
**non-Cartesian MRI with low-rank (subspace) modelling**. It uses a Toeplitz embedding; a
NUFFT-based operator with the same interface is included as the reference.

In the literature, most low-rank reconstructions, such as MR Multitasking, MR fingerprinting,
T2 shuffling and GRASP-Pro, take an hour or more per data set. A few examples among many:

- **MR Multitasking:** a median of 71 minutes for cardiovascular imaging on a GPU
  workstation[^multitasking].
- **MR fingerprinting:** over 4 hours for a whole-brain 3D subspace reconstruction in
  BART[^vista-mrf].
- **T2 shuffling:** about 60 minutes for the knee[^t2sh].
- **GRASP-Pro:** 6 to 9 hours per slice for DCE-MRI on a desktop CPU[^grasp-pro].

**MR_AHA aims to bring this down to seconds on modern GPU hardware.**

[^multitasking]: A. G. Christodoulou et al., "Magnetic resonance multitasking for
    motion-resolved quantitative cardiovascular imaging," *Nature Biomedical Engineering*
    2(4):215–226, 2018. [doi:10.1038/s41551-018-0217-y](https://doi.org/10.1038/s41551-018-0217-y)

[^vista-mrf]: C. Liao, X. Cao et al., "High-resolution myelin-water fraction and quantitative
    relaxation mapping using 3D ViSTa-MR fingerprinting," *Magnetic Resonance in Medicine*
    91(6):2278–2293, 2024. [doi:10.1002/mrm.29990](https://doi.org/10.1002/mrm.29990)

[^t2sh]: S. Bao, J. I. Tamir et al., "Fast comprehensive single-sequence four-dimensional
    pediatric knee MRI with T2 shuffling," *Journal of Magnetic Resonance Imaging*
    45(6):1700–1711, 2017. [doi:10.1002/jmri.25508](https://doi.org/10.1002/jmri.25508)

[^grasp-pro]: A. J. Mertens et al., "Accelerated dynamic magnetic resonance imaging from
    Spatial-Subspace Reconstructions (SPARS)," *PLOS ONE* 20(1):e0317271, 2025, which reports
    the GRASP-Pro reconstruction times (Tables 1 and 2).
    [doi:10.1371/journal.pone.0317271](https://doi.org/10.1371/journal.pone.0317271)

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

::::{tab-set}

:::{tab-item} Python
:sync: python

```python
import mr_aha

cfg = mr_aha.make_config("stack", (320, 320, 56), L=16, n_coils=12)

with mr_aha.create_op(cfg, SEs, phi2, traj) as op:   # Toeplitz kernel computed here
    for _ in range(n_iter):
        AhAx = op(x)
```

:::

:::{tab-item} MATLAB
:sync: matlab

```matlab
config = mr_default_config('stack', [320 320 56], 16);

op = mr_aha.create_op(SEs, phi2, traj, config);   % Toeplitz kernel computed here
for it = 1:n_iter
    AhAx = op * x;
end
delete(op);                                          % frees the GPU memory
```

:::

::::

## Where to go next

| | |
|---|---|
| [Installation](getting-started/install.md) | requirements, CMake options |
| [Python](getting-started/python.md), [MATLAB](getting-started/matlab.md), [C / C++](getting-started/c.md) | using the operator from each language |
| [The operations](guide/operations.md) | `create_op`, `calc_toep`, `AhA_Toeplitz`, `calc_mask` and the NUFFT reference |
| [Conventions](guide/conventions.md) | array shapes, `config`, the Gram-block convention, triangular packing |
| [Masking](guide/masking.md) | the masked operator and how the mask is chosen |
| [Multi-GPU](guide/multi-gpu.md) | spreading the coils over GPUs, how the kernel is distributed |
| [Apple silicon](guide/apple-silicon.md) | the Metal backend |
| [Why Toeplitz?](background/toeplitz.md) | the method, the [implementation notes](background/implementation.md), and [how the CUDA operator gets its speed](background/performance.md) |
| [Benchmarks](benchmarks/index.md) | measured speed and accuracy, and scripts to measure your own |
| API reference | [Python](api/python/mr_aha/index), [C](api/c.md), [MATLAB](api/matlab/index.md) |

## Repositories

* [MR_AHA](https://github.com/mikecjz/MR_AHA): the library.
* [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks): Python and MATLAB scripts
  that time the Toeplitz and NUFFT operators on one GPU and multi-GPU.
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
Other libraries <benchmarks/other-libraries>
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
