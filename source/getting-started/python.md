# Python

## Install

```bash
cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON && cmake --build build -j
pip install -e python
pip install cupy-cuda12x        # match your CUDA toolkit
```

## Quick start

Create one object, `AhAOp`, from the Gram blocks, the trajectory and the sensitivity maps with
`mr_aha.create_op` (or `mr_aha.AhAOp.create_op`). Then apply it as often as needed:

```python
import mr_aha

cfg = mr_aha.make_config("stack", (320, 320, 56), L=16, n_coils=12)

with mr_aha.create_op(cfg, SEs, phi2, traj) as op:  # kernel computed here
    for _ in range(n_iter):
        y = op(x)                                    # or op.apply(x, out=y)
    print(op.info)                                   # method, devices, timings
```

`SEs`, `phi2`, `traj` and `x` are CuPy arrays. On Apple silicon they are NumPy arrays instead;
everything else is the same. See [Apple silicon](../guide/apple-silicon.md).

## Array layout

The library documents arrays in **column-major** (MATLAB) order. NumPy and CuPy are row-major, so
every shape of the [array-shape table](../guide/conventions.md#array-shapes) is reversed here.
The memory layout is identical, and no data is ever moved.

| column-major (docs)          | row-major (this package)   |
| ---------------------------- | -------------------------- |
| `x[Nx, Ny, Nz, L]`           | `x[L, Nz, Ny, Nx]`         |
| `SEs[Nx, Ny, Nz, Nc]`        | `SEs[Nc, Nz, Ny, Nx]`      |
| `phi2_toep[Mx, My, Mz, T]`   | `phi2_toep[T, Mz, My, Mx]` |
| `phi2[Ntrajs, L, L]`         | `phi2[L, L, Ntrajs]`       |
| `phi2[Ntrajs, Nkz, L, L]`    | `phi2[L, L, Nkz, Ntrajs]`  |
| `traj[Ntrajs, Nkx, ndim]`    | `traj[ndim, Nkx, Ntrajs]`  |

**Gram blocks** follow MR-Multitasking. `phi2` holds `Phi_rt(:,idx) * Phi_rt(:,idx)'` of the
rank x time basis, and every block multiplies the rank coefficients of a k-space location as a
row vector from the right (`k * Phi2`). Row-major `phi2[a, b, ...]` holds the column-major entry
`(b, a)`, so that is `out[l] = sum_m phi2[l, m] * v[m]`.

A MR-Multitasking `Phi2` read with h5py is row-major `(Nz, Ntrajs, L, L)`. Pass it as
`P.transpose(2, 3, 0, 1)`, plus an `fftshift` along kz if it is in natural order. See
[Gram blocks](../guide/conventions.md#gram-blocks-mr-multitasking-convention).

## Options of `create_op`

| argument | default | meaning |
| --- | --- | --- |
| `method` | `"toeplitz"` | `"toeplitz"`: the kernel `phi2_toep` is computed at creation, and every application is FFT + block-diagonal multiplication. `"nufft"`: every application runs a type-2 NUFFT, the Gram multiplication on the samples and a type-1 NUFFT per coil (cufinufft, inside the library). It needs no kernel and much less memory, and takes longer per application. |
| `devices` | `None` | `None`: the current GPU only. `"all"`: every visible GPU. A list of CUDA ordinals: those GPUs (the current one always takes part). Coils are split over the GPUs. |
| `toep_dist` | `"auto"` | Multi-GPU Toeplitz: how every GPU gets the kernel. `"split"`: each computes a share, and the shares are exchanged. `"replicate"`: each computes all of it. `"broadcast"`: one computes and copies to the others. `"auto"`: a cost model chooses. |
| `share_primary` | `True` | Use `SEs` (and `mask`) in place instead of copying them. Do not modify them while the operator is open. |
| `nufft_opts` | `None` | `method="nufft"`: `{"upsampfac": 2.0, "gpu_method": 0}`. |

`op.kernel(k)` returns a read-only view of the Toeplitz kernel the operator holds on its `k`-th
GPU, for example to save it.

## Masked operator

A masked operator keeps `phi2_toep` only at selected upsampled k-space samples (Toeplitz method
only). `calc_mask` derives them from the trajectory: it keeps every grid location within
`mask_radius=2` cells of a sample. `calc_mask(cfg, traj, radius=...)` overrides the configured
radius for one call.

```python
cfg  = mr_aha.make_config("3D", (160, 160, 160), L=12, n_coils=12, is_masked=True)
mask = mr_aha.calc_mask(cfg, traj)                   # or your own indices
op   = mr_aha.create_op(cfg, SEs, phi2, traj, mask)  # also sets cfg.num_masked
```

`mask` holds **0-based** linear indices into the centred upsampled grid. See
[Masking](../guide/masking.md).

## Multi-GPU (CUDA)

```python
with mr_aha.create_op(cfg, SEs, phi2, traj, devices=[0, 1, 2, 3]) as op:
    for _ in range(n_iter):
        y = op(x)     # x and y on the current CuPy device
    print(op.info)    # devices used, coils per device, how the kernel was made
```

Every GPU applies its own coils, and the partial results are summed on the current CuPy device.

* With the Toeplitz method every GPU holds the whole kernel, which the GPUs compute in parallel
  (`toep_dist`).
* With the NUFFT method every GPU holds its own NUFFT plans.

The library may use fewer GPUs than given (the smallest number with the same coils per GPU), and
it keeps small problems on one GPU. See [Multi-GPU](../guide/multi-gpu.md).

## NUFFT method

```python
cfg_nu = mr_aha.make_config("3D", (160, 160, 160), L=12, n_coils=12)
op_nu  = mr_aha.create_op(cfg_nu, SEs, phi2, traj, method="nufft")
```

## NUFFT reference

`mr_aha.AhANufftPlan` evaluates the same operator with type-2 / type-1 cufinufft transforms and
no Toeplitz embedding. It is pure CuPy on top of the `cufinufft` package and uses no MR_AHA C
code. That makes it the **independent reference** the tests validate both methods of `create_op`
against (`method="nufft"` runs the same algorithm inside the library). Use `create_op` for
reconstructions.

It needs `pip install cufinufft`. The cufinufft wheel expects `libcudart.so.12` on the library
path: load your CUDA module or install `nvidia-cuda-runtime-cu12`.

```python
with mr_aha.AhANufftPlan(cfg, traj, type1_opts={"gpu_method": 1},
                         type2_opts={"gpu_method": 2}) as ref:
    y_ref = ref.apply(x, SEs, phi2)    # agrees with create_op(...)(x) to ~1e-6
```

## Low-level primitives

`mr_aha.calc_toep(cfg, phi2, traj, mask)` computes the kernel on the current GPU, and
`mr_aha.aha_toeplitz(cfg, x, SEs, phi2_toep, mask)` applies it once. Both are stateless;
`aha_toeplitz` allocates its buffers on every call. Together they are what the Toeplitz operator
does internally on one GPU.
