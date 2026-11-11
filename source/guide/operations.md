# The operations

## `create_op` — the operator object

```
op   = mr_aha_op.create_op(SEs, phi2, traj, config, ...)       % MATLAB
op   = mr_aha.create_op(cfg, SEs, phi2, traj, ...)             # Python
       mr_op_create(&cfg, &inputs, &options, &op)              /* C */
AhAx = op * x  /  op(x)  /  mr_op_execute(op, d_x, d_AhAx)
```

The one stateful object of the library: the complete operator $x \mapsto A^H A x$, built once from
the Gram blocks, the trajectory and the sensitivity maps, then applied as often as an iterative
reconstruction needs. Two methods evaluate the same operator:

* `method = 'toeplitz'` (default): the kernel `phi2_toep` is computed at creation (`calc_toep`
  below) and every application is the Toeplitz pipeline of `AhA_Toeplitz` below. Fast per
  application; the kernel is the largest array.
* `method = 'nufft'`: every application runs, per coil, a type-2 NUFFT, the Gram multiplication on
  the samples and a type-1 NUFFT (cufinufft, inside the library). No kernel, so far less memory,
  but slower per application. It is the baseline the Toeplitz method is compared against.

`devices` puts the operator on several GPUs of one node (coils split over the GPUs, see
[Multi-GPU](multi-gpu.md)); by default it runs on the current GPU only, at the cost of the
single-GPU primitives below.

## `calc_toep` — one-off pre-computation

```
phi2_toep = calc_toep(phi2, traj, config)
```

Turns the non-Cartesian Gram blocks $\Phi_2 = \Phi\Phi^H$ (MR-Multitasking's
`Phi_rt(:,idx) * Phi_rt(:,idx)'`, see [Gram blocks](conventions.md#gram-blocks-mr-multitasking-convention)), one
$L \times L$ matrix per trajectory, into the Cartesian block-diagonal multiplier used by the
operator. Internally: pack each block into a
triangle, type-1 cufinufft onto the upsampled grid, centred inverse FFT, optional mask gather.
This is a stateless, single-GPU primitive; the Toeplitz operator calls it internally (and, on
several GPUs, spreads its $(L^2+L)/2$ independent triangular components over the GPUs, see
[Multi-GPU](multi-gpu.md)).

> **Experimental: NUFFT settings of `calc_toep`.** Environment variables, read when the kernel
> is computed (by `calc_toep` or `create_op`), change its type-1 transform.
> `MR_AHA_TOEP_UPSAMPFAC` sets the NUFFT's internal upsampling factor σ (default **2.0**; 1.25 is
> the other value with cufinufft's fast Horner kernel) on both backends: cufinufft on CUDA,
> mlx-nufft (`mr_aha._psf`) on Apple silicon. `MR_AHA_TOEP_GPU_METHOD` (CUDA only) sets
> cufinufft's spreading method (default 0 = cufinufft's choice, 1 = GM-sort, 2 = shared memory).
> σ only sizes the NUFFT's fine grid; the Toeplitz grid `phi2_dim` is unchanged. At the same
> `eps`, σ = 1.25 is *slower* on both backends (the grid is dense, so the 50 % wider spreading
> kernel outweighs the smaller FFT). On CUDA it pays off only at a looser `eps`, e.g. 1.7× faster
> for the 3D kooshball at `eps = 1e-4` with an error similar to σ = 2 at `eps = 1e-5`; on Apple
> silicon it is 1.1–1.8× slower for stack and 3D at every `eps`. GM-sort is 1.2–2.5× faster but loses
> accuracy in 3D. These switches are for benchmarking and may change; see
> [benchmarks/README.md](../benchmarks/upsampfac.md).

## `AhA_Toeplitz` — the hot loop

```
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config)
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask)
```

For every coil: multiply by the sensitivity map, zero-pad onto the upsampled grid, FFT, multiply the
rank coefficients at each k-space location, as a row vector from the right, by that location's
Hermitian $L \times L$ block ($v \leftarrow v\,B$, as MR-Multitasking does), inverse FFT, crop,
multiply by the adjoint sensitivity map, accumulate. No NUFFT is evaluated here, which is the whole point of the
Toeplitz embedding. Stateless and single-GPU (`mr_aha_toeplitz` / `aha_toeplitz`); the operator object
applies the same pipeline with its buffers kept between applications.

## `calc_mask` — automatic mask for the masked operator

```
mask = calc_mask(traj, config)
```

Derives the upsampled k-space locations worth keeping from the trajectory alone. By default it keeps
every grid location within `config.mask_radius` (default 2) cells of a sample. The alternative
`config.mask_method = 'threshold'` thresholds the trajectory's unit-weight Toeplitz kernel at
`config.mask_threshold`. See [Masking](masking.md).

## `AhA_cufinufft` / `mr_aha.AhANufftPlan` — reference without Toeplitz

```
AhAx = AhA_cufinufft(x, SEs, phi2, traj, config[, plan])     % MATLAB
y    = mr_aha.AhANufftPlan(cfg, traj).apply(x, SEs, phi2)    # Python
```

The same operator evaluated the "vanilla" way, coil by coil: type-2 cufinufft, one $L\times L$ block
per readout, type-1 cufinufft. Pure MATLAB / CuPy on top of finufft's own interfaces (no MR_AHA C
code): the **independent reference** the tests validate both methods of the operator against
(`method = 'nufft'` runs the same algorithm inside the library). All agree to $\sim 10^{-6}$.
