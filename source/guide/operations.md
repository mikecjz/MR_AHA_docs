# The operations

## `create_op` — the operator object

```
op   = mr_aha.create_op(SEs, phi2, traj, config, ...)       % MATLAB
op   = mr_aha.create_op(cfg, SEs, phi2, traj, ...)             # Python
       mr_op_create(&cfg, &inputs, &options, &op)              /* C */
AhAx = op * x  /  op(x)  /  mr_op_execute(op, d_x, d_AhAx)
```

`create_op` builds the one stateful object of the library: the complete operator
$x \mapsto A^H A x$. It is built once from the Gram blocks, the trajectory and the sensitivity
maps, then applied as often as an iterative reconstruction needs. Two methods evaluate the same
operator:

* `method = 'toeplitz'` (default). The kernel `phi2_toep` is computed at creation
  ([`calc_toep`](#calc_toep--one-off-pre-computation)), and every application is the Toeplitz
  pipeline of [`AhA_Toeplitz`](#aha_toeplitz--the-hot-loop). This is the fast method. The kernel
  is its largest array.
* `method = 'nufft'`. Every application runs, per coil, a type-2 NUFFT, the Gram multiplication on
  the samples and a type-1 NUFFT (cufinufft, inside the library). It needs no kernel, so far less
  memory, and takes longer per application. It is the baseline the Toeplitz method is compared
  against.

By default the operator runs on the current GPU, at the same cost as the single-GPU primitives
below. `devices` puts it on several GPUs of one node, with the coils split over the GPUs; see
[Multi-GPU](multi-gpu.md).

## `calc_toep` — one-off pre-computation

```
phi2_toep = calc_toep(phi2, traj, config)
```

`calc_toep` turns the non-Cartesian Gram blocks $\Phi_2 = \Phi\Phi^H$ into the Cartesian
block-diagonal multiplier used by the operator. There is one $L \times L$ Gram block per
trajectory (MR-Multitasking's `Phi_rt(:,idx) * Phi_rt(:,idx)'`, see
[Gram blocks](conventions.md#gram-blocks-mr-multitasking-convention)). The steps are:

1. pack each block into a triangle;
2. type-1 cufinufft onto the upsampled grid;
3. centred inverse FFT;
4. optional mask gather.

`calc_toep` is a stateless, single-GPU primitive. The Toeplitz operator calls it internally. On
several GPUs the operator spreads the $(L^2+L)/2$ independent triangular components over the
GPUs; see [Multi-GPU](multi-gpu.md).

> **Experimental: NUFFT settings of `calc_toep`.** Two environment variables, read when the
> kernel is computed (by `calc_toep` or `create_op`), change its type-1 transform:
>
> * `MR_AHA_TOEP_UPSAMPFAC` sets the NUFFT's internal upsampling factor σ on both backends
>   (cufinufft on CUDA, mlx-nufft on Apple silicon). The default is **2.0**; the other supported
>   value is 1.25. σ only sizes the NUFFT's fine grid; the Toeplitz grid `phi2_dim` is unchanged.
> * `MR_AHA_TOEP_GPU_METHOD` (CUDA only) sets cufinufft's spreading method: 0 = cufinufft's
>   choice (default), 1 = GM-sort, 2 = shared memory.
>
> The defaults are the recommended settings. One combination is faster on CUDA: σ = 1.25 with
> `eps = 1e-4` computes the 3D kooshball kernel 1.7× faster than σ = 2 with `eps = 1e-5`, at a
> similar error. These switches are for benchmarking and may change; see
> [NUFFT upsampling factor](../benchmarks/upsampfac.md) for the measurements.

## `AhA_Toeplitz` — the hot loop

```
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config)
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask)
```

For every coil:

1. multiply by the sensitivity map;
2. zero-pad onto the upsampled grid;
3. FFT;
4. at each k-space location, multiply the rank coefficients, as a row vector from the right, by
   that location's Hermitian $L \times L$ block ($v \leftarrow v\,B$, as MR-Multitasking does);
5. inverse FFT;
6. crop;
7. multiply by the adjoint sensitivity map and accumulate.

No NUFFT is evaluated here, which is the whole point of the Toeplitz embedding. `AhA_Toeplitz`
(`mr_aha_toeplitz` in C, `aha_toeplitz` in Python) is stateless and single-GPU. The operator
object applies the same pipeline with its buffers kept between applications.

## `calc_mask` — automatic mask for the masked operator

```
mask = calc_mask(traj, config)
```

`calc_mask` derives the upsampled k-space locations worth keeping from the trajectory alone: it
keeps every grid location within `config.mask_radius` (default 2) cells of a sample. See
[Masking](masking.md).

## `AhA_cufinufft` / `mr_aha.AhANufftPlan` — reference without Toeplitz

```
AhAx = AhA_cufinufft(x, SEs, phi2, traj, config[, plan])     % MATLAB
y    = mr_aha.AhANufftPlan(cfg, traj).apply(x, SEs, phi2)    # Python
```

This is the same operator evaluated without the Toeplitz embedding, coil by coil: type-2
cufinufft, one $L\times L$ block per readout, type-1 cufinufft. It is pure MATLAB / CuPy on top of
finufft's own interfaces and uses no MR_AHA C code. That makes it the **independent reference**
the tests validate both methods of the operator against (`method = 'nufft'` runs the same
algorithm inside the library). All three agree to $\sim 10^{-6}$.
