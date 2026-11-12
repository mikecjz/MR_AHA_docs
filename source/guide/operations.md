# The operations

The library offers its operations at two levels:

* **Packaged operation.** `create_op` builds the complete $A^H A$ operator once, and the object
  it returns is then applied as often as needed. It computes the kernel, keeps the sensitivity
  maps and every working buffer, and can be multi-GPU. Use it for reconstructions.
* **Standalone operations.** `calc_toep`, `AhA_Toeplitz` and `calc_mask` are the individual
  steps the packaged operator is built from, each available on its own. They are stateless and
  single-GPU: every call takes all its inputs and allocates its own buffers. They are for users
  who prefer granular control: inspecting or saving the kernel, reusing it across sessions, or
  building a pipeline of their own.

## Packaged operation: `create_op` and apply

::::{tab-set-code}

```python
op = mr_aha.create_op(cfg, SEs, phi2, traj, ...)
AhAx = op(x)
```

```matlab
op = mr_aha.create_op(SEs, phi2, traj, config, ...);
AhAx = op * x;
```

```c
mr_op_create(&cfg, &inputs, &options, &op);
mr_op_execute(op, d_x, d_AhAx);
```

::::

`create_op` builds the one stateful object of the library: the complete operator
$x \mapsto A^H A x$. It is built once from the Gram blocks, the trajectory and the sensitivity
maps. Applying it then takes only $x$, as often as an iterative reconstruction needs. Two methods
evaluate the same operator:

* `method = 'toeplitz'` (default). The kernel `phi2_toep` is computed at creation
  ([`calc_toep`](#calc_toep--one-off-pre-computation)), and every application is the Toeplitz
  pipeline of [`AhA_Toeplitz`](#aha_toeplitz--the-hot-loop). This is the fast method. The kernel
  is its largest array.
* `method = 'nufft'`. Every application runs, per coil, a type-2 NUFFT, the Gram multiplication on
  the samples and a type-1 NUFFT (cufinufft, inside the library). It needs no kernel, so far less
  memory, and takes longer per application. It is the baseline the Toeplitz method is compared
  against.

By default the operator runs on the current GPU, at the same cost as the standalone operations
below. `devices` makes it multi-GPU on one node, with the coils split over the GPUs; see
[Multi-GPU](multi-gpu.md).

## Standalone operations

These are for users who prefer granular control over each step. You decide when the kernel is
computed, where it is stored and how it is applied. Each operation runs once, on the current GPU,
and keeps nothing between calls. They call the same C functions as `create_op` and compute the
same operator.

### `calc_toep` — one-off pre-computation

::::{tab-set-code}

```python
phi2_toep = mr_aha.calc_toep(cfg, phi2, traj)          # unmasked
phi2_toep = mr_aha.calc_toep(cfg, phi2, traj, mask)    # masked
```

```matlab
phi2_toep = calc_toep(phi2, traj, config);
```

```c
mr_calc_toep(&cfg, d_phi2, d_traj, Ntrajs, Nkx, d_mask, d_phi2_toep);
```

::::

`calc_toep` turns the non-Cartesian Gram blocks $\Phi_2 = \Phi\Phi^H$ into the Cartesian
block-diagonal multiplier used by the operator. There is one $L \times L$ Gram block per
trajectory (`Phi_rt(:,idx) * Phi_rt(:,idx)'`, see
[Gram blocks](conventions.md#gram-blocks)). The steps are:

1. pack each block into a triangle;
2. type-1 cufinufft onto the upsampled grid;
3. centred inverse FFT;
4. optional mask gather.

The packaged Toeplitz operator calls `calc_toep` internally. A multi-GPU operator spreads the $(L^2+L)/2$ independent triangular components over the
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
> The defaults are the recommended settings. Two settings are faster on CUDA: σ = 1.25 with
> `eps = 1e-4` computes the 3D kooshball kernel 1.7× faster than σ = 2 with `eps = 1e-5`, at a
> similar error, and GM-sort is 1.3–1.6× faster for 2D and stack at the same accuracy. These
> switches are for benchmarking and may change; see
> [NUFFT upsampling factor](../benchmarks/upsampfac.md) for the measurements.

### `AhA_Toeplitz` — the hot loop

::::{tab-set-code}

```python
AhAx = mr_aha.aha_toeplitz(cfg, x, SEs, phi2_toep)
AhAx = mr_aha.aha_toeplitz(cfg, x, SEs, phi2_toep, mask)
```

```matlab
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config);
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask);
```

```c
mr_aha_toeplitz(&cfg, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx);   /* d_mask = NULL when unmasked */
```

::::

For every coil:

1. multiply by the sensitivity map;
2. zero-pad onto the upsampled grid;
3. FFT;
4. at each k-space location, multiply the rank coefficients, as a row vector from the right, by
   that location's Hermitian $L \times L$ block ($v \leftarrow v\,B$);
5. inverse FFT;
6. crop;
7. multiply by the adjoint sensitivity map and accumulate.

No NUFFT is evaluated here, which is the whole point of the Toeplitz embedding. The packaged
operator applies the same pipeline, with its buffers kept between applications.

### `calc_mask` — automatic mask for the masked operator

::::{tab-set-code}

```python
mask = mr_aha.calc_mask(cfg, traj)
```

```matlab
mask = calc_mask(traj, config);
```

```c
mr_calc_mask(&cfg, d_traj, Ntrajs, Nkx, d_mask, &num_masked);
```

::::

`calc_mask` derives the upsampled k-space locations worth keeping from the trajectory alone: it
keeps every grid location within `config.mask_radius` (default 2) cells of a sample. A masked
operator from MATLAB's `mr_aha.create_op` calls it when no mask is given; in Python and C, pass
its result to `create_op`. See [Masking](masking.md).

### `aha_nufft_plan` — reference without Toeplitz

::::{tab-set-code}

```python
plan = mr_aha.aha_nufft_plan(cfg, traj)        # once per trajectory
AhAx = plan.apply(x, SEs, phi2)                # every application; plan.execute is an alias
```

```matlab
plan = aha_nufft_plan(traj, config);            % once per trajectory
AhAx = plan.apply(x, SEs, phi2);                % every application; plan.execute is an alias
delete(plan);                                   % frees the cufinufft plans
```

::::

`aha_nufft_plan` has the same name and the same two steps in both languages: create a plan
object for the trajectory, then apply it. For a single application, `mr_aha.aha_nufft` (Python)
and `AhA_cufinufft` (MATLAB) build a plan, apply it once and release it.

This is the same operator evaluated without the Toeplitz embedding, coil by coil: type-2
cufinufft, one $L\times L$ block per readout, type-1 cufinufft. It is pure MATLAB / CuPy on top of
finufft's own interfaces and uses no MR_AHA C code. That makes it the **independent reference**
the tests validate both methods of the operator against (`method = 'nufft'` runs the same
algorithm inside the library). All three agree to $\sim 10^{-6}$.

**Why a plan.** A NUFFT spends part of its time on set-up that depends only on the trajectory:
cufinufft sorts the non-uniform points into bins on its fine grid, chooses the spreading
parameters for the requested `eps`, and allocates its work arrays. The plan does this once, for
one type-2 and one type-1 transform, and every application then only spreads, transforms and
interpolates. The one-shot functions repeat the set-up on every call, so inside an iterative
reconstruction build the plan once and reuse it. The Toeplitz path needs no plan per application, because
`calc_toep` uses the trajectory once and the result, `phi2_toep`, lives on a Cartesian grid.
