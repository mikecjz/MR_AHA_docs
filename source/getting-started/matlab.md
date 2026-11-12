# MATLAB

Put both `matlab/` and `build/matlab/` on the MATLAB path. Then create the operator once and
apply it as often as needed:

```matlab
config = mr_default_config('stack', [320 320 56], 16);

op = mr_aha.create_op(SEs, phi2, traj, config);   % once: kernel computed here
for it = 1:n_iter
    AhAx = op * x;                                   % inside the CG loop (or op.apply(x))
end
delete(op);                                          % frees the GPU memory
```

`create_op` takes name-value options (see `help mr_aha`):

| Option | Values |
|---|---|
| `'method'` | `'toeplitz'` (default) or `'nufft'` |
| `'devices'` | `[]` (the current GPU), `'all'`, or 1-based `gpuDevice` indices; see [Multi-GPU](../guide/multi-gpu.md) |
| `'mask'` | the mask of a masked operator |
| `'toep_dist'`, `'share_primary'` | multi-GPU options; see [Multi-GPU](../guide/multi-gpu.md) |
| `'nufft_opts'` | options of the NUFFT method |

## Masked operator

The mask is derived from the trajectory:

```matlab
config = mr_default_config('stack', [320 320 56], 16, 'is_masked', true);
mask   = calc_mask(traj, config);                    % 1-based int64 gpuArray
op     = mr_aha.create_op(SEs, phi2, traj, config, 'mask', mask);
```

## Low-level primitives

The stateless single-GPU primitives are `Om = calc_toep(phi2, traj, config)` and
`AhAx = AhA_Toeplitz(x, SEs, Om, config[, mask])`.

## NUFFT reference

The reference needs `-DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON` and `build/finufft/matlab` on the path,
or any other cufinufft MATLAB installation:

```matlab
plan = mr_nufft_plan(traj, config);             % once: plans + point sorting
AhAx = AhA_cufinufft(x, SEs, phi2, traj, config, plan);
```

## What the wrappers do

The `.m` wrappers:

* move host arrays to the GPU and cast them to complex `single`;
* derive `n_coils` from `size(SEs)`;
* convert the 1-based mask to the 0-based `int64` mask the gateway expects.

Call `AhA_Toeplitz_mex` / `calc_toep_mex` directly only if you want to skip that.

```{note}
MATLAB must have initialised its GPU before the first MEX call. Running `gpuDevice(1)`, or
creating any `gpuArray`, is enough. The wrappers do this implicitly by moving their inputs.
```

## Apple silicon

On Apple silicon the same functions take ordinary MATLAB arrays, and no Parallel Computing
Toolbox is needed. `calc_toep`, `mr_aha.create_op` and `AhA_cufinufft` run their NUFFTs through
MATLAB's Python interface. See [Apple silicon](../guide/apple-silicon.md).
