# MATLAB

Put both `matlab/` and `build/matlab/` on the MATLAB path.

```matlab
config = mr_default_config('stack', [320 320 56], 16);

op = mr_aha_op.create_op(SEs, phi2, traj, config);   % once: kernel computed here
for it = 1:n_iter
    AhAx = op * x;                                   % inside the CG loop (or op.apply(x))
end
delete(op);                                          % frees the GPU memory
```

Name-value options: `'method'` (`'toeplitz'` or `'nufft'`), `'devices'` (`[]`: the current GPU,
`'all'`, or 1-based `gpuDevice` indices), `'mask'`, `'toep_dist'`, `'share_primary'`,
`'nufft_opts'`; see `help mr_aha_op`. Masked operator, with the mask derived from the trajectory:

```matlab
config = mr_default_config('stack', [320 320 56], 16, 'is_masked', true);
mask   = calc_mask(traj, config);                    % 1-based int64 gpuArray
op     = mr_aha_op.create_op(SEs, phi2, traj, config, 'mask', mask);
```

The stateless single-GPU primitives remain: `Om = calc_toep(phi2, traj, config)` and
`AhAx = AhA_Toeplitz(x, SEs, Om, config[, mask])`.

NUFFT reference (needs `-DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON`, and `build/finufft/matlab` on the
path, or any other cufinufft MATLAB installation):

```matlab
plan = mr_nufft_plan(traj, config);             % once: plans + point sorting
AhAx = AhA_cufinufft(x, SEs, phi2, traj, config, plan);
```

The `.m` wrappers move host arrays to the GPU, cast to complex `single`, derive `n_coils` from
`size(SEs)`, and convert the 1-based mask to the 0-based `int64` mask the gateway expects. Call
`AhA_Toeplitz_mex` / `calc_toep_mex` directly only if you want to skip that.

> ⓘ MATLAB must have initialised its GPU before the first MEX call. Running `gpuDevice(1)`, or
> simply creating any `gpuArray`, is enough; the wrappers do this implicitly by moving their inputs.

On Apple silicon the same functions take ordinary MATLAB arrays (no Parallel Computing Toolbox);
`calc_toep`, the `'threshold'` mask, `mr_aha_op.create_op` and `AhA_cufinufft` run their NUFFTs
through MATLAB's Python interface. See [Apple silicon](../guide/apple-silicon.md).
