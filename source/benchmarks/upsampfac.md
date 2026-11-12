# Experimental: NUFFT upsampling factor in `calc_toep`

`calc_toep` runs its type-1 transforms at cufinufft's standard upsampling factor σ = 2. Two
experimental environment variables change that:

* `MR_AHA_TOEP_UPSAMPFAC` sets σ (default 2.0; the other supported value is 1.25);
* `MR_AHA_TOEP_GPU_METHOD` (CUDA only) sets the type-1 spreading method (default 0 = cufinufft's
  choice, 1 = GM-sort).

σ only sizes the NUFFT's fine grid; the Toeplitz grid `phi2_dim` is unchanged.

## Outcome

The defaults (σ = 2, cufinufft's choice of method) are the recommended settings on both backends.
Three settings compute the kernel faster on CUDA:

* **A looser `eps` in 3D.** At `eps = 1e-5` the 3D kernel takes 12.6 s instead of 18.0 s, with an
  operator error of 3.3·10⁻⁶.
* **σ = 1.25 with `eps = 1e-4` in 3D.** The kernel takes 7.2 s, 1.7× faster than σ = 2 at
  `eps = 1e-5`, at a similar error (4.8·10⁻⁶). The kernel width that cufinufft picks for
  σ = 1.25 is conservative, so the result is more accurate than the requested `eps`.
* **GM-sort in 2D and stack.** It is 1.6× (2D) and 1.3× (stack) faster than cufinufft's choice at
  the same accuracy.

## Measurements

| problem | σ | `eps` | method | time | error |
|---|---|---|---|---|---|
| 2D | 2 | 1e-6 | auto (default) | 18 ms | 9.3e-7 |
| 2D | 2 | 1e-6 | GM-sort | 11 ms | 1.1e-6 |
| stack | 2 | 1e-6 | auto (default) | 2.95 s | 1.2e-6 |
| stack | 2 | 1e-6 | GM-sort | 2.28 s | 1.1e-6 |
| 3D (masked) | 2 | 1e-6 | auto (default) | 18.0 s | 1.9e-6 |
| 3D (masked) | 2 | 1e-5 | auto | 12.6 s | 3.3e-6 |
| 3D (masked) | 1.25 | 1e-4 | auto | 7.2 s | 4.8e-6 |

* **Machine.** One TITAN RTX (`pact400-recon3`).
* **Time** is the kernel computation inside `create_op` (minimum of two).
* **Error** is one $A^HA$ application against the NUFFT operator (σ = 2, `eps` = 1e-6).
* **Problems.** Those of the [main study](results.md). The 3D problem uses the default mask
  (`calc_mask`, radius 2); 2D and stack are unmasked.
* **Script and raw numbers.**
  [`toep_upsampfac.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/toep_upsampfac.py);
  `results/pact400-recon3/toep_upsampfac_<problem>.json`.

The variables are experimental and not part of the API.
