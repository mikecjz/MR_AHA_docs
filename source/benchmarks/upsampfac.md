# Experimental: NUFFT upsampling factor in `calc_toep`

`calc_toep` runs its type-1 transforms at cufinufft's standard upsampling factor σ = 2. Two
experimental environment variables change that: `MR_AHA_TOEP_UPSAMPFAC` (σ, default 2.0) and
`MR_AHA_TOEP_GPU_METHOD` (type-1 spreading method, default 0 = cufinufft's choice).

> **Outcome.** The defaults (σ = 2, cufinufft's choice of method) are the recommended settings on
> both backends. One combination is faster on CUDA: for the 3D kooshball, σ = 1.25 at
> `eps = 1e-4` computes the kernel 1.7× faster than σ = 2 at `eps = 1e-5`, at a similar error.

## CUDA

[`toep_upsampfac.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/toep_upsampfac.py)
sweeps σ ∈ {2, 1.25}, `eps` ∈ {1e-6, 1e-5, 1e-4} and method ∈ {auto, 1 = GM-sort} on one TITAN
RTX (`pact400-recon3`).

* **Time** is the kernel computation inside `create_op` (minimum of two).
* **Error** is one $A^HA$ application against the NUFFT operator (σ = 2, `eps` = 1e-6).
* The 3D problem uses the default mask (`calc_mask`, radius 2); 2D and stack are unmasked.
* Raw numbers: `results/pact400-recon3/toep_upsampfac_<problem>.json`.

Each cell gives time / error.

| problem | eps | method | σ = 2 (default) | σ = 1.25 |
|---|---|---|---|---|
| 2D | 1e-6 | auto | 18 ms / 9.3e-7 | 21 ms / 1.7e-6 |
| | 1e-6 | GM-sort | 11 ms / 1.1e-6 | 14 ms / 1.9e-6 |
| | 1e-4 | GM-sort | 7 ms / 1.6e-5 | 7 ms / 5.2e-6 |
| stack | 1e-6 | auto | 2.95 s / 1.2e-6 | 3.45 s / 2.2e-6 |
| | 1e-6 | GM-sort | 2.28 s / 1.1e-6 | 2.55 s / 2.2e-6 |
| | 1e-5 | GM-sort | 2.11 s / 1.4e-6 | 2.27 s / 2.1e-6 |
| | 1e-4 | GM-sort | 1.96 s / 2.1e-5 | 1.75 s / 4.7e-6 |
| 3D (masked) | 1e-6 | auto | 18.0 s / 1.9e-6 | 32.7 s / 2.3e-6 |
| | 1e-5 | auto | 12.6 s / 3.3e-6 | 19.4 s / 2.9e-6 |
| | 1e-4 | auto | 8.0 s / 1.9e-5 | 7.2 s / 4.8e-6 |
| | 1e-6 | GM-sort | 7.0 s / 1.5e-5 | 10.2 s / 8.5e-5 |
| | 1e-5 | GM-sort | 5.8 s / 6.0e-6 | 7.9 s / 6.1e-5 |
| | 1e-4 | GM-sort | 5.0 s / 2.1e-5 | 4.6 s / 6.9e-5 |

* **At the same `eps`, σ = 1.25 is slower (up to 1.8×).** The Toeplitz type-1 transform has
  2N modes per axis, so the samples are dense on the fine grid and spreading, not the FFT,
  dominates. σ = 1.25 needs a kernel about 1.4× wider for the same `eps`, i.e. about 2× (2D) to
  3× (3D) more spreading work, which outweighs the 0.39× (2D) / 0.24× (3D) smaller FFT.
* **At matched accuracy σ = 1.25 can win.** Its kernel width formula is more conservative, so at
  `eps = 1e-4` it reaches about 5·10⁻⁶. For 3D (auto method) that is 7.2 s against 12.6 s for
  σ = 2 at `eps = 1e-5` (3.3·10⁻⁶): 1.7× faster. For stack the gain is 1.2× at a 3× larger error;
  2D gains nothing.
* **GM-sort (method 1) trades accuracy in 3D.** It is 1.2–1.6× faster than cufinufft's choice
  (shared memory) in 2D and stack at equal accuracy, and 2.5× faster in 3D, but there its error
  grows 5–30× (to 10⁻⁵–10⁻⁴), probably from fp32 global atomics in dense cells.
* **Default unchanged.** σ = 2 and cufinufft's method choice stay the defaults; the variables are
  experimental and not part of the API.

## Apple silicon

Measured on an M3 Ultra (`AW650523HJY`). On Metal the kernel's type-1 NUFFT is mlx-nufft
(`mr_aha._psf`), which reads the same `MR_AHA_TOEP_UPSAMPFAC`. There is no spreading-method
switch. The problems and the protocol are the same: time is `calc_toep` in Python (minimum of
two), the error is against the mlx-nufft NUFFT operator at σ = 2 and `eps` = 1e-6, and 3D is
masked. Raw numbers: `results/AW650523HJY/toep_upsampfac_<problem>.json`.

| problem | eps | σ = 2 (default) | σ = 1.25 |
|---|---|---|---|
| 2D | 1e-6 | 10.1 ms / 9.9e-7 | 11.6 ms / 4.2e-6 |
| | 1e-5 | 9.8 ms / 2.8e-6 | 8.9 ms / 2.0e-6 |
| | 1e-4 | 7.1 ms / 1.6e-5 | 7.9 ms / 6.2e-6 |
| stack | 1e-6 | 3.50 s / 8.9e-7 | 4.73 s / 4.1e-6 |
| | 1e-5 | 3.33 s / 2.0e-6 | 4.47 s / 2.0e-6 |
| | 1e-4 | 3.31 s / 2.0e-5 | 3.56 s / 5.8e-6 |
| 3D (masked) | 1e-6 | 14.1 s / 1.5e-5 | 19.2 s / 2.9e-5 |
| | 1e-5 | 10.5 s / 2.1e-5 | 19.2 s / 2.9e-5 |
| | 1e-4 | 8.0 s / 2.1e-5 | 14.6 s / 3.7e-5 |

* **σ = 1.25 does not pay off on Metal.** It is 1.1–1.8× slower for stack and 3D at every `eps`,
  and also at matched accuracy (stack: 3.56 s at `eps = 1e-4` against 3.33 s for σ = 2 at
  `eps = 1e-5`); 2D is too small to matter. The CUDA gain at matched accuracy does not carry over:
  σ = 2 already gets faster as `eps` loosens (3D: 14.1 → 8.0 s), while σ = 1.25 in 3D runs at the
  width cap of 8 that mlx-nufft applies in fp32, so it does not get cheaper.
* **Accuracy.** σ = 1.25 raises the operator error 2–5× at `eps = 1e-6`. The 2D
  kernel itself differs from the CPU-finufft kernel by 2.8·10⁻⁴ at σ = 1.25 (8.9·10⁻⁷ at σ = 2;
  CPU finufft at σ = 1.25: 8.5·10⁻⁷), so mlx-nufft's fp32 σ = 1.25 path is the limit, not the
  switch. The 3D errors (1.5–2·10⁻⁵ even at σ = 2) are measured against mlx-nufft's own fp32 3D
  NUFFT operator, which sits near that floor.
