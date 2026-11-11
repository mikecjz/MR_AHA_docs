# Sparse trajectories: two-arm spiral

Radial trajectories cover most of the 2×-upsampled grid, so the mask can only drop 20–35 % of it.
To see what a sparse trajectory gains, [`spiral_mask.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/spiral_mask.py) repeats the experiment
with a **two-arm Archimedean spiral**: 2 arms × 1024 samples, one revolution, from
[`make_spiral.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/make_spiral.py) (mrinufft's `initialize_2D_spiral`). Everything was measured on
the TITAN RTX only.

The coefficient images, coil maps and Gram blocks are synthetic. The images are random ellipses or
ellipsoids, the coil maps are 12 smooth complex Gaussians, each readout (and each kz for the stack)
gets a random rank-3 PSD block, and L = 12. The CG right-hand side is $A^HA\,x_\text{true}$
computed with the NUFFT operator.

**Kept fraction and kernel size** (`phi2_toep` plus the `int64` mask, L = 12; the stack has
Nz = 56):

| N | kept at 0.05 | kept at 0.01 | 2D kernel: unmasked → 0.05 / 0.01 | stack kernel: unmasked → 0.05 / 0.01 |
|---|---|---|---|---|
| 128 | 39.6 % | 82.3 % | 39 → 15.6 / 32.5 MB | 2.13 → 0.86 / 1.78 GB |
| 192 | 18.6 % | 68.8 % | 88 → 16.5 / 61.2 MB | 4.80 → 0.90 / 3.35 GB |
| 256 | 10.6 % | 50.0 % | 156 → 16.8 / 79.0 MB | 8.53 → 0.92 / 4.32 GB |
| 320 | 8.0 % | 38.6 % | 244 → 19.7 / 95.3 MB | 13.3 → 1.08 / 5.21 GB |
| 384 | 6.4 % | 30.9 % | 351 → 22.7 / 110 MB | 19.2 → 1.24 / 6.00 GB |

The same 2048 samples cover a smaller share of a bigger grid, so the saving grows with N. At 0.05
the kernel is 9–15× smaller for N ≥ 256.

**Accuracy and measured memory.** Errors are against the NUFFT operator. "CG diff" compares the
20-iteration CG solutions of the masked Toeplitz and NUFFT operators. The NUFFT CG solution itself
is 0.94 (2D) and 0.81 (stack) away from `x_true`: two arms are extremely undersampled.

| problem | threshold | kept | kernel | operator error, image | operator error, white noise | CG diff | one $A^HA$ |
|---|---|---|---|---|---|---|---|
| 2D 256² | unmasked | 100 % | 156 MB | 5.2·10⁻⁷ | 4.8·10⁻⁷ | 2.2·10⁻⁵ | 13.2 ms |
| | 0.05 | 10.6 % | 16.8 MB | 1.0·10⁻³ | 9.8·10⁻³ | 1.6·10⁻² | **7.6 ms** |
| | 0.01 | 50.0 % | 79.0 MB | 1.0·10⁻⁴ | 2.5·10⁻³ | 7.8·10⁻⁴ | 11.0 ms |
| | 0.003 | 83.3 % | 132 MB | 4.6·10⁻⁶ | 3.3·10⁻⁴ | 6.6·10⁻⁵ | 13.7 ms |
| stack 320² × 56 | unmasked | 100 % | 13.3 GB | does not fit with the CG vectors (~19.5 GB) | | | |
| | 0.05 | 8.0 % | 1.08 GB | 1.1·10⁻³ | 1.2·10⁻² | 9.6·10⁻³ | **0.88 s** |
| | 0.01 | 38.6 % | 5.21 GB | 1.1·10⁻⁴ | 3.6·10⁻³ | 7.9·10⁻⁴ | 1.12 s |

For comparison, one NUFFT $A^HA$ takes 14.2 ms (2D) and 1.46 s (stack). With only 2048 samples
per plane, the NUFFT is cheap and the unmasked Toeplitz operator barely beats it in 2D.

Findings:

* **The kernel is no longer the main allocation once masked.** For the stack at 0.05, the device
  memory of the operator is about 1.1 GB of kernel plus 2.8 GB measured while creating the
  the operator (then `AhAPlan`) and applying it once. That 2.8 GB is mostly the padded `[Mx·My·Mz, L]` work buffer
  (2.1 GB) plus the output; masking does not shrink it. Unmasked, the same operator would need about
  13.3 + 2.6 ≈ 16 GB, so masking cuts the total about 4×.
* **`calc_toep` uses whatever memory it is given.** Its chunk size scales with `mem_fraction`
  (default 0.5 of free memory), so the stack peaked at about 10 GB. With `mem_fraction = 0.02` the
  peak drops to 1.6 GB at 0.05, which is the kernel plus about 0.5 GB, and it was no slower (1.8 s
  either way). Lower `mem_fraction` when the kernel must coexist with other data.
* **Here masking is also faster.** When only 6–11 % of the grid is kept, the gather → multiply →
  scatter path touches far less kernel data than the dense multiply. The 2D operator gets 1.7×
  faster (7.6 vs 13.2 ms). With radial data (80 % kept) the same path is not faster.
* **The default 0.05 is coarser for sparse sampling.** The operator error is about 10⁻³, against
  4·10⁻⁶ for radial data at the same threshold. In the image domain the unit-weight kernel is the
  trajectory's point-spread function; in k-space it is each sample's Dirichlet kernel, whose tails
  decay slowly. With two arms, the tails between the arms carry a much larger share of the
  kernel's energy, and the threshold removes exactly those tails.
  * Threshold 0.01 restores 10⁻⁴ operator error and a 7.8·10⁻⁴ CG difference while keeping
    the kernel 2–3× smaller.
  * 0.003 matches the radial accuracy but keeps 83 % of the grid.
  * Whether a 10⁻² CG difference matters depends on the application. Here it is two orders of
    magnitude below the reconstruction's own error against the truth.

An alternative mask, keeping only grid locations within r cells of a sample, is compared with
this one in [neighbour_mask.md](neighbour-mask.md).
