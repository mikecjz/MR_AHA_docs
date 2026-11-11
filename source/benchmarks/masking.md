# Masking threshold experiment

`calc_mask` evaluates the Toeplitz kernel of the trajectory with all weights set to one, max-filters
it over each location's 3×3 (3×3×3) neighbourhood, and keeps locations above a threshold. The study
behind the design ([`mask_threshold.py`](https://github.com/mikecjz/MR_AHA/blob/b4fb00c/benchmarks/mask_threshold.py)) measured, for every threshold:

* the kept fraction;
* the error of one masked application against the NUFFT operator, for an image-like input and for
  white noise (the worst case);
* the difference between the 20-iteration CG solutions of the masked and NUFFT operators.

Results on the L40S, where every threshold fits, at the chosen default **0.05**:

| problem | kept | operator error, image | operator error, white noise | CG diff vs NUFFT | CG diff of unmasked Toeplitz |
|---|---|---|---|---|---|
| 2d | 83.1 % | 4.3·10⁻⁶ | 8.4·10⁻⁴ | 1.8·10⁻⁵ | 3.4·10⁻⁶ |
| stack | 81.2 % | 1.3·10⁻⁶ | 3.6·10⁻³ | 2.7·10⁻⁵ | 3.6·10⁻⁶ |
| 3d | 63.7 % | 3.9·10⁻⁶ | 7.3·10⁻⁴ | 1.7·10⁻⁴ | 1.5·10⁻⁴ |

How the design was arrived at:

1. **Plain thresholding of $|T|$ is not enough.** The unit-weight kernel is a sum of Dirichlet
   kernels, and it cancels to nearly zero at isolated points *inside* the sampled disc. The kernel
   of the real Gram blocks does not cancel there. Even a perfect disc mask ($|k| \le$ the sampled
   radius) leaves a 2·10⁻² white-noise error, because the exact kernel has slowly decaying tails.
   Max-filtering before thresholding fixed most of this: at the same kept fraction the image-input
   error fell about tenfold (2D, 80 % kept: 3.5·10⁻⁵ instead of 3.8·10⁻⁴). A 5×5 or 7×7 filter
   gained little more, so the 3×3 filter was kept.
2. **The threshold must be absolute, not relative to $\max|T|$.** For radial sampling the
   maximum sits at the k-space centre, where every spoke contributes. It is therefore the number of
   spokes (378, 280 and 15680 here) and has nothing to do with the level at the edge of k-space.
   With a relative threshold, 2D wanted about 10⁻⁴ and 3D about 10⁻⁵. Measured in samples (one
   isolated on-grid sample contributes exactly 1), the good operating points coincide at a few
   hundredths in all three problems.
3. **Choice.** Above 0.1 the CG difference grows quickly: τ = 1 gives 5·10⁻³ in 2D and 3·10⁻² for
   the stack. Below 0.03 the kept fraction rises steeply in 3D (91.7 % at 0.01) for no gain in 3D
   accuracy. 0.05 sits in between: image-input errors stay at a few 10⁻⁶ (at most 4× the unmasked
   operator's own error against the NUFFT), and CG solutions move by about 2·10⁻⁵.

**Why masking is not faster.** The masked execute path in `mr_aha.cu` gathers the retained
locations into a compact buffer, multiplies them, zeroes the *whole* padded grid with `cudaMemset`
and scatters the result back. That costs about as much as the dense in-place multiply it replaces,
and the two padded FFTs, which masking does not touch, dominate the operator anyway. A cheaper
masked path is possible; see the recommendations below.
