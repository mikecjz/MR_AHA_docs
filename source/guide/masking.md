# Masking

`mask` holds linear indices into the *centred* upsampled grid `phi2_dim`, 1-based in MATLAB and
0-based in Python and C. Samples outside the mask contribute nothing. This is what makes a large 3D
operator fit in GPU memory when the upsampled k-space support is sparse.

`calc_mask` derives the mask from the trajectory, using one of two methods (`mask_method`).

**`'neighbour'` (default).**

1. Map every sample to its nearest point on the upsampled grid: `round(k·M/2π) + M/2` per axis,
   modulo `M`.
2. Keep every location within `mask_radius` cells of such a point (a $(2r+1)^d$ box). The default
   radius is 2.

This needs no NUFFT, and the same trajectory always gives the same mask. On the densely sampled
test problems radius 2 is as accurate as the unmasked operator:

* 280-spoke stack-of-stars: 79 % of the grid kept, operator error 3·10⁻⁶;
* 3D kooshball: 54 % kept, operator error 4·10⁻⁶.

It is less accurate on the 377-spoke 160² radial data (80 % kept, error 5·10⁻⁵) and on sparse
trajectories (64 spokes: 5·10⁻⁴; two-arm spiral: 3·10⁻²). There, raise `mask_radius` or use the
threshold method.

**`'threshold'`.**

1. Evaluate the Toeplitz kernel $T(\mathbf{k})$ of the trajectory with every weight set to one
   (exactly `calc_toep` with $L = 1$ and $\Phi\Phi^H = 1$).
2. Max-filter $|T|$ over the $3\times3$ ($3\times3\times3$ in 3D) neighbourhood of every location.
3. Keep the locations where the result exceeds `mask_threshold`.

The threshold is **absolute, in samples**: one isolated sample lying on the upsampled grid
contributes exactly 1 to $T$, so the default 0.05 drops locations that receive less than 5 % of one
sample's worth. A threshold relative to $\max|T|$ does not transfer between trajectories, because
the radial maximum sits at the k-space centre and equals the number of spokes (378, 280 and 15680
for the three test trajectories). Step 2 fills isolated near-zeros where the unit-weight Dirichlet
sum cancels inside the sampled region but the kernel of the real Gram blocks does not; it lowers the
masking error about tenfold at equal kept fraction.

This method follows the kernel's slowly decaying tails along the kx and ky axes, so it is more
accurate per kept location on sparse trajectories. At 0.05 it keeps 81–83 % of the grid in 2D and
stack, with an operator error of a few $10^{-6}$; it keeps 64 % for the kooshball.

[`benchmarks/neighbour_mask.md`](../benchmarks/neighbour-mask.md) compares the two methods, and
[`benchmarks/README.md`](../benchmarks/masking.md) documents the threshold study.
