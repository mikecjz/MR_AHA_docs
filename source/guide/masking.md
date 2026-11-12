# Masking

`mask` holds linear indices into the *centred* upsampled grid `phi2_dim`, 1-based in MATLAB and
0-based in Python and C. Samples outside the mask contribute nothing. This is what makes a large 3D
operator fit in GPU memory when the upsampled k-space support is sparse.

`calc_mask` derives the mask from the trajectory:

1. Map every sample to its nearest point on the upsampled grid: `round(k·M/2π) + M/2` per axis,
   modulo `M`.
2. Keep every location within `mask_radius` cells of such a point (a $(2r+1)^d$ box). The default
   radius is 2.

This needs no NUFFT, and the same trajectory always gives the same mask on both backends. On the
densely sampled test problems radius 2 is as accurate as the unmasked operator:

* 280-spoke stack-of-stars: 79 % of the grid kept, operator error 3·10⁻⁶;
* 3D kooshball: 54 % kept, operator error 4·10⁻⁶.

It is less accurate on the 377-spoke 160² radial data (80 % kept, error 5·10⁻⁵) and on sparse
trajectories (64 spokes: 5·10⁻⁴; two-arm spiral: 3·10⁻²). There, raise `mask_radius`.

[Neighbourhood mask](../benchmarks/neighbour-mask.md) has the measurements for radii 1 to 8.
