# Conventions

**Column-major.** Every array shape in this document and in the C header is column-major (MATLAB /
Fortran). Row-major callers (NumPy, CuPy) reverse the dimension order. The memory layout is
identical, so nothing ever has to be permuted.

**Centred arrays.** Image-space and k-space arrays have their origin in the *middle* of the array,
at index `floor(dim/2)`. The library performs the required shifts internally by index remapping, so
the convention is exact for even and odd matrix sizes alike.

**Single precision.** All complex data is `complexFloat` (`single` gpuArray, `complex64` CuPy).

## Array shapes

| Array | 2D | 3D and stack |
| ----- | -- | ------------ |
| `x` | `[Nx, Ny, L]` | `[Nx, Ny, Nz, L]` |
| `SEs` | `[Nx, Ny, Ncoils]` | `[Nx, Ny, Nz, Ncoils]` |
| `phi2_toep` (unmasked) | `[Mx, My, (L²+L)/2]` | `[Mx, My, Mz, (L²+L)/2]` |
| `phi2_toep` (masked) | `[num_masked, (L²+L)/2]` | `[num_masked, (L²+L)/2]` |
| `phi2` | `[Ntrajs, L, L]` | `[Ntrajs, L, L]`, or `[Ntrajs, Nkz, L, L]` for stack (centred kz order) |
| `traj` | `[Ntrajs, Nkx, 2]` | `[Ntrajs, Nkx, 3]`, or `[Ntrajs, Nkx, 2]` for stack |
| `AhAx` | same as `x` | same as `x` |

There is one sensitivity map per coil; it is shared by every rank.

`traj` holds coordinates in **radians per unit of the un-padded image grid**, i.e. the usual
$[-\pi, \pi)$ range.

## `config`

| Field | Meaning |
| ----- | ------- |
| `traj_type` | `'2D'`, `'3D'` or `'stack'` |
| `n_dim` | `[Nx Ny]` or `[Nx Ny Nz]` |
| `L` | number of ranks |
| `triangulation` | `'upper'` or `'lower'` — which half of each Hermitian block is stored |
| `phi2_dim` | upsampled Toeplitz size `[Mx My]` or `[Mx My Mz]`, typically `2*n_dim` |
| `is_masked` | whether `phi2_toep` is stored only at selected upsampled samples |

Optional: `n_coils`, `num_masked`, `eps`, `mem_fraction`, `gpu_device_id`, `verbose`,
`mask_radius`. The MATLAB
helper `mr_default_config` and the Python `mr_aha.make_config` fill in sensible defaults.

## Gram blocks (MR-Multitasking convention)

The library follows MR-Multitasking. The temporal basis $\Phi$ is **rank × time** (`Phi_rt`,
`[L, T]`), the spatial coefficients keep the rank as their **last** dimension, and the Gram block of
a readout (and kz encode) is

$$\Phi_2 = \sum_{t\,\in\,\text{readout}} \Phi(:,t)\,\Phi(:,t)^H, \qquad
\texttt{Phi2 = Phi\_rt(:,idx) * Phi\_rt(:,idx)'} .$$

Every block multiplication (by `phi2` in the NUFFT reference operators, by `phi2_toep` in the
Toeplitz operator) treats the $L$ rank coefficients of one k-space location as a **row vector
multiplied from the right**, $v \leftarrow v\,\Phi_2$, i.e. $v_l \leftarrow \sum_m v_m (\Phi_2)_{ml}$ —
exactly MR-Multitasking's `reshape(k, [], L) * Phi2`. Since MR-Multitasking forms $A^Hb$ as
`k.' * Phi_rt'`, this is its normal operator, and a MR-Multitasking `Phi2` (`[L, L, Ntrajs, Nz]`)
is passed unchanged apart from the dimension order: `permute(Phi2, [3 4 1 2])` in MATLAB, or the
row-major `P.transpose(2, 3, 0, 1)` of the h5py array in Python (plus an `fftshift` along kz if it is
in natural order).

## Triangular packing

Only $(L^2+L)/2$ of the $L^2$ entries of each block are stored; the rest follows from
$A_{ij} = \overline{A_{ji}}$. Both packings are the column-major packing of the corresponding MATLAB
triangle, i.e. exactly `A(tril(true(L)))` / `A(triu(true(L)))`.
