# Conventions

**Column-major.** Every array shape in this document and in the C header is column-major (MATLAB /
Fortran). Row-major callers (NumPy, CuPy) reverse the dimension order. The memory layout is
identical, so nothing ever has to be permuted.

**Centred arrays.** Image-space and k-space arrays have their origin in the *middle* of the array,
at index `floor(dim/2)`. The library performs the required shifts internally by index remapping, so
the convention is exact for even and odd matrix sizes alike.

**Single precision.** All complex data is `complexFloat` (`single` gpuArray, `complex64` CuPy).

## Array shapes

Dimension symbols used below:

* `Nx`, `Ny`, `Nz` — image matrix size (`config.n_dim`). `Nz` is the number of slices / kz
  partitions for stack and 3D; 2D has no `Nz`.
* `L` — number of subspace ranks (temporal basis functions, `config.L`).
* `Ncoils` — number of receive coils.
* `Mx`, `My`, `Mz` — size of the upsampled (oversampled) Toeplitz grid (`config.phi2_dim`),
  typically `2*Nx`, `2*Ny`, `2*Nz`. For stack, `Mz = Nz`: kz is already Cartesian and is not
  upsampled.
* `num_masked` — number of upsampled-grid locations kept by the mask (see [Masking](masking.md)).
* `Ntrajs` — number of readouts (spokes, spiral interleaves, …).
* `Nkx` — number of k-space samples along one readout.
* `Nkz` — number of kz encodes of a stack acquisition; always equal to `Nz`.

The arrays:

* **`x`** — the subspace image: one complex image per rank, with the rank as the last dimension.
  * 2D: `[Nx, Ny, L]`
  * 3D and stack: `[Nx, Ny, Nz, L]`
* **`SEs`** — coil sensitivity maps, one per coil, shared by every rank.
  * 2D: `[Nx, Ny, Ncoils]`
  * 3D and stack: `[Nx, Ny, Nz, Ncoils]`
* **`phi2_toep`** (unmasked) — the Toeplitz kernel: one packed Hermitian $L \times L$ block
  ([triangular packing](#triangular-packing)) at every point of the upsampled grid.
  * 2D: `[Mx, My, (L²+L)/2]`
  * 3D and stack: `[Mx, My, Mz, (L²+L)/2]`
* **`phi2_toep`** (masked) — the same kernel stored only at the `num_masked` grid locations
  selected by the mask.
  * 2D, 3D and stack: `[num_masked, (L²+L)/2]`
* **`phi2`** — the [Gram block](#gram-blocks) $\Phi_2$ of each readout: a full $L \times L$
  matrix per readout (and per kz encode for stack).
  * 2D and 3D: `[Ntrajs, L, L]`
  * stack: `[Ntrajs, Nkz, L, L]`, kz in centred order
* **`traj`** — non-Cartesian k-space coordinates of every sample, the last dimension holding the
  (kx, ky[, kz]) components. Stack holds only the in-plane coordinates; kz is Cartesian.
  * 2D and stack: `[Ntrajs, Nkx, 2]`
  * 3D: `[Ntrajs, Nkx, 3]`
* **`AhAx`** — the output $A^H A x$; same shape as `x`.

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

## Gram blocks

The temporal basis $\Phi$ is **rank × time** (`Phi_rt`,
`[L, T]`), the spatial coefficients keep the rank as their **last** dimension, and the Gram block of
a readout (and kz encode) is

$$\Phi_2 = \sum_{t\,\in\,\text{readout}} \Phi(:,t)\,\Phi(:,t)^H, \qquad
\texttt{Phi2 = Phi\_rt(:,idx) * Phi\_rt(:,idx)'} .$$

Every block multiplication (by `phi2` in the NUFFT reference operators, by `phi2_toep` in the
Toeplitz operator) treats the $L$ rank coefficients of one k-space location as a **row vector
multiplied from the right**, $v \leftarrow v\,\Phi_2$, i.e. $v_l \leftarrow \sum_m v_m (\Phi_2)_{ml}$ —
in MATLAB, `reshape(k, [], L) * Phi2`.

## Triangular packing

Only $(L^2+L)/2$ of the $L^2$ entries of each block are stored; the rest follows from
$A_{ij} = \overline{A_{ji}}$. Both packings are the column-major packing of the corresponding MATLAB
triangle, i.e. exactly `A(tril(true(L)))` / `A(triu(true(L)))`.
