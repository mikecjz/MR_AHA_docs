# Implementation notes

This page summarises the design choices behind the operator. Each paragraph names one choice and
what it buys. [How the CUDA operator gets its speed](performance.md) explains the performance
ones in depth.

## Numerics

**No scaling in the hot loop.** The two cuFFTs of `AhA_Toeplitz` are unnormalised, so the
multiplier between them must carry a factor $1/\prod M$. `calc_toep` puts it into `phi2_toep`: it
evaluates the type-1 NUFFT with sign $-1$ and then a *normalised* centred inverse FFT. The
point-spread functions of the two signs are spatial reflections of one another,
$p^{-}[d] = p^{+}[-d]$, so the result equals $\mathrm{DFT}(p^{+})/\prod M$, which is exactly the
multiplier needed. The iteration itself never scales anything.

**Centring by index remap.** Image-space and k-space arrays have their origin in the middle. The
library realises this through the indices it reads and writes:

- the pad and crop kernels write and read at $(x_d - \lfloor N_d/2 \rfloor) \bmod M_d$;
- the block-diagonal kernel converts between natural and centred order on the fly.

This costs no extra pass over the data and is exact for even and odd matrix sizes. Pad and crop
are an exact adjoint pair, so the composed operator is Hermitian whenever the multiplier is.

**Hermitian by construction.** Only one triangle of each $L \times L$ block is computed and
stored. The block-diagonal kernel produces the mirrored half by conjugation. See
[Why Toeplitz?](toeplitz.md#hermitian-by-construction).

**Stack-of-stars.** `kz` is already Cartesian, so `Mz = Nz`. `calc_toep` runs a 2D type-1 NUFFT per
kz encode, with no transform and no shift along z, which is why `phi2` carries an extra `Nkz`
axis. `AhA_Toeplitz` runs a full 3D FFT, because z takes part in the image-domain transform.

## Memory

**`calc_toep`** computes the $(L^2+L)/2$ triangular components in chunks. It sizes the chunk
against `cudaMemGetInfo` and `config.mem_fraction` (default: half the free memory), and bounds
cufinufft's internal fine grid through `gpu_maxbatchsize`.

**The operator** (an `mr_aha_plan` inside) holds its working buffers for its whole lifetime, so
the iteration allocates nothing:

- the padded grids of one coil group, `[Mx, My, Mz, L]` per coil;
- the cuFFT work area;
- when masked, an `[Mx, My, Mz]` `int32` index map.

## Block-diagonal stage (CUDA)

This stage is the per-location multiplication by the Hermitian $L \times L$ block. It is bound by
memory bandwidth, because it reads `phi2_toep`, the largest array of the operator. Four choices
keep those reads to a minimum:

- **Kernels specialised on $L$** (for $L \le 30$). The rank vectors stay in registers, and the
  stored triangle is walked once, so each stored element is read once.
- **Coil groups** (default 4). The padded grids of a group sit side by side, one batched cuFFT
  transforms them all, and one kernel pass reads `phi2_toep` once for the whole group. The group
  shrinks, down to one coil, until its buffers fit `config.mem_fraction` of the free GPU memory.
- **In-place masked kernel.** The masked variant multiplies in place on the padded grid, through
  a grid-to-mask index map.
- **Tiled kernels** (for $30 < L \le 512$). The rank vectors no longer fit in registers, so a
  thread block stages them in shared memory for a tile of 16 grid locations and up to 4 coils.
  It copies the stored triangle from DRAM panel by panel, once for all of them. On `2d72`
  ($L = 72$) this is 3 to 9 times faster than the generic kernels
  ([details](performance.md#25-ranks-above-30-tiled-kernels)).

The generic shared-memory kernels, one coil at a time, remain as the fallback for a mask of more
than 2³¹ locations. All three kernel families agree to single-precision rounding
(`tests/check_bd_kernels.py`).

Environment variables:

| Variable | Read | Effect |
|---|---|---|
| `MR_AHA_CUDA_COIL_GROUP=n` | when a plan is created | requested coil group size |
| `MR_AHA_CUDA_GENERIC_BD=1` | when a plan is created | use the generic kernels for every $L$ (testing and A/B timing) |
| `MR_AHA_CUDA_TILED_BD=1` | when a plan is created | use the tiled kernels for every $L$ (testing and A/B timing) |
| `MR_AHA_CUDA_BDT_NC`, `MR_AHA_CUDA_BDT_P`, `MR_AHA_CUDA_BDT_BLOCKS` | at every application | tiled kernels: coils per batch (1, 2, 4), locations per tile (8, 16, 32), thread blocks per SM (1, 2); for tuning |
| `MR_AHA_CUDA_PROFILE=1` | at every application | print the GPU time of each stage |

## Re-entrancy

Kernel geometry travels by value, and the library keeps no shared device state. Several plans,
CUDA streams and host threads can therefore be in flight at once, and `mr_last_error` is
thread-local. `config.stream` selects the stream. `config.gpu_device_id = -1` (the default for
MATLAB and Python) keeps whichever device the caller has selected.
