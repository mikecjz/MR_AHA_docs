# Implementation notes

**No scaling anywhere in the hot loop.** `calc_toep` evaluates the type-1 NUFFT with sign $-1$ and
then a *normalised* centred inverse FFT. Since the point-spread functions of the two signs are
spatial reflections of one another, $p^{-}[d] = p^{+}[-d]$, this equals
$\mathrm{DFT}(p^{+})/\prod M$ — exactly the multiplier needed between the two unnormalised cuFFTs of
`AhA_Toeplitz`. The $1/\prod M$ is therefore baked into `phi2_toep` and no scaling survives in the
iteration.

**Centring by index remap, not phase modulation.** The pad and crop kernels write and read at
$(x_d - \lfloor N_d/2 \rfloor) \bmod M_d$, and the block-diagonal kernel converts between natural
and centred order on the fly. This makes the pad/crop pair an exact adjoint pair (so the composed
operator is Hermitian whenever the multiplier is), removes two full array passes per FFT, and is
exact for odd matrix sizes, which a $(-1)^{x+y}$ modulation is not.

**Stack-of-stars.** `kz` is already Cartesian, so `Mz = Nz`: `calc_toep` runs a 2D type-1 per kz
encode with no transform and no shift along z, while `AhA_Toeplitz` still runs a full 3D FFT because
z participates in the image-domain transform. `phi2` therefore carries an extra `Nkz` axis.

**Memory.** `calc_toep` chunks its work over the $(L^2+L)/2$ triangular components and sizes the
chunk against `cudaMemGetInfo`, honouring `config.mem_fraction` (default: half the free memory); it
also bounds cufinufft's internal fine grid through `gpu_maxbatchsize`. An `mr_aha_plan` holds its
working buffers for the lifetime of the plan, so the iteration allocates nothing: the padded grids
of one coil group (`[Mx, My, Mz, L]` per coil, see below), the cuFFT work area and, when masked, an
`[Mx, My, Mz]` int32 index map.

**Block-diagonal stage (CUDA).** This stage, the per-location $L \times L$ Hermitian multiply, is
memory-bound on reading `phi2_toep`, the largest array of the operator. For $L \le 30$ it uses
kernels specialised on $L$: the rank vectors stay in registers, the stored triangle is walked once
so that each element is read once (not once per row), and the masked variant works in place on the
padded grid through a grid-to-mask index map instead of gathering into a compact buffer. Coils are
processed in **groups** (default 4): their padded grids sit side by side, one batched cuFFT
transforms them all, and one kernel pass reads `phi2_toep` once for the whole group instead of once
per coil. The group shrinks until its buffers fit `config.mem_fraction` of the free GPU memory
(a group of 1 is always tried). Environment variables, read when a plan is created:
`MR_AHA_CUDA_COIL_GROUP=n` (requested group size), `MR_AHA_CUDA_GENERIC_BD=1` (generic
shared-memory kernels, one coil at a time; always used for $L > 30$, where they are as fast); and at every application
`MR_AHA_CUDA_PROFILE=1` prints the GPU time of each stage. The results are bitwise identical to the
generic kernels (`tests/check_bd_kernels.py`).

**Re-entrancy.** No `__constant__` or other shared device state: kernel geometry travels by value.
Several plans, CUDA streams and host threads can be in flight at once, and `mr_last_error` is
thread-local. `config.stream` selects the stream; `config.gpu_device_id = -1` (the default for
MATLAB and Python) keeps whichever device the caller has selected.
