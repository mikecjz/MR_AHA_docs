# How the CUDA $A^H A$ operator was made faster

This note explains the mechanisms behind the 2026-11 performance work on the CUDA backend:
what each change does, why it saves time, and what was tried and rejected. The measurements are
in [CUDA operator optimisation and multi-GPU](../benchmarks/cuda-multigpu.md);
the user-facing description is in [Implementation notes](implementation.md) and
[Multi-GPU](../guide/multi-gpu.md). The work followed two plans, one for a single GPU
and one for several GPUs; a third one moved the pre-computation of `phi2_toep` onto all GPUs
(§3.4, measurements there).

| | before | after |
|---|---|---|
| stack-of-stars 320² × 56, L = 12, 12 coils, one H100 | 254.5 ms | 178.9 ms |
| the same, three H100s (NVLink) | — | 65.3 ms |
| 3D kooshball 160³, L = 12, 12 coils, one H100 | 362.6 ms | 254.6 ms |
| the same, three H100s | — | 88.2 ms |
| set-up of the 4-GPU operator (kernel computed and on every GPU), 3D, 4 L40S | 12.1 s | 4.3 s |
| the same, 2D radial 320², L = 72 | 3.7 s | 1.9 s |


## 1. Where the time went

The operator is, per coil $c$,

```
pad × S_c  →  forward FFT  →  block-diagonal multiply  →  inverse FFT  →  crop × conj(S_c)
```

and the per-stage GPU timer added for this work (`MR_AHA_CUDA_PROFILE=1`, CUDA events between the
stages, no extra synchronisation) showed on the TITAN RTX, stack problem:

| stage | before | share |
|---|---|---|
| pad × S_c | 66 ms | 5 % |
| forward FFT | 324 ms | 24 % |
| **block-diagonal** | **613 ms** | **45 %** |
| inverse FFT | 324 ms | 24 % |
| crop × conj(S_c) | 35 ms | 3 % |

The block-diagonal stage multiplies, at every upsampled k-space location, the length-$L$ row vector
of rank coefficients from the right by a Hermitian $L \times L$ block ($v \leftarrow v\,B$, the
MR-Multitasking convention) stored as its $(L^2+L)/2$ upper or lower triangle in `phi2_toep`. It does very little arithmetic per byte, so it is limited by memory
bandwidth, and the bytes are dominated by `phi2_toep` itself: 78 complex values per location at
$L = 12$, against 12 values of data read and 12 written. For the stack problem `phi2_toep` is
14.3 GB, and the old operator streamed it from DRAM for every one of the 12 coils.

The old kernel had three inefficiencies, each addressed below:

1. each off-diagonal stored element was read **twice** (once as $A_{ij}$ for row $i$, once as
   $\overline{A_{ji}}$ for row $j$);
2. the whole kernel was read **once per coil**;
3. because $L$ was a run-time value, each thread staged its vector in shared memory, which limited
   the block size and the occupancy for large $L$.


## 2. Single GPU

### 2.1 Read each stored element once

Instead of evaluating the matrix-vector product row by row, the kernel walks the stored triangle
once, in packed order. Packed component $t$ is the element $(r, c)$, and its single load contributes
to two outputs:

$$
y_r \mathrel{+}= A_{rc}\, v_c, \qquad
y_c \mathrel{+}= \overline{A_{rc}}\, v_r \quad (r \ne c).
$$

This halves the reads of `phi2_toep` for the off-diagonal part. The order in which each $y_i$
accumulates its terms ($j = 0, 1, \ldots, L-1$) is the same as in the row-by-row kernel, so the
result is **bitwise identical** (checked by `tests/check_bd_kernels.py` for $L = 1 \ldots 33$, both
triangles, masked and unmasked).

### 2.2 Specialise on $L$ so everything lives in registers

The kernels `mr_k_bd_dense_t<L, PAIR>` and `mr_k_bd_masked_t<L, PAIR>` (`src/cuda/mr_kernels_bd.cu`)
are templates on the rank, instantiated for $L = 1 \ldots 30$ and selected at plan creation through a
`switch` (`mr_bd_lookup`). With $L$ a compile-time constant, every loop is fully unrolled, every
array index is a constant, and the input and output vectors are plain registers: no shared memory,
no occupancy limit from it.

**Register pressure was the hard part.** Fully unrolled, the walk is a straight line of ~78 loads
(at $L = 12$), and the compiler likes to issue loads early to hide latency. Left alone, it
computed the addresses of, and loaded, many elements ahead of their use. Each such element held a
64-bit address and a complex value in registers, and the kernel spilled to local memory from
$L = 11$ on. Measured with `-Xptxas -v`:

| variant | L = 12, coil pair, sm_75 |
|---|---|
| `__ldg` (read-only cache) loads | 255 registers, spills |
| plain loads, `phi2_toep[t * stride]` | 216 registers, no spill (but spills again once a location-range parameter was added) |
| **running pointer `p += stride`, empty `asm volatile("" ::: "memory")` at each column, no `__restrict__`** | **206 registers, no spill** |

`__ldg` and `__restrict__` make it worse because they mark the loads as invariant, which allows
the compiler to move them anywhere. The empty `asm` with a memory clobber is a compiler barrier:
loads cannot move across it, so at most one column's loads (≤ $L$) are in flight at once. That is
still plenty of memory-level parallelism. On the TITAN RTX the spilling version cost 16 % in the
block-diagonal stage (282 vs 243 ms for the stack problem).

Above $L \approx 30$, a single coil's two vectors ($4L$ registers) plus addressing no longer fit in
255 registers and the specialised kernel spills whatever the code shape. Measured (TITAN RTX, 384² 2D,
8 coils), the specialised kernel is 7 % faster at $L = 28$, equal at $L = 30$, and 3 % slower at
$L = 32$. So $L > 30$ uses the generic kernel (`MR_BD_MAX_L = 30`).

### 2.3 Coil groups: read `phi2_toep` once per group, not once per coil

The plan now processes coils in groups (default 4, `MR_AHA_CUDA_COIL_GROUP`):

- the padded grids of a group sit side by side in one buffer `[Mx, My, Mz, L, group]`;
- **one batched cuFFT** (`cufftMakePlanMany64` over `group · L` volumes) transforms all of them;
- **one block-diagonal launch** serves the whole group. Each thread owns one k-space location
  and processes the group's coils two at a time (`PAIR`, for $L \le 16$). Every element of
  `phi2_toep` it loads is applied to both coils' vectors from registers, so a coil pair costs one
  read. The elements loaded for one pair are still in L2 when the same thread handles the next pair,
  so `phi2_toep` comes from DRAM about **once per group**.

For the stack problem on the TITAN RTX, this took the block-diagonal stage from 408 ms (specialised,
one coil at a time) to 243 ms (groups of 2). The kernel then moved about 570 GB/s, against 672 GB/s
peak.

**Memory decides the group size.** Each extra coil in a group costs one padded grid (2.2 GB for the
stack problem, 3.1 GB for 3D), and the Toeplitz kernel already occupies most of the GPU. The plan
queries the cuFFT work-area size without allocating (`cufftSetAutoAllocation(0)`), adds the grids
and index map, and shrinks the group until everything fits `mem_fraction` (default 0.5) of the free
memory at plan creation. A group of 1 (the old coil-by-coil operator) is always attempted. Both cuFFT
plans (full group and the smaller last group) share one work area. On the 80 GB H100 groups of 4 fit
for every benchmark problem. Groups of 6 gain only 1–2 % more, so the default stays at 4.

### 2.4 Masked operator: in place, through an index map

A masked `phi2_toep` keeps only selected locations. The old masked path took four steps:

1. gather the selected locations into a compact `[num_masked, L]` buffer and multiply;
2. `memset` the whole padded grid;
3. scatter the results back.

That cost an extra 1.8–2 GB buffer per coil and three full passes.

The new masked kernel works like the dense one, with one thread per grid location. It looks up its
position in the mask through an `int32` map `inv[Mx, My, Mz]`:

- `inv[j] = -1`: outside the mask. The thread writes zeros for every rank and coil, which is the
  "zero outside the mask" semantics, in the same pass.
- `inv[j] = m`: the location's block is column `m` of the `[num_masked, (L^2+L)/2]` kernel, so the
  thread multiplies in place.

The map is rebuilt at every execution (a memset plus one pass over the mask, about 0.2 ms) because
the caller may pass a different mask each time. It costs 4 bytes per grid location (92 MB for the
stack problem) instead of $8L$ bytes per masked location *per coil*. For a sorted mask, consecutive
threads read consecutive kernel columns, so the reads stay coalesced.

The effect: masked operators went from slightly *slower* than unmasked to 3–11 % *faster*, and
the masked 3D problem now fits on the 24 GB TITAN RTX.

### 2.5 Things that did not help

- **Threads per block 128 vs 256** for the specialised kernels: identical times. 256 is kept, with
  `__launch_bounds__(256)`.
- **Groups larger than 4:** 1–2 % at best, for 50 % more memory.
- **Specialising $L > 30$:** see 2.2.
- **Fusing pad/crop into cuFFT callbacks** and **CUDA graphs:** not attempted. After these changes
  the two FFTs are about 70 % of the GPU time on the H100, so further gains have to come from the
  FFTs themselves.


## 3. Several GPUs

### 3.1 Decomposition by coil

$A^H A x = \sum_c (\ldots)_c$, and each coil's pipeline is independent. The operator object
(`mr_op_create`, `src/cuda/mr_op.cu`; `create_op` in Python and MATLAB) gives each GPU a contiguous
range of whole coils and, with the Toeplitz method, an ordinary single-GPU plan for them, which keeps its own coil groups, specialised kernels and stream. At creation every
GPU receives a copy of `phi2_toep` (and the mask) and the sensitivity maps of its own coils. One
application then only moves `x` in and a partial result out:

```
caller's stream ── event ──┬─▶ GPU k: copy x ─ its coils ─ copy partial to the primary ─ event
                           └─▶ primary: its coils, straight into the output ─ event
caller's stream waits for all events ─▶ one kernel adds the staged partials into the output
```

A single host thread issues all of this asynchronously, with cross-device ordering through events
only. No host thread blocks, and the call returns immediately, like the single-GPU one. Every entry
point saves and restores the caller's current device, so MATLAB's "one GPU per session" model and
CuPy's current device are unaffected.

**How many GPUs.** The time is set by the busiest GPU, i.e. $\lceil C/G \rceil$ coils. The plan
uses the smallest $G' \le G$ with the same $\lceil C/G' \rceil$ (12 coils on 5 GPUs use 4), and
keeps problems below $2^{22}$ upsampled values per coil on one GPU (the 2D benchmark is
launch-bound at 0.5 ms).

**Scaling.** On three H100s the speed-up is ×2.76 (stack) to ×2.87 (3D). The rest is the copy of
`x` (≈ 1.5 ms over NVLink for the stack problem), the gather and the reduction.

### 3.2 Remainder split by rank

When $C$ is not a multiple of $G$, the last $r = C \bmod G$ coils would leave GPUs idle. Optionally
they are split across *all* GPUs by rank. Padding, FFTs and cropping are independent per rank, but
the block-diagonal stage needs all $L$ ranks of a location together, so the data is transposed
twice:

1. GPU $g$ pads and transforms its ranks $R_g$ (about $L/G$ of them);
2. **all-to-all:** GPU $g$ sends the k-space slab $K_h$ (a contiguous range of about
   $M_xM_yM_z/G$ locations) of its ranks to GPU $h$, which then holds all $L$ ranks of $K_h$;
3. GPU $h$ multiplies $K_h$ with its replicated kernel (the specialised kernels take a location
   range for this, `mr_bd_range`);
4. all-to-all back, inverse FFT and crop of $R_g$.

The copies are plain `cudaMemcpyPeerAsync` per rank. A single pitched `cudaMemcpy3DPeerAsync` would
do, but its source pitch (the whole grid) exceeds the 2 GB maximum pitch on large grids. Each GPU does
$1/G$ of the remaining coil's arithmetic, but moves its padded data twice, so the split pays only
over fast links.

**Cost model** (`MR_AHA_MGPU_SPLIT=auto`, the default). At plan creation the library measures the
GPU-to-GPU bandwidth (64 MB copies) and times the single-GPU operator on the primary over $n$ coils,
$T(n)$, *with its coil groups*. It then compares

$$
t_\text{whole} = T(\lceil C/G' \rceil), \qquad
t_\text{split} = T(\lfloor C/G \rfloor) + r\left(\frac{T(1)}{G} + \frac{2(G-1)}{G^2}\,
\frac{8 M_xM_yM_zL}{\text{bw}}\right),
$$

and splits only if $t_\text{split} < 0.95\, t_\text{whole}$. A first version timed one coil
without groups and scaled it linearly. That overstated $t_\text{whole}$, and it chose the split
where it lost. Measured on H100s, the split gains 4 % for 7 coils on 3 GPUs (3D) and loses 12 % for
8 coils on 3 GPUs, and the model picks correctly in both. On PCIe without peer access (~4 GB/s) it
never splits.

### 3.3 Robustness

- **Device resets under a live plan.** MATLAB resets a GPU on `reset(gpuDevice)` and whenever one is
  selected or deselected, which can destroy the CUDA context of the primary *and* of other GPUs of a
  plan. Using the plan's dead stream or event handles then crashes inside the driver. Freeing its old
  pointers could also free memory that a newer plan has since been given at the same addresses. Each
  device slot therefore records the unique ID of its context (`cuCtxGetId`). Release first checks that
  the primary context is active (`cuDevicePrimaryCtxGetState`, which does not create one) and that it
  is still the same instance, and skips the device otherwise. `mr_op_abandon` additionally
  skips the primary without probing it. The MATLAB gateway detects resets through a one-element
  sentinel gpuArray per plan.
- **Failure mid-issue:** if an execution fails after some work was issued, the call waits for the
  device streams before returning, so nothing keeps reading `x` or writing the output behind the
  caller's back.
- **Memory:** every GPU holds a full copy of the kernel, so multi-GPU does not raise the largest
  problem size. `MR_MGPU_SHARE_PRIMARY` (`share_primary` in Python/MATLAB) lets the primary GPU use
  the caller's arrays in place instead of a second copy.

### 3.4 Set-up: computing `phi2_toep` on all GPUs

Before this work, a multi-GPU operator was built from a kernel that `calc_toep` had computed on
the primary GPU. That has two costs, and both grow with the number of GPUs $G$:

1. **$T_\text{calc}$ on one GPU** while the other $G-1$ are idle.
2. **A broadcast.** The primary copies the whole kernel ($K$ bytes) to each other GPU in turn, so
   its own link carries $(G-1)K$. On 4 L40S over PCIe, for 2D radial with L = 72 ($K$ = 8.6 GB),
   set-up rises from 2.0 s on one GPU to 2.7 s on two and 3.7 s on four. That is four times as
   long as 10 CG iterations.

**Why the work splits cleanly.** The $(L^2+L)/2$ triangular components of the kernel are
independent: each one is a type-1 NUFFT, a centred inverse FFT and a scaling. Each fills one
contiguous slab of the output (`Mx*My*Mz` elements, or `num_masked` when masked). A range of
components is therefore one contiguous byte range. It can be computed on any GPU and moved with
a single `cudaMemcpyPeerAsync`. `calc_toep` was refactored into a per-device context
(`mr_toep_ctx`: packed Gram blocks, chunk buffers, cufinufft and cuFFT plans), which evaluates any
range of components into any destination (`src/cuda/mr_calc_toep.cu`).

**The engine** (`mr_toep_distribute`, `src/cuda/mr_toep_mgpu.cu`) runs one host thread per GPU:

1. The components are split into $G$ contiguous, nearly equal shares. Chunks are capped at a
   quarter of a share, so that transfers can start early.
2. Every GPU copies the small inputs from the primary (`phi2` and the trajectory, tens of MB). It
   plans, then computes and times the first chunk of its share (the *pilot*). The pilot is useful
   work whichever mode follows.
3. A cost model decides between two modes. Let $t_e$ be the slowest pilot time per component,
   $c$ the components each GPU still has to compute, $R$ the time a GPU needs to receive the
   components it does not compute, and $k$ the number of chunks:
   $$T_\text{exchange} \approx \max(c\,t_e, R) + R/k, \qquad T_\text{self} \approx (T - p)\,t_e .$$
   $R$ uses a bandwidth probe that runs on all GPUs at once, with the exchange's own schedule.
   Roughly, the exchange wins whenever a GPU can receive a component faster than it computes one:
   about 4 GB/s for the 2D L = 72 kernel.
4. **Exchange.** Each GPU computes the rest of its share directly into its own copy of the
   kernel. After each chunk it sends the chunk to the other GPUs, overlapped with the next
   chunk. Every component is computed once, so all copies are bit-identical.
   **Self.** Every GPU computes the components it lacks itself, with no transfers. This is meant
   for GPUs without peer access.

The engine serves the multi-GPU Toeplitz operator (`create_op(..., devices=...)`, which always
computes its own kernel from the Gram blocks): every GPU of the operator is a receiver of the whole
kernel; GPUs the operator does not use for coils only help compute. The `toep_dist` argument
(`'auto'`, `'split'` = exchange, `'replicate'` = self, `'broadcast'` = the old path *A* below)
selects the mode, and `MR_AHA_TOEP_SPLIT` (same values, `off` = `broadcast`) overrides it. The
measurements below were taken before the consolidation into one operator object, when the engine
also served `calc_toep(..., devices=...)` (*D* below, gathering on the primary only); that entry
point was removed because a kernel on one GPU computed by several is only useful to a single-GPU
operator, which the same GPUs could run themselves.

**Lesson: send one copy at a time.** The first version gave each GPU one stream per destination,
so all $G(G-1)$ transfers ran at once. On the L40S nodes the four GPUs sit on three different
NUMA nodes (`nvidia-smi topo`: every pair `SYS`). There, sustained all-to-all traffic collapsed
to 1–3 GB/s per GPU, although a short probe showed 11.5 GB/s:

| 2D L = 72 on 4 GPUs | exchange phase | whole set-up |
|---|---|---|
| all copies at once | 0.3–6.6 s | 7.8 s |
| one copy at a time, rotated | 0.45–0.54 s | 1.9 s |

With all copies at once, 3D took 14.5 s, slower than the broadcast. The current schedule gives
each GPU a single copy stream and sends to the receivers in rotated order $k+1, k+2, \ldots$
$(\bmod G)$. Because the GPUs advance in step, at any moment the destinations form a
permutation, so every GPU has about one outgoing and one incoming transfer. This is a ring
all-gather without the forwarding. The bandwidth probe uses the same schedule, so the cost
model sees the bandwidth the exchange will actually get.

**Measured** on 4 L40S (PCIe, `esplhpc-cp075`). Synthetic inputs have the shapes of the
benchmark problems, 12 coils each. Each figure is the median of 2 runs after a warm-up, in
seconds. *A* is the old path (`calc_toep` + the then `MultiGPUPlan`, kernel shared on
the primary; now `toep_dist='broadcast'`), *B* is every GPU computing the whole kernel
(`'replicate'`), *C* is split + exchange (`'split'`), and *D* is `calc_toep` on $G$ GPUs gathered on
the primary (removed since).

| set-up (s) | 2D radial 320², L = 72, 8.6 GB | stack 320² × 56, L = 12, 14.3 GB | 3D 160³, L = 12, 20.5 GB |
|---|---|---|---|
| `calc_toep`, 1 GPU | 2.03 | 2.54 | 7.90 |
| 2 GPUs: A → C | 2.68 → **1.48** | 3.89 → **2.19** | 9.25 → **5.33** |
| 4 GPUs: A | 3.69 | 5.59 | 12.14 |
| 4 GPUs: B (replicate) | 4.12 | 4.24 | 10.15 |
| 4 GPUs: C (split) | **1.87** | **2.49** | **4.26** |
| 4 GPUs: C, no overlap | 2.56 | 2.81 | 4.88 |
| 4 GPUs: `auto` (chose C) | 2.23 | 2.85 | 4.38 |
| 4 GPUs: D (kernel on the primary only) | 1.37 | 1.73 | 3.48 |
| **set-up + 10 CG iterations, 4 GPUs: A → C** | 4.73 → **2.77** | 8.94 → **5.81** | 15.88 → **8.01** |

- **Set-up no longer grows with $G$.** With C, building the 4-GPU operator takes less time than
  `calc_toep` alone on one GPU (2D 1.87 vs 2.03 s, 3D 4.26 vs 7.90 s), even though the kernel ends
  up on all four GPUs. For the 3D problem the set-up drops 2.8× and the end-to-end time 2×.
- **Overlap** of computation and exchange saves 0.3–0.7 s.
- **`auto` costs 0.1–0.35 s more than forcing C.** That is the bandwidth probe (≈ 0.1 s) plus
  run-to-run spread, and `auto` chose the exchange in every case. B is slower here than one GPU
  computing alone (2D 4.1 vs 2.0 s), so on PCIe nodes with peer access it only matters when the
  exchange cannot run at all.
- **Where the 4-GPU time goes** (2D, C):
  - setup and pilot, 0.95 s: per GPU, input copies, cufinufft and cuFFT planning, and a pilot
    of 165 components (0.09 s);
  - exchange, 0.45–0.54 s: 492 more components per GPU (0.26 s) overlapped with receiving
    6.5 GB at 14–16 GB/s;
  - releasing the buffers, 0.05 s;
  - building the operator's own state, ≈ 0.3 s.

  Planning is now the largest part, and the next thing to shrink.
- **Side finding:** a smaller chunk also helps a single GPU. With `MR_AHA_TOEP_MAX_CHUNK=128`,
  single-GPU `calc_toep` drops from 2.03 to 1.55 s for 2D L = 72 (smaller buffers and plans; the
  stack and 3D problems are unaffected). The default is unchanged for now.
- **NVLink** (H100) has not been measured yet. There the transfers are almost free (≈ 0.02 s
  for 6.5 GB at the measured 369 GB/s), so C should approach $T_\text{calc}/G$ plus planning.


### 3.5 The NUFFT method of the operator

`create_op(..., method='nufft')` (`src/cuda/mr_nufft_op.cu`) evaluates the same operator without a
kernel: per coil, the coil-weighted image goes through one batched type-2 cufinufft (all $L$ ranks,
$L N_z$ planes for a stack), the Gram multiplication, and one batched type-1 cufinufft back. It
mirrors the Python reference `AhANufftPlan` step by step, with two differences:

* **Samples reordered by readout.** cufinufft receives the trajectory with the samples of each
  readout contiguous (`kx + Nkx*t` instead of `t + Ntrajs*kx`). The rank vectors of a readout are
  then an `Nkx x L` matrix with leading dimension `Ntrajs*Nkx`, and the whole Gram multiplication
  is one `cublasCgemmStridedBatched` over the readouts ($K_t \leftarrow K_t \Phi_{2,t}$; for a
  stack, `cublasCgemmBatched` over readouts and kz encodes with pre-computed pointer arrays). No
  transpose of the data, any $L$. cufinufft bins the points anyway, so only the floating-point
  summation order changes.
* **No Python per-coil overhead**, and the same code serves MATLAB and several GPUs (coil split as
  for the Toeplitz method; each GPU holds its own plans, Gram blocks and trajectory).

**Measured** on one TITAN RTX (`pact400-recon3`), random inputs, per application of $A^H A$
(median of 5), against `AhANufftPlan` on the same inputs (relative difference ≤ 2.4·10⁻⁷):

| problem | `AhANufftPlan` | `create_op(..., method='nufft')` |
|---|---|---|
| 2D 256², L = 40, 8 coils, 400 × 512 samples | 173.5 ms | 146.2 ms |
| 2D 320², L = 72, 4 coils, 754 × 320 samples | 206.3 ms | 170.4 ms |
| stack 128² × 32, L = 12, 8 coils, 200 × 256 samples | 648.6 ms | 621.7 ms |
| 3D 64³, L = 8, 4 coils, 2000 × 128 samples | 187.8 ms | 183.2 ms |

The time is dominated by the two cufinufft transforms; set-up (plans + point sorting) takes a few
milliseconds. For comparison, the Toeplitz method applies the 2D 320², L = 72, 12-coil operator in
0.67 s on the same GPU after a 1.9 s set-up (kernel), and both single-GPU paths are unchanged by
the consolidation (old `calc_toep` + `AhAPlan` vs `create_op`: kernel 1.84–1.89 s vs 1.85–1.86 s,
$A^H A$ 663–666 ms vs 666–669 ms, same memory).

## 4. Verification

- `tests/check_bd_kernels.py`: specialised vs generic kernels, bitwise identical (132
  combinations: $L$ = 1–33, both triangles, masked/unmasked, coil groups 1/2/4).
- `tests/check_reference.py` / `tests/check_reference_matlab.m`: unchanged reference data still met.
- `tests/check_mgpu.py` and `tests/check_mgpu_matlab.m` (run on H100s only): multi-GPU vs single-GPU
  to 6·10⁻⁸ in 192 combinations each (2D/stack/3D, 5 and 8 coils, split on/off, masked/unmasked,
  both triangles, shared/copied constants). The MATLAB test also covers device hygiene, leaks,
  `clear mex`, `reset(gpuDevice)` and switching `gpuDevice`.
- 20-iteration CG on the stack data: the solution moves by 2.5·10⁻⁶, which is the run-to-run noise
  of the unchanged NUFFT operator (2.8·10⁻⁶).
The two multi-GPU tests below (and `tests/check_mgpu.py`) were replaced by `tests/check_op.py` and
`tests/check_op_matlab.m` when the operator objects were consolidated into `create_op`; the
records are kept as they were measured.

- `tests/check_toep_mgpu.py` (4 L40S): multi-GPU `phi2_toep` vs single-GPU in 1092 combinations
  (every reference problem: 2D/stack/3D, odd sizes, both triangles, unmasked and both masks,
  L = 1 with fewer components than GPUs; plus synthetic L = 12 and L = 72; 2, 3 and 4 GPUs; every
  `MR_AHA_TOEP_SPLIT` mode; both `calc_toep(devices=...)` and `from_gram`):
  - worst kernel difference 3.5·10⁻⁷ (cufinufft's run-to-run noise), 3.3·10⁻⁷ against the stored
    references, and 2.9·10⁻⁷ for the operator;
  - the copies on all GPUs bit-identical in split mode;
  - the current device unchanged and no memory leaked;
  - a GPU with its memory filled on purpose gives a clean error naming that GPU.
- `tests/check_mgpu_matlab.m` (4 L40S): the operator equivalence still passes in all 192
  combinations (worst 6.7·10⁻⁸). The new pre-computation section, `calc_toep(..., 'devices', ...)`
  and `mr_aha_op.from_gram` against the single-GPU kernel and operator, passes 24/24 (worst
  2.1·10⁻⁷).
- `tests/check_reference.py` and `tests/check_reference_matlab.m`: unchanged after the
  `calc_toep` refactoring (single GPU, TITAN RTX; Python also on an L40S). On the HPC, the MATLAB
  container crashes with "error while deallocating GPU memory" right after
  `check_mgpu_matlab`. The same crash already occurred before this work (job 6433021,
  2026-11-06); its cause has not been investigated.
