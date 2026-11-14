# How the CUDA $A^H A$ operator gets its speed

This note explains the design choices that make the CUDA backend fast: what each one does and why
it saves time. It covers one application of the operator on one GPU (§2, with the kernels for
ranks above 30 in §2.5), multi-GPU (§3.1 to
§3.3), the multi-GPU set-up of the Toeplitz kernel (§3.4) and the NUFFT method (§3.5).

* For a short summary, read [Implementation notes](implementation.md).
* For how to use multi-GPU, read [Multi-GPU](../guide/multi-gpu.md).
* For the full measurements, read
  [CUDA operator optimisation and multi-GPU](../benchmarks/cuda-multigpu.md).

The gains at a glance, for 12 coils and $L = 12$:

| one $A^H A$ | generic kernels, one coil at a time, one H100 | default, one H100 | default, three H100s (NVLink) |
|---|---|---|---|
| stack-of-stars 320² × 56 | 254.5 ms | 178.9 ms | 65.3 ms |
| 3D kooshball 160³ | 362.6 ms | 254.6 ms | 88.2 ms |

| set-up of a 4-GPU operator (4 L40S, PCIe) | kernel computed on one GPU, then copied | kernel computed on all four GPUs (default) |
|---|---|---|
| 3D kooshball 160³, $L = 12$ | 12.1 s | 4.3 s |
| 2D radial 320², $L = 72$ | 3.7 s | 1.9 s |

| one $A^H A$, 2D radial 320², $L = 72$, 12 coils | generic kernels | default (tiled kernels, §2.5) |
|---|---|---|
| one H100 | 335 ms | 36 ms |
| one L40S | 309 ms | 90 ms |
| one TITAN RTX | 663 ms | 224 ms |

The reference columns can be reproduced with the current library:
`MR_AHA_CUDA_GENERIC_BD=1` selects the generic kernels, and `toep_dist='broadcast'` computes the
kernel on one GPU.


## 1. Where the time goes

The operator is, per coil $c$,

```
pad × S_c  →  forward FFT  →  block-diagonal multiply  →  inverse FFT  →  crop × conj(S_c)
```

`MR_AHA_CUDA_PROFILE=1` prints the GPU time of each stage (CUDA events between the stages, no
extra synchronisation). With the generic kernels, on the TITAN RTX and the stack problem:

| stage | time | share |
|---|---|---|
| pad × S_c | 66 ms | 5 % |
| forward FFT | 324 ms | 24 % |
| **block-diagonal** | **613 ms** | **45 %** |
| inverse FFT | 324 ms | 24 % |
| crop × conj(S_c) | 35 ms | 3 % |

The block-diagonal stage is the one to design for. At every upsampled k-space location it
multiplies the length-$L$ row vector of rank coefficients from the right by a Hermitian
$L \times L$ block ($v \leftarrow v\,B$; see [Gram blocks](../guide/conventions.md#gram-blocks)). The block is stored in
`phi2_toep` as its $(L^2+L)/2$ upper or lower triangle.

The stage does very little arithmetic per byte, so it is limited by memory bandwidth. The bytes
are dominated by `phi2_toep`: at $L = 12$ it holds 78 complex values per location, against 12
values of data read and 12 written. For the stack problem `phi2_toep` is 14.3 GB. How often it is
read decides the time of the stage.

The default kernels follow three rules, explained in §2:

1. each stored element of `phi2_toep` is read **once** per pass over the triangle (§2.1);
2. the rank vectors live in **registers**, because $L$ is a compile-time constant (§2.2);
3. `phi2_toep` is read from DRAM about **once per group of coils** (§2.3).

With them the block-diagonal stage takes 20–25 % of the GPU time and runs close to the DRAM
bandwidth. The two FFTs are then about 70 % of the time on the H100.


## 2. One GPU

### 2.1 Each stored element is read once

The kernel walks the stored triangle once, in packed order. Packed component $t$ is the element
$(r, c)$ of the block, and its single load contributes to two outputs of $y = v\,A$:

$$
y_c \mathrel{+}= v_r\, A_{rc}, \qquad
y_r \mathrel{+}= v_c\, \overline{A_{rc}} \quad (r \ne c).
$$

The second term uses the Hermitian symmetry $A_{cr} = \overline{A_{rc}}$, so the mirrored half of
the block never has to be loaded. The results agree with the generic kernels to single-precision
rounding (`tests/check_bd_kernels.py`: both triangles, masked and unmasked, coil groups of 1, 2
and 4).

### 2.2 Kernels specialised on $L$: everything in registers

The kernels `mr_k_bd_dense_t<L, PAIR>` and `mr_k_bd_masked_t<L, PAIR>` (`src/cuda/mr_kernels_bd.cu`)
are templates on the rank. They are instantiated for $L = 1 \ldots 30$ and selected at plan
creation through a `switch` (`mr_bd_lookup`). With $L$ a compile-time constant:

- every loop is fully unrolled;
- every array index is a constant;
- the input and output vectors are plain registers, so the kernels need no shared memory and
  nothing limits the block size or the occupancy.

The kernels run with 256 threads per block (`__launch_bounds__(256)`).

**Staying within the register file.** Fully unrolled, the walk is a straight line of 78 loads at
$L = 12$. The kernel reads `phi2_toep` with plain loads through a running pointer
(`p += stride`), and places an empty `asm volatile("" ::: "memory")` after each column of the
triangle. The empty statement is a compiler barrier: loads cannot move across it, so at most one
column's loads (≤ $L$) are in flight at once. That is still plenty of memory-level parallelism.
At $L = 12$ with a coil pair the kernel uses 206 of the 255 registers (`-Xptxas -v`, sm_75) and
nothing spills to local memory. Keeping the kernel out of local memory is worth 16 % of the
block-diagonal stage (TITAN RTX, stack problem).

**Range of $L$.** Up to $L = 30$ (`MR_BD_MAX_L`), one coil's two vectors ($4L$ registers) plus
addressing fit in the register file. Larger ranks use the tiled kernels of §2.5, which keep the
vectors in shared memory instead.

### 2.3 Coil groups: `phi2_toep` is read once per group

The plan processes coils in groups (default 4, `MR_AHA_CUDA_COIL_GROUP`):

- the padded grids of a group sit side by side in one buffer `[Mx, My, Mz, L, group]`;
- **one batched cuFFT** (`cufftMakePlanMany64` over `group · L` volumes) transforms all of them;
- **one block-diagonal launch** serves the whole group.

In that launch each thread owns one k-space location and processes the group's coils two at a
time (`PAIR`, for $L \le 16$). Every element of `phi2_toep` it loads is applied to both coils'
vectors from registers, so a coil pair costs one read. The elements loaded for one pair are still
in the L2 cache when the same thread handles the next pair. `phi2_toep` therefore comes from DRAM
about **once per group**.

For the stack problem on the TITAN RTX, groups of 2 take the block-diagonal stage from 408 ms
(specialised kernels, one coil at a time) to 243 ms. The kernel then moves about 570 GB/s; the
card's peak is 672 GB/s.

**Memory decides the group size.** Each coil in a group needs one padded grid (2.2 GB for the
stack problem, 3.1 GB for 3D), and the Toeplitz kernel already occupies most of the GPU. At plan
creation the library:

1. queries the size of the cuFFT work area without allocating it (`cufftSetAutoAllocation(0)`);
2. adds the padded grids and the index map of the masked operator (§2.4);
3. shrinks the group, down to one coil, until everything fits `mem_fraction` (default 0.5) of the
   free memory.

Both cuFFT plans (the full group and the smaller last group) share one work area. The default of 4
fits every benchmark problem on the 80 GB H100.

### 2.4 Masked operator: in place, through an index map

A masked `phi2_toep` keeps only selected locations. The masked kernel works like the dense one,
with one thread per grid location. Each thread looks up its location in an `int32` map
`inv[Mx, My, Mz]`:

- `inv[j] = m`: the location's block is column `m` of the `[num_masked, (L^2+L)/2]` kernel, and
  the thread multiplies in place.
- `inv[j] = -1`: the location is outside the mask. The thread writes zeros for every rank and
  coil, which is the "zero outside the mask" semantics, in the same pass.

The map is rebuilt at every execution, because the caller may pass a different mask each time.
That takes a memset and one pass over the mask, about 0.2 ms. The map costs 4 bytes per grid
location (92 MB for the stack problem), and it is the only extra buffer the masked operator needs.
For a sorted mask, consecutive threads read consecutive kernel columns, so the reads stay
coalesced.

The effect: masked operators are 3–11 % faster than unmasked ones, and the masked 3D problem fits
on the 24 GB TITAN RTX.

### 2.5 Ranks above 30: tiled kernels

Above $L = 30$ the rank vectors of one location no longer fit in registers, so the specialised
kernels of §2.2 stop there. Ranks from 31 to 512 (`MR_BDT_MAX_L`) use the tiled kernels
`mr_k_bdt<NC, OPT, MASKED>` (`src/cuda/mr_kernels_bdt.cu`). They follow the same three rules as
the specialised kernels, with shared memory where those use registers.

**Why large ranks need their own kernel.** Large ranks used to run the generic shared-memory
kernels. Those lose time in three ways, which grow with $L$:

- **Few threads per SM.** Every thread stages its own length-$L$ vector in shared memory
  (576 bytes at $L = 72$). Only about 12 warps then fit on an SM of the H100, and about 5 on the
  L40S, so too few loads are in flight to use the memory bandwidth.
- **Two reads per stored element.** The generic kernels compute one output at a time from the
  full $L \times L$ block, so every off-diagonal element of `phi2_toep` comes from DRAM twice.
- **One coil per launch.** `phi2_toep` is read once per coil. The masked variant also needs a
  compact buffer, a memset and a scatter pass.

The block-diagonal stage matters more as $L$ grows: at $L = 72$ `phi2_toep` holds 2628 values per
location, against 72 values of data.

**One tile of locations per thread block.** A block of 256 threads owns a *tile* of 16
consecutive grid locations and a batch of up to 4 coils of the coil group (§2.3):

1. The block stages the inputs of its tile, `[coil, rank, location]`, in shared memory.
2. It walks the stored triangle in *panels* of whole stored columns. Each panel is copied from
   DRAM to shared memory once, for all 16 locations. Consecutive locations sit next to each
   other in `phi2_toep`, so these loads are coalesced.
3. Each thread owns one location and a few of its outputs, with one accumulator per output and
   coil in registers. It adds every contribution the panel holds to them, reading the panel and
   the inputs from shared memory. As in §2.1, one stored element $A_{rc}$ feeds two outputs
   ($y_c$ directly, $y_r$ through its conjugate).
4. After the last panel the block writes the outputs back in place.

So each stored element comes from DRAM **once per coil batch**, the panel that was read serves
every coil of the batch, and the accumulators never leave the registers.

**Asynchronous copies.** On GPUs of compute capability 8.0 or newer the panel is copied with
`cp.async`, which moves data from the L2 cache to shared memory without passing through
registers. A thread can then have all the loads of a panel in flight at once. Ordinary loads
were 25–35 % slower on the whole operator (H100: 45 ms against 34 ms; L40S: 101 ms against
86 ms; `2d72`, masked). Older GPUs such as the TITAN RTX use ordinary loads.

**The launch configuration depends on the GPU.** `mr_bdt_pick` chooses it at every launch from
the shared memory and the registers of the device. The reasons were measured on `2d72` with the
masked operator on the H100:

| setting | choice | measured |
|---|---|---|
| coils per batch | as many of the group as the shared memory allows, 4 at most | 34 ms with 4 coils, 49 ms with 2 |
| locations per tile | 16; 8 above $L = 256$, where a thread would own too many outputs | |
| thread blocks per SM | two, if a panel of at least $2L$ elements still fits; else one | 34 ms with two, 43 ms with one |
| panel | the rest of the shared memory, up to the whole triangle | |

The rule for two blocks comes from the L40S, which has less shared memory per SM: two blocks
would leave a panel of only 93 elements there and take 89 ms, against 86 ms with one block.
`MR_AHA_CUDA_BDT_NC`, `MR_AHA_CUDA_BDT_P` and `MR_AHA_CUDA_BDT_BLOCKS` override the choice, for
tuning.

**What stays the same.** The tiled kernels have the contract of the specialised ones: they work
in place on the padded grids, the masked variant goes through the index map of §2.4, they serve
coil groups (§2.3), and they accept the location ranges of the remainder split (§3.2). The plan
launches both families through the same two functions (`mr_bd_launch_dense`,
`mr_bd_launch_masked`). The generic kernels remain as the fallback for a mask of more than 2³¹
locations and behind `MR_AHA_CUDA_GENERIC_BD=1`; `MR_AHA_CUDA_TILED_BD=1` selects the tiled
kernels for every rank, for tests and A/B timing.

**The effect.** One $A^H A$ on `2d72` (320² image, $L = 72$, 12 coils), in ms. "Generic" is the
same build with `MR_AHA_CUDA_GENERIC_BD=1`:

| GPU | unmasked, generic | unmasked, tiled | gain | masked, generic | masked, tiled | gain |
|---|---:|---:|---:|---:|---:|---:|
| H100 | 335.3 | **36.2** | 9.3× | 165.5 | **32.9** | 5.0× |
| L40S | 309.1 | **90.0** | 3.4× | 287.8 | **85.8** | 3.4× |
| TITAN RTX | 663.0 | **223.7** | 3.0× | 574.6 | **198.0** | 2.9× |

With the generic kernels the operator hardly gained from the L40S to the H100 at this rank; with
the tiled kernels it gains 2.5×, in step with the other problems.

On the H100 the block-diagonal stage of the masked operator takes 22.3 of 32.9 ms, and each of
the two FFTs 4.1 ms.


## 3. Multi-GPU

### 3.1 Decomposition by coil

$A^H A x = \sum_c (\ldots)_c$, and each coil's pipeline is independent. The operator object
(`mr_op_create`, `src/cuda/mr_op.cu`; `create_op` in Python and MATLAB) gives each GPU a
contiguous range of whole coils. With the Toeplitz method each GPU runs an ordinary single-GPU
plan for its coils, with its own coil groups, specialised kernels and stream.

At creation every GPU receives a copy of `phi2_toep` (and the mask) and the sensitivity maps of
its own coils. One application then only moves `x` in and a partial result out:

```
caller's stream ── event ──┬─▶ GPU k: copy x ─ its coils ─ copy partial to the primary ─ event
                           └─▶ primary: its coils, straight into the output ─ event
caller's stream waits for all events ─▶ one kernel adds the staged partials into the output
```

A single host thread issues all of this asynchronously, with cross-device ordering through events
only. No host thread blocks, and the call returns immediately, like the single-GPU one. Every entry
point saves and restores the caller's current device, so MATLAB's "one GPU per session" model and
CuPy's current device are unaffected.

**How many GPUs.** The time is set by the busiest GPU, which has $\lceil C/G \rceil$ coils. The
operator uses the smallest $G' \le G$ with the same $\lceil C/G' \rceil$: 12 coils on 5 GPUs use
4. Problems below $2^{22}$ upsampled values per coil stay on one GPU, where they are bound by
launch overhead (the 2D benchmark takes 0.5 ms).

**Scaling.** On three H100s the speed-up is ×2.76 (stack) to ×2.87 (3D). The rest is the copy of
`x` (≈ 1.5 ms over NVLink for the stack problem), the gather and the reduction.

### 3.2 Remainder split by rank

When $C$ is not a multiple of $G$, the last $r = C \bmod G$ coils do not fill every GPU. The
operator can split them across *all* GPUs by rank. Padding, FFTs and cropping are independent per
rank, but the block-diagonal stage needs all $L$ ranks of a location together, so the data is
transposed twice:

1. GPU $g$ pads and transforms its ranks $R_g$ (about $L/G$ of them);
2. **all-to-all:** GPU $g$ sends the k-space slab $K_h$ (a contiguous range of about
   $M_xM_yM_z/G$ locations) of its ranks to GPU $h$, which then holds all $L$ ranks of $K_h$;
3. GPU $h$ multiplies $K_h$ with its copy of the kernel (the specialised kernels take a location
   range for this, `mr_bd_range`);
4. all-to-all back, inverse FFT and crop of $R_g$.

The copies are one `cudaMemcpyPeerAsync` per rank. Each GPU does $1/G$ of the remaining coil's
arithmetic but moves its padded data twice, so the split pays over fast links.

**Cost model** (`MR_AHA_MGPU_SPLIT=auto`, the default). At plan creation the library measures the
GPU-to-GPU bandwidth $\text{bw}$ (64 MB copies) and times the single-GPU operator on the primary
over $n$ coils, $T(n)$, with its coil groups. It then compares

$$
t_\text{whole} = T(\lceil C/G' \rceil), \qquad
t_\text{split} = T(\lfloor C/G \rfloor) + r\left(\frac{T(1)}{G} + \frac{2(G-1)}{G^2}\,
\frac{8 M_xM_yM_zL}{\text{bw}}\right),
$$

and splits only if $t_\text{split} < 0.95\, t_\text{whole}$. On H100s over NVLink it splits 7
coils on 3 GPUs (3D), which gains 4 %, and keeps whole coils for 8 coils on 3 GPUs, where whole
coils are faster. Over PCIe without peer access (~4 GB/s) it keeps whole coils.

### 3.3 Robustness

- **Device resets under a live plan.** MATLAB resets a GPU on `reset(gpuDevice)` and whenever one
  is selected or deselected. That can destroy the CUDA context of the primary *and* of other GPUs
  of a plan, together with the plan's streams, events and memory. Each device slot therefore
  records the unique ID of its context (`cuCtxGetId`). Release first checks that the primary
  context is active (`cuDevicePrimaryCtxGetState`, which does not create one) and that it is still
  the same instance, and skips the device otherwise. `mr_op_abandon` additionally skips the
  primary without probing it. The MATLAB gateway detects resets through a one-element sentinel
  gpuArray per plan.
- **Failure mid-issue.** If an execution fails after some work was issued, the call waits for the
  device streams before returning, so nothing keeps reading `x` or writing the output behind the
  caller's back.
- **Memory.** Every GPU holds a full copy of the kernel, so multi-GPU does not raise the largest
  problem size. `MR_OP_SHARE_PRIMARY` (`share_primary` in Python and MATLAB) lets the primary GPU
  use the caller's arrays in place instead of a second copy.

### 3.4 Set-up: computing `phi2_toep` on all GPUs

A multi-GPU Toeplitz operator needs the kernel on every GPU. The operator computes it on all its
GPUs at once: each GPU computes about $1/G$ of the kernel and receives the other $(G-1)/G$ over
its own link. The kernel is therefore on every GPU sooner than one GPU could compute it alone.

**Why the work splits cleanly.** The $(L^2+L)/2$ triangular components of the kernel are
independent: each one is a type-1 NUFFT, a centred inverse FFT and a scaling. Each fills one
contiguous slab of the output (`Mx*My*Mz` elements, or `num_masked` when masked). A range of
components is therefore one contiguous byte range. It can be computed on any GPU and moved with
a single `cudaMemcpyPeerAsync`. `calc_toep` works through a per-device context (`mr_toep_ctx`:
packed Gram blocks, chunk buffers, cufinufft and cuFFT plans), which evaluates any range of
components into any destination (`src/cuda/mr_calc_toep.cu`).

**The engine** (`mr_toep_distribute`, `src/cuda/mr_toep_mgpu.cu`) runs one host thread per GPU:

1. The components are split into $G$ contiguous, nearly equal shares. Chunks are capped at a
   quarter of a share, so that transfers can start early.
2. Every GPU copies the small inputs from the primary (`phi2` and the trajectory, tens of MB). It
   plans, then computes and times the first chunk of its share (the *pilot*). The pilot is useful
   work whichever mode follows.
3. A cost model decides between two modes, *exchange* and *self*. Let $T$ be the number of
   components, $p$ the size of the pilot, $t_e$ the slowest pilot time per component, $c$ the
   components each GPU still has to compute, $R$ the time a GPU needs to receive the components it
   does not compute, and $k$ the number of chunks:

   $$T_\text{exchange} \approx \max(c\,t_e, R) + R/k, \qquad T_\text{self} \approx (T - p)\,t_e .$$

   $R$ comes from a bandwidth probe that runs on all GPUs at once, with the exchange's own
   schedule. Roughly, the exchange wins whenever a GPU can receive a component faster than it
   computes one: about 4 GB/s for the 2D $L = 72$ kernel.
4. **Exchange.** Each GPU computes the rest of its share directly into its own copy of the
   kernel. After each chunk it sends the chunk to the other GPUs, overlapped with the next
   chunk. Every component is computed once, so all copies are bit-identical.

   **Self.** Every GPU computes the components it lacks itself, with no transfers. This is for
   GPUs without peer access.

Every GPU of the operator receives the whole kernel. GPUs that the operator does not use for coils
still help compute it. The `toep_dist` argument selects the mode, and the environment variable
`MR_AHA_TOEP_SPLIT` (same values, `off` = `broadcast`) overrides it:

| `toep_dist` | what happens |
|---|---|
| `'auto'` (default) | the cost model of step 3 chooses between `'split'` and `'replicate'` |
| `'split'` | exchange: each GPU computes a share, the shares are exchanged |
| `'replicate'` | self: every GPU computes the whole kernel |
| `'broadcast'` | the primary computes the whole kernel and copies it to the others |

**One copy at a time, in rotated order.** Each GPU has a single copy stream and sends each chunk
to the receivers in rotated order $k+1, k+2, \ldots$ $(\bmod\ G)$. Because the GPUs advance in
step, the destinations form a permutation at any moment, so every GPU has about one outgoing and
one incoming transfer. This is a ring all-gather without the forwarding. With this schedule each
GPU receives at 14–16 GB/s on four L40S that sit on three different NUMA nodes
(`nvidia-smi topo`: every pair `SYS`), and at about 365 GB/s on 2 or 4 H100 joined by NVLink.
The bandwidth probe uses the same schedule, so the cost model sees the bandwidth the exchange will
actually get.

**Measured** on two systems: 4 L40S over PCIe (two pairs, each pair on its own PCIe switch, the
pairs on different NUMA nodes) and 4 H100 with NVLink between every pair. Synthetic inputs have the
shapes of the benchmark problems, 12 coils each. Each figure is the median of 2 runs after a
warm-up, in seconds, L40S / H100. The fastest set-up per problem and system is in bold.

| set-up (s), L40S / H100 | 2D radial 320², $L = 72$, 8.6 GB | stack 320² × 56, $L = 12$, 14.3 GB | 3D 160³, $L = 12$, 20.5 GB |
|---|---|---|---|
| 1 GPU (`calc_toep`) | 2.05 / 2.16 | 2.41 / 1.93 | 7.75 / 3.97 |
| 2 GPUs, `'split'` | **1.46** / **1.06** | **2.15** / **1.48** | 5.36 / 2.81 |
| 4 GPUs, `'split'` | 1.82 / 1.32 | 2.40 / 1.69 | **4.20** / **2.36** |
| 4 GPUs, `'auto'` (chose `'split'`) | 1.88 / 1.32 | 2.54 / 1.50 | 4.31 / 2.39 |
| 4 GPUs, `'split'` without overlap | 2.22 / 1.34 | 2.86 / 1.56 | 4.60 / 2.31 |
| 4 GPUs, `'replicate'` | 4.12 / 4.03 | 4.15 / 3.58 | 9.69 / 6.63 |
| 4 GPUs, `'broadcast'` | 3.58 / 2.27 | 5.03 / 2.02 | 11.34 / 4.60 |

- **Faster than one GPU.** Building the 4-GPU operator, with the kernel on all four GPUs, takes
  less time than `calc_toep` alone on one GPU, in every case.
- **Two GPUs can set up faster than four.** In 2D and stack, 2 GPUs beat 4 on both systems, even
  with NVLink. Going from 2 to 4 GPUs adds about 0.3 s of per-GPU set-up and saves less than that
  in the exchange (2D, `'split'`):

  | | L40S, 2 GPUs | L40S, 4 GPUs | H100, 2 GPUs | H100, 4 GPUs |
  |---|---|---|---|---|
  | setup and pilot | 0.66 s | 0.95 s | 0.49 s | 0.78 s |
  | exchange | 0.59 s | 0.51 s | 0.43 s | 0.26 s |
  | bandwidth received per GPU | 25.8 GB/s | 14.4 GB/s | 368 GB/s | 364 GB/s |

  Set-up runs on all GPUs at once in one process and gets slower as more GPUs take part (see
  `'replicate'` below). On the L40S the exchange also gains little, because each GPU receives
  $3/4$ of the kernel instead of $1/2$, at a lower rate across NUMA nodes. In 3D each component is
  expensive (≈ 90 ms against 0.5 ms in 2D), so halving each GPU's share outweighs the extra set-up.
- **End to end, 4 GPUs are still fastest.** More GPUs make every CG iteration faster. Set-up plus
  10 CG iterations, `'split'`:

  | set-up + 10 CG (s), L40S / H100 | 2D | stack | 3D |
  |---|---|---|---|
  | 1 GPU | 5.95 / 5.68 | 10.80 / 3.65 | 19.27 / 6.68 |
  | 2 GPUs | 3.07 / 2.75 | 6.68 / 2.46 | 11.26 / 4.14 |
  | 4 GPUs | 2.72 / 2.17 | 5.60 / 2.25 | 8.04 / 3.09 |

- **Overlap** of computation and exchange saves about 0.4 s on the L40S. With NVLink the transfer
  takes almost no time, so the overlap makes no measurable difference.
- **`'auto'` costs at most 0.15 s more than forcing `'split'`**, the bandwidth probe plus
  run-to-run spread. It chose the exchange in every case above. In an earlier run on another set of
  4 L40S, the exchange was slow (2D `'split'` 5.6 s); `'auto'` chose `'replicate'` there and took
  3.0 s, the fastest of the three modes.
- **`'replicate'` is slower than one GPU.** Each GPU runs the whole single-GPU computation, so all
  of them should finish in about the single-GPU time. They do not, because the GPUs are driven by
  threads of one process, and the threads contend inside it, for example for locks in the CUDA
  driver and for the CPU-side work of the cuFFT and cufinufft calls. The 2D figure is the same on both systems (4.12 and 4.03 s), although
  the H100 computes the 3D kernel twice as fast. Per-component time on the H100 rises from 0.74 ms
  with 2 GPUs to 1.50 ms with 4 in 2D, and from 45 to 68 ms in 3D. Run as 4 separate processes
  instead, one GPU each, the same computations finish close to the single-GPU time (H100, median of
  5 runs):

  | `calc_toep` (s), H100 | 2D | stack | 3D |
  |---|---|---|---|
  | 1 process, 1 GPU | 2.01 | 1.67 | 4.11 |
  | 4 processes at once, 1 GPU each | 2.15–2.30 | 1.70–1.79 | 4.22–4.27 |
  | 1 process, 4 GPUs (`'replicate'`) | 4.03 | 3.58 | 6.63 |

  The slowdown therefore comes from the threads sharing one process, not from the GPUs or the
  node. The same contention lengthens the set-up phase of `'split'`.
- **Where the 4-GPU time goes** (2D, `'split'`, L40S / H100):
  - setup and pilot, 0.95 / 0.78 s: per GPU, input copies, cufinufft and cuFFT planning, and a
    pilot of 165 components (≈ 0.09 s);
  - exchange, 0.51 / 0.26 s: 492 more components per GPU (≈ 0.26 s) overlapped with receiving
    6.5 GB;
  - releasing the buffers, 0.05 / 0.01 s;
  - building the operator's own state, ≈ 0.26 s on both.

  Planning is the largest part.
- **Tuning tip: a smaller chunk also helps a single GPU.** With `MR_AHA_TOEP_MAX_CHUNK=128`,
  single-GPU `calc_toep` for 2D $L = 72$ drops from 2.05 to 1.53 s on the L40S and from 2.16 to
  1.30 s on the H100 (smaller buffers and plans). The 3D problem is unaffected.


### 3.5 The NUFFT method of the operator

`create_op(..., method='nufft')` (`src/cuda/mr_nufft_op.cu`) evaluates the same operator without a
kernel. Per coil, the coil-weighted image goes through one batched type-2 cufinufft (all $L$
ranks, $L N_z$ planes for a stack), the Gram multiplication, and one batched type-1 cufinufft
back. Two choices keep it fast:

* **Samples ordered by readout.** cufinufft receives the trajectory with the samples of each
  readout contiguous (`kx + Nkx*t`). The rank vectors of a readout are then an `Nkx x L` matrix
  with leading dimension `Ntrajs*Nkx`, and the whole Gram multiplication is one
  `cublasCgemmStridedBatched` over the readouts ($K_t \leftarrow K_t \Phi_{2,t}$). For a stack it
  is one `cublasCgemmBatched` over readouts and kz encodes, with pre-computed pointer arrays. The
  data is never transposed, and any $L$ works. cufinufft bins the points itself, so the sample
  order changes only the floating-point summation order.
* **Everything inside the library.** There is no per-coil overhead from Python or MATLAB, and the
  same code serves both languages and multi-GPU. The coils are split over the GPUs as for the
  Toeplitz method; each GPU holds its own plans, Gram blocks and trajectory.

**Measured** on one TITAN RTX, random inputs, per application of $A^H A$
(median of 5), against the Python reference `aha_nufft_plan` on the same inputs (relative
difference ≤ 2.4·10⁻⁷):

| problem | `aha_nufft_plan` | `create_op(..., method='nufft')` |
|---|---|---|
| 2D 256², $L = 40$, 8 coils, 400 × 512 samples | 173.5 ms | 146.2 ms |
| 2D 320², $L = 72$, 4 coils, 754 × 320 samples | 206.3 ms | 170.4 ms |
| stack 128² × 32, $L = 12$, 8 coils, 200 × 256 samples | 648.6 ms | 621.7 ms |
| 3D 64³, $L = 8$, 4 coils, 2000 × 128 samples | 187.8 ms | 183.2 ms |

The time is dominated by the two cufinufft transforms. Set-up (plans and point sorting) takes a
few milliseconds. For comparison, the Toeplitz method applies the 2D 320², $L = 72$ operator with
12 coils in 0.23 s on the same GPU, after a 2.6 s set-up for the kernel.


## 4. Verification

The tests are described in [Testing](../development/testing.md). The checks behind this note:

- **Specialised and tiled against generic kernels.** `tests/check_bd_kernels.py` applies the
  operator with the default kernels and with the generic ones for 14 ranks from 1 to 130, both
  triangles, masked and unmasked, with coil groups of 1, 2 and 4. It also forces the tiled
  kernels at every rank, with coil batches of 1, 2 and 4. All 504 combinations agree to
  single-precision rounding (worst relative difference 2.8·10⁻⁷).
- **Reference data.** `tests/check_reference.py` and `tests/check_reference_matlab.m` compare
  every public function with stored results.
- **Multi-GPU against single-GPU.** `tests/check_op.py` and `tests/check_op_matlab.m` cover 2D, stack
  and 3D problems, masked and unmasked, both triangles, every `toep_dist`, and the remainder
  split on and off. Measured during development:
  - the multi-GPU operator agrees with the single-GPU one to 7·10⁻⁸;
  - a kernel computed on 2, 3 or 4 GPUs agrees with the single-GPU kernel to 3.5·10⁻⁷, which is
    cufinufft's run-to-run noise;
  - with `'split'`, the kernel copies on all GPUs are bit-identical;
  - the current device is unchanged, and no memory leaks;
  - a GPU whose memory was filled on purpose gives a clean error that names that GPU.
- **MATLAB device handling.** The MATLAB test also covers `clear mex`, `reset(gpuDevice)` and
  switching `gpuDevice` under a live operator.
- **Reconstruction.** With the specialised kernels and coil groups, the 20-iteration CG solution
  of the stack problem moves by 2.5·10⁻⁶. That is the run-to-run noise of the NUFFT operator
  (2.8·10⁻⁶).
