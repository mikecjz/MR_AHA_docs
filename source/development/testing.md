# Testing

The tests are scripts in `tests/` of MR_AHA. They run against whatever library the Python
package or the MATLAB path loads (`MR_AHA_LIBRARY` selects a specific build) and work on both
backends: arrays are CuPy arrays on CUDA and NumPy arrays on Metal.

## Reference checks

```bash
MR_AHA_LIBRARY=build/lib/libmr_aha_c.so python tests/check_reference.py
python tests/check_reference.py --stages aha --cases 2d_50x40_L3_c3
matlab -batch "run('tests/run_matlab_tests.m')"
```

`tests/check_reference.py` re-runs every public function on stored inputs and compares the
outputs with stored CUDA results. The stages can be checked separately, which is how a new
backend is validated piece by piece:

| Stage | What is compared |
|---|---|
| `aha` | `aha_toeplitz` with each stored kernel (lower/upper, unmasked and masked) |
| `toep` | `calc_toep` on the stored Gram blocks and trajectory |
| `mask` | `calc_mask` (the mask must match exactly) |
| `nufft` | the NUFFT reference operator `AhANufftPlan` |
| `op` | `create_op` with both methods |

`tests/check_reference_matlab.m` does the same through the MATLAB interface;
`tests/run_matlab_tests.m` puts the library on the path and runs it together with
`check_op_matlab`.

The reference files (`data/ref_mlx/*.npz`) are written by `tests/make_reference.py` from the
CUDA backend: small synthetic cases covering the three trajectory types, odd and even sizes,
odd upsampled sizes, and `L` in {1, 3, 4}, plus one case on real 2D radial data.
`tests/export_reference_mat.py` converts them to `.mat` files for MATLAB. The data files are
not part of the repository; generate them on a CUDA machine:

```bash
python tests/make_reference.py            # generate and verify
python tests/export_reference_mat.py      # for the MATLAB checks
```

## The operator object

```bash
CUDA_VISIBLE_DEVICES=1 python tests/check_op.py            # one GPU
python tests/check_op.py --devices 0 1 2 3 [--quick]       # multi-GPU
```

`tests/check_op.py` (CUDA) checks `create_op` against the stateless primitives and the
independent NUFFT reference on random 2D, stack and 3D problems.

On one GPU:

* a Toeplitz operator with an injected kernel is bit-identical to `aha_toeplitz`;
* with its own kernel it matches `calc_toep` and `aha_toeplitz` to cufinufft's run-to-run noise;
* the NUFFT method matches `AhANufftPlan` to `1e-5`.

Multi-GPU, it checks every `toep_dist`, the remainder split by rank, the kernel copies on
every GPU, memory leaks, and a forced allocation failure on another GPU.

`tests/check_op_matlab.m` is the MATLAB counterpart. Its multi-GPU checks run only with
`MR_AHA_TEST_MGPU=1`.

## Block-diagonal kernels

```bash
python tests/check_bd_kernels.py
```

Both backends use block-diagonal kernels specialised on `L` and fall back to generic kernels
for larger ranks. This test applies the operator with both kernel families on random inputs for
many ranks, both triangulations, unmasked and masked, with several coil-group sizes, and checks
that they agree to single-precision rounding.
