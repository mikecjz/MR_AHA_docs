# C / C++

```c
#include "mr_aha/mr_aha.h"

mr_config cfg;
mr_config_defaults(&cfg, MR_TRAJ_STACK);
cfg.n_dim[0] = 320; cfg.n_dim[1] = 320; cfg.n_dim[2] = 56;
cfg.phi2_dim[0] = 640; cfg.phi2_dim[1] = 640; cfg.phi2_dim[2] = 56;
cfg.L = 16; cfg.n_coils = 12;

mr_op_inputs in = {0};
in.d_SEs = d_SEs; in.d_phi2 = d_phi2; in.d_traj = d_traj;
in.Ntrajs = Ntrajs; in.Nkx = Nkx;               /* d_mask when masked */
mr_op_options opt;
mr_op_options_defaults(&opt);                   /* Toeplitz, current GPU */
/* opt.method = MR_OP_NUFFT; opt.devices = devs; opt.n_devices = 4; */

mr_op op;
mr_op_create(&cfg, &in, &opt, &op);             /* kernel computed here */
for (int it = 0; it < n_iter; ++it)
    mr_op_execute(op, d_x, d_AhAx);
mr_op_destroy(op);
```

Every entry point returns `MR_SUCCESS` (0) or a negative `mr_status`; `mr_status_string` gives the
category and `mr_last_error` the detailed, thread-local diagnostic. The stateless single-GPU
primitives `mr_calc_toep`, `mr_aha_plan_*` and `mr_aha_toeplitz` remain available. Link against `mr_aha`, or `target_link_libraries(your_target mr_aha)`
after `add_subdirectory(MR_AHA)`.

[`examples/example_aha.cu`](https://github.com/mikecjz/MR_AHA/blob/main/examples/example_aha.cu) is a complete, self-contained program that
builds a synthetic stack-of-stars problem, checks the operator's Hermitian symmetry and compares
the two methods.
