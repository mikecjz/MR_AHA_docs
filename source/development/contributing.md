# Contributing

Bug reports, questions, benchmarks on new hardware and pull requests are welcome. The full
guide is [`CONTRIBUTING.md`](https://github.com/mikecjz/MR_AHA/blob/main/.github/CONTRIBUTING.md)
in the library repository. In short:

* **Contributor License Agreement.** MR_AHA is dual-licensed (see [Licensing](licensing.md)),
  so every contributor signs the
  [CLA](https://github.com/mikecjz/MR_AHA/blob/main/licenses/CLA.md) once, by commenting on
  their first pull request.
* **Conventions.** Follow the [array layout and Gram-block conventions](../guide/conventions.md).
* **Comments are the documentation.** The API reference of this site is generated from the
  comments in the code: Doxygen comments for C, C++ and CUDA, docstrings for Python, help text
  for MATLAB. Document every public function's array shapes, types and units.
* **Accuracy.** Changes to the numerics must keep the [tests](testing.md) passing at their
  tolerances.
* **Performance.** For changes aimed at speed, include before/after timings, for example from
  [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks), and the GPU they were
  measured on.

## Updating this site

The sources are in [MR_AHA_docs](https://github.com/mikecjz/MR_AHA_docs). The guides are
Markdown files under `source/`; the API reference is generated at build time from the library,
which is a pinned submodule (`third_party/MR_AHA`). To publish the reference of a newer
library version, move the submodule to that commit and push; the site rebuilds on every push to
`main`.
