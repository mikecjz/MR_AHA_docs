# MR_AHA_docs

Sources of the documentation website of [MR_AHA](https://github.com/mikecjz/MR_AHA), the GPU
library for the forward normal operator $A^H A$ of non-Cartesian MRI with low-rank modelling.

**The site: <https://mikecjz.github.io/MR_AHA_docs/>**

## What is here

```
source/                  the pages (Markdown, MyST) and the Sphinx configuration
  getting-started/       installation; Python, MATLAB and C quick starts
  guide/                 operations, conventions, masking, multi-GPU, Apple silicon
  background/            why Toeplitz, implementation notes, performance notes
  benchmarks/            published benchmark studies
  api/                   API reference (generated at build time, see below)
  development/           testing, VS Code, contributing, licensing
tools/gen_matlab_api.py  writes the MATLAB reference pages from the help blocks
Doxyfile                 Doxygen settings for the C reference
third_party/MR_AHA/      the library, as a pinned submodule
.github/workflows/       builds the site and publishes it to GitHub Pages
```

The guides are written by hand. The API reference is generated from the comments in the
library, so it is never edited here:

| Reference | Generated from | By |
|---|---|---|
| Python | docstrings in `python/mr_aha/` | sphinx-autoapi (reads the source; nothing is imported, so no GPU is needed) |
| C | Doxygen comments in `include/mr_aha/mr_aha.h` | Doxygen + Breathe |
| MATLAB | help blocks in `matlab/*.m` | `tools/gen_matlab_api.py` |

## Build locally

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_docs.git
cd MR_AHA_docs
pip install -r requirements.txt          # plus Doxygen from your package manager
sphinx-build -b html -W --keep-going source _build/html
```

Open `_build/html/index.html`. `-W` turns warnings into errors, as the published build does.

To preview docstring changes before they are committed to the library, point the build at your
working checkout:

```bash
MR_AHA_ROOT=~/code/MR_AHA sphinx-build -b html source _build/html
```

## Publishing

Every push to `main` rebuilds the site and publishes it
([workflow](.github/workflows/docs.yml)).

The API reference follows the pinned submodule, not the latest library. To publish the
reference of a newer library commit:

```bash
git -C third_party/MR_AHA fetch
git -C third_party/MR_AHA checkout <commit or tag>
git add third_party/MR_AHA
git commit -m "Reference: MR_AHA <commit or tag>"
git push
```

## License

Copyright (C) 2024-2026 Junzhou Chen. The documentation is released under the same terms as
MR_AHA, the [GNU AGPL v3 only](LICENSE); see
[MR_AHA's licensing notes](https://github.com/mikecjz/MR_AHA/blob/main/licenses/LICENSING.md).
