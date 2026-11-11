#!/usr/bin/env python
"""Generate the MATLAB reference pages from the help blocks of the ``.m`` files.

MATLAB help text follows MATLAB's own conventions (upper-case function names,
indented argument tables), not reStructuredText, so it is reproduced verbatim:
every function or class of the MATLAB interface gets one page with its
signature and its help block in a preformatted block, exactly what
``help <name>`` prints in MATLAB. An index page lists them with their one-line
summaries.

Usage::

    python tools/gen_matlab_api.py <MR_AHA>/matlab source/api/matlab

``source/conf.py`` runs this at every build, so the pages follow the pinned
``third_party/MR_AHA`` submodule and are not committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Functions shown first on the index page, in this order; the remaining
#: files follow alphabetically under "Helpers".
MAIN = ["mr_aha_op", "mr_default_config", "calc_toep", "AhA_Toeplitz",
        "calc_mask", "mr_nufft_plan", "AhA_cufinufft", "mr_backend"]


def parse(path: Path) -> tuple:
    """Extract the declaration and the help block of a MATLAB file.

    The help block is the run of comment lines that follows the ``function``
    or ``classdef`` line (continuation lines of the declaration included).

    Args:
        path: The ``.m`` file.

    Returns:
        ``(declaration, summary, help_lines)``: the declaration line(s) joined,
        the one-line summary (the H1 line without the function name) and the
        help text with the comment markers removed.
    """
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    decl = []
    while i < len(lines):
        decl.append(lines[i].strip())
        cont = lines[i].rstrip().endswith("...")
        i += 1
        if not cont:
            break
    help_lines = []
    while i < len(lines) and lines[i].lstrip().startswith("%"):
        text = lines[i].lstrip()[1:]
        help_lines.append(text[1:] if text.startswith(" ") else text)
        i += 1
    summary = ""
    if help_lines:
        first = help_lines[0].strip()
        head, _, rest = first.partition(" ")
        summary = rest.strip() if head.upper() == path.stem.upper() else first
        # An H1 line may continue on the next line(s) up to the first blank one.
        for extra in help_lines[1:]:
            if not extra.strip():
                break
            summary += " " + extra.strip()
    return " ".join(d.rstrip(".").strip() for d in decl), summary, help_lines


def main(src: str, dst: str) -> int:
    """Write one page per ``.m`` file of ``src`` and an index into ``dst``.

    Args:
        src: The ``matlab/`` folder of an MR_AHA checkout.
        dst: Output folder (created; existing pages are overwritten).

    Returns:
        0 on success, 1 if ``src`` has no ``.m`` files.
    """
    src_dir, dst_dir = Path(src), Path(dst)
    files = sorted(src_dir.glob("*.m"), key=lambda p: p.stem.lower())
    if not files:
        print(f"gen_matlab_api: no .m files in {src_dir}", file=sys.stderr)
        return 1
    dst_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for f in files:
        decl, summary, help_lines = parse(f)
        fence = "````"
        page = [f"# {f.stem}", "", summary, "", "```matlab", decl, "```", "",
                f"{fence}text", *help_lines, fence, "",
                f"Source: [`matlab/{f.name}`](https://github.com/mikecjz/MR_AHA/"
                f"blob/main/matlab/{f.name})", ""]
        (dst_dir / f"{f.stem}.md").write_text("\n".join(page))
        rows[f.stem] = summary

    main_names = [n for n in MAIN if n in rows]
    helpers = [n for n in rows if n not in main_names]
    out = ["# MATLAB API", "",
           "The MATLAB interface is a set of `.m` files in `matlab/` on top of MEX",
           "gateways. Each page below reproduces the help text of one of them, the",
           "same text `help <name>` prints in MATLAB.", ""]
    for title, names in (("Main functions", main_names), ("Helpers", helpers)):
        if not names:
            continue
        out += [f"## {title}", "", "| | |", "|---|---|"]
        out += [f"| [`{n}`]({n}.md) | {rows[n]} |" for n in names]
        out += ["", "```{toctree}", ":hidden:", "", *names, "```", ""]
    (dst_dir / "index.md").write_text("\n".join(out))
    print(f"gen_matlab_api: {len(rows)} pages in {dst_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
