#!/usr/bin/env python3
"""Export marimo notebooks as downloadable Jupyter notebooks.

Wraps `marimo export ipynb` -- marimo's own supported conversion -- and applies
two small edits that a *downloaded* notebook needs and the in-repo one does not:

1. **A dependency cell.** The marimo notebook declares its dependencies in a
   PEP 723 header, which `marimo export ipynb` does not carry across (Jupyter has
   no equivalent). Those dependencies are read out of the header and turned into a
   `%pip install` cell, so the download runs from a bare kernel.

2. **Bootstrap surgery.** A book notebook resolves `chaoslib` either by
   micropip-installing a bundled wheel (in the browser) or by adding the repo
   root to `sys.path` (locally). Neither works for a file someone downloaded, so
   that block is removed and `chaoslib` is pip-installed from the public repo
   instead.

A header markdown cell is prepended explaining where the notebook came from and
the one thing that genuinely differs from the live version: marimo's sliders are
inert in Jupyter, so a parameter is changed by editing `value=` and re-running.

Usage:
    python3 scripts/export_ipynb.py notebooks/chNN.py -o site/static/ipynb/chNN.ipynb
    python3 scripts/export_ipynb.py notebooks/*.py --outdir site/static/ipynb
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: Installed from the public repo, because a downloaded notebook has no checkout.
CHAOSLIB_SPEC = (
    '"chaoslib @ git+https://github.com/anichaospred/anichaospred.github.io'
    '#subdirectory=chaos-book"'
)


def read_pep723_dependencies(path: Path) -> list[str]:
    """Pull the dependency list out of a notebook's PEP 723 script header.

    Returns the requirement strings with any trailing comment stripped, so
    `"marimo==0.23.9",   # pinned: ...` becomes `marimo==0.23.9`.
    """
    text = path.read_text()
    block = re.search(r"# /// script\n(.*?)# ///", text, re.S)
    if not block:
        return []
    deps: list[str] = []
    inside = False
    for line in block.group(1).splitlines():
        stripped = line.lstrip("#").strip()
        if stripped.startswith("dependencies"):
            inside = True
            continue
        if inside:
            if stripped.startswith("]"):
                break
            match = re.match(r'"([^"]+)"', stripped)
            if match:
                deps.append(match.group(1))
    return deps


def strip_pyodide_bootstrap(source: str) -> tuple[str, bool]:
    """Remove the emscripten/sys.path branch from an import cell.

    Returns ``(new_source, needed_chaoslib)``. The block is identified by
    ``sys.platform == "emscripten"`` and removed together with the ``import sys``
    that serves it; the surviving imports (marimo, numpy, chaoslib, ...) are left
    exactly as the chapter wrote them.

    Line-based rather than AST-based on purpose: the cell must come out looking
    like hand-written code, and an AST round-trip would discard the comments that
    explain it.
    """
    if "emscripten" not in source:
        return source, False

    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.fullmatch(r"\s*import sys\s*", line):
            i += 1
            continue
        if "sys.platform" in line and "emscripten" in line:
            # Skip the whole if/else: everything until a line that is neither
            # blank nor indented past column 0.
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "" or nxt.startswith((" ", "\t")) or nxt.strip().startswith("else:"):
                    i += 1
                    continue
                break
            continue
        out.append(line)
        i += 1

    # Collapse the run of blank lines the removal leaves behind.
    cleaned: list[str] = []
    for line in out:
        if line.strip() == "" and cleaned and cleaned[-1].strip() == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n", True


def build_install_cell(deps: list[str], needs_chaoslib: bool) -> list[str]:
    """The `%pip install` cell placed at the top of the download."""
    pip_deps = " ".join(f'"{d}"' for d in deps) if deps else ""
    lines = [
        "# Dependencies for this notebook. Safe to skip if you already have them.\n",
        "# The versions come from the marimo notebook's own script header, so this\n",
        "# reproduces the environment the published version runs in.\n",
    ]
    if pip_deps:
        lines.append(f"%pip install -q {pip_deps}\n")
    if needs_chaoslib:
        lines += [
            "\n",
            "# chaoslib holds the book's shared numerics. Installed from the public\n",
            "# repository, since a downloaded notebook has no checkout to import from.\n",
            f"%pip install -q {CHAOSLIB_SPEC}\n",
        ]
    return lines


def build_header_cell(title: str, source_name: str, page_url: str | None) -> list[str]:
    """Provenance and the one behavioural difference worth warning about."""
    lines = [
        f"# {title}\n",
        "\n",
        "*A Jupyter conversion of an interactive [marimo](https://marimo.io) notebook.*\n",
        "\n",
    ]
    if page_url:
        lines += [f"Run it live, in the browser, at <{page_url}>.\n", "\n"]
    lines += [
        "**One thing behaves differently here.** In the live version the parameters are\n",
        "sliders you drag, and every figure redraws as you move them. Jupyter cannot drive\n",
        "marimo's widgets, so they render as *static* controls showing their default\n",
        "values. To change a parameter, edit the `value=` argument where the control is\n",
        "defined and re-run the cells below it.\n",
        "\n",
        "Cells are in top-down reading order, matching the published chapter. The numerics\n",
        "are unchanged.\n",
        "\n",
        f"Generated from `{source_name}` by `scripts/export_ipynb.py`.\n",
    ]
    return lines


def notebook_title(src: Path) -> str:
    """The chapter's own H1, so the download is titled like the published page.

    Falls back to a prettified filename. Reading the heading out of the source
    beats deriving it from the stem, which yields things like "Ch06 Lorenz63".
    """
    text = src.read_text()
    # Search only after the first `mo.md(`, so the PEP 723 header's "# /// script"
    # and ordinary Python comments cannot be mistaken for the title. The first
    # markdown H1 after that point is the chapter heading.
    start = text.find("mo.md(")
    if start != -1:
        for match in re.finditer(r"^\s*#\s+(\S.*?)\s*$", text[start:], re.M):
            heading = match.group(1).strip()
            if heading.startswith(("/", "-", "=", "#")):
                continue
            return heading
    return src.stem.replace("_", " ").replace("-", " ").title()


def chapter_url(src: Path, site_root: Path, base_url: str | None) -> str | None:
    """Locate the chapter's page in the Hugo content tree and build its URL.

    The part a chapter lives in is recorded only by its directory, so it is read
    from the filesystem rather than assumed -- chapter 20 is under part5 while
    chapters 4 and 6 are under part2, and hard-coding one base URL gets the
    others wrong.
    """
    if not base_url:
        return None
    matches = sorted(site_root.glob(f"content/part*/{src.stem}.md"))
    if not matches:
        return None
    part = matches[0].parent.name
    return f"{base_url.rstrip('/')}/{part}/{src.stem}/"


def convert(src: Path, dest: Path, page_url: str | None = None) -> None:
    """Export one marimo notebook to a downloadable .ipynb."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.ipynb"
        # --sort top-down keeps the chapter's reading order; the default
        # topological order rearranges cells and makes the prose jump around.
        subprocess.run(
            [
                sys.executable, "-m", "marimo", "export", "ipynb",
                "--sort", "top-down", str(src), "-o", str(raw),
            ],
            check=True,
            capture_output=True,
        )
        nb = json.loads(raw.read_text())

    deps = read_pep723_dependencies(src)
    needs_chaoslib = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        new_source, hit = strip_pyodide_bootstrap(source)
        if hit:
            cell["source"] = new_source.splitlines(keepends=True)
            needs_chaoslib = True

    title = notebook_title(src)
    nb["cells"] = [
        {"cell_type": "markdown", "id": "generated-header", "metadata": {},
         "source": build_header_cell(title, src.name, page_url)},
        {"cell_type": "code", "id": "generated-install", "execution_count": None,
         "metadata": {}, "outputs": [], "source": build_install_cell(deps, needs_chaoslib)},
    ] + nb["cells"]

    # nbformat >= 4.5 requires a cell id; marimo's export omits them, which makes
    # every downstream tool emit a MissingIDFieldWarning.
    for index, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"cell-{index:03d}")

    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"  {src.name} -> {dest}  ({len(nb['cells'])} cells)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument(
        "--base-url",
        help="Site root for the 'read it live' link, e.g. https://anichaospred.github.io",
    )
    parser.add_argument(
        "--site-root", type=Path, default=Path("site"),
        help="Hugo site directory, used to find which part each chapter lives in",
    )
    parser.add_argument(
        "--page-url",
        help="Explicit 'read it live' URL, used for every notebook. For sites whose "
             "notebooks all sit on one page, rather than one page per chapter.",
    )
    args = parser.parse_args()

    if args.output and len(args.sources) != 1:
        parser.error("-o takes exactly one source; use --outdir for several")
    if not args.output and not args.outdir:
        parser.error("give -o or --outdir")

    for src in args.sources:
        dest = args.output or (args.outdir / f"{src.stem}.ipynb")
        url = args.page_url or chapter_url(src, args.site_root, args.base_url)
        convert(src, dest, url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
