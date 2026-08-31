# House rules for this repository

Read [`docs/authoring.md`](docs/authoring.md) before adding a chapter and
[`docs/architecture.md`](docs/architecture.md) before changing the build. This file is
the short version of what must not be got wrong.

## Non-negotiable

1. **`make test` must pass before every commit.** The suite tests numbers, not execution.
2. **New numerics go in `chaoslib` first, with correctness tests, before any notebook uses
   them.** Prefer identities that hold *exactly* — $\sum_i\lambda_i = \mathrm{tr}\mathbf{J}$,
   $\det\mathbf{M} = e^{\tau\,\mathrm{tr}\mathbf{J}}$, the adjoint identity, energy
   conservation — because they test the implementation rather than its convergence. This
   caught two real bugs during construction that plots would not have.
3. **Never invent a citation, a page number, or a literature value.** Write
   *[citation needed]* instead. Values asserted in tests must be traceable to a named
   source or to an identity derived in the docstring.
4. **NumPy, SciPy and Plotly only.** Pyodide has to run everything; a dependency it cannot
   install breaks every chapter at once.
5. **Verify physics standalone before writing prose.** A throwaway script that checks the
   number beats a figure that looks plausible.

## Build facts that are easy to break

- Chapters export as `-o site/static/nb/chNN_slug.html` — a **file**, not a directory. That
  form makes the parent directory the output directory, which is what gives one shared
  `assets/` and one shared `public/`. Switching to `-o …/chNN_slug/` silently multiplies the
  site by ~27 MB per chapter.
- `make clean` must run before a full export: marimo *merges* into an existing `assets/`
  rather than replacing it.
- Per-chapter heavy data goes in `site/static/data/`, never `notebooks/public/` — there is
  only one shared `public/` and everything in it is paid for once but shipped to every
  chapter.
- The PEP 723 header pins marimo exactly and leaves numpy/scipy/plotly floating. Do not
  "tidy" that into exact pins; see the *Reproducibility* section of the README.
- Marimo cell signatures are dependency declarations. Removing a name from one cell's
  `return` breaks every cell that lists it. Cell-local names must be underscore-prefixed.
- `marimo export -o /tmp/...` fails under the sandbox with `PermissionError`. Export into
  the repo or a scratch directory.
- `marimo export` drops a stray `CLAUDE.md` into the output directory; the Makefile removes
  it. Do not be alarmed by it, and do not commit it.

## Verifying a notebook change

`marimo export html --sandbox notebooks/chNN.py -o <scratch>.html` *executes every cell*.
`html-wasm` only bundles, and will happily produce a page whose cells all raise in the
reader's browser.

Check **the exit code and stderr**, not just `grep -c marimo-error` — a notebook whose
cells all raised still grepped to 0 while the command printed "some cells failed to
execute". For figures, count the rendered images and extract one to look at: "it
rendered" is not "it is right".

## Git workflow

Separate, explicitly-approved steps: implement → **commit** (only when asked) →
**push and open a PR** (only when asked, as a separate request) → **merge** (only when
asked, after checking CI). Always branch fresh off `origin/main`; one chapter per branch.
Do not commit, push, or merge unprompted.

## Style

Prose is publication-grade: precise, active, AGU/AMS register. Chapters are
weather/climate-first — open with a forecasting question and let the mathematics arrive as
the answer. Colours are semantic and come from `chaoslib.plotting`; do not choose colours
per figure.
