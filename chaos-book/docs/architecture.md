# Architecture — how a notebook becomes a browser-runnable chapter

What happens between editing `notebooks/chNN_slug.py` and a reader moving a slider on
the live site, and the constraints — almost all of them performance constraints —
that shape how chapters are written.

## The pipeline at a glance

```
chaoslib/*.py ──(pip wheel)──▶ notebooks/public/chaoslib-0.1.0-py3-none-any.whl
                                        │
notebooks/chNN_slug.py ──(marimo export html-wasm)──▶ site/static/nb/chNN_slug.html
                                        │              + ONE shared assets/
                                        │              + ONE shared public/
                                        │                       │
                                        │   Hugo shortcode {{< marimo src="/nb/chNN_slug.html" >}}
                                        ▼                       ▼
                              reader's browser boots Pyodide, micropip-installs
                              the bundled wheel, and runs the notebook locally
```

There is no server-side compute anywhere. The exported bundle is static files; every
number the reader sees is computed in their own browser by CPython compiled to
WebAssembly.

## Step by step

### 1. `chaoslib` is packaged as a wheel

`make wheel` (a dependency of `make notebooks`) runs
`pip wheel --no-deps -q -w notebooks/public .` whenever any `chaoslib/*.py` or
`pyproject.toml` changes. The wheel lands in `notebooks/public/` because marimo copies
a notebook's sibling `public/` folder into the export directory — that is how the
shared library travels to the browser.

### 2. Every notebook has a dual-mode import cell

```python
if sys.platform == "emscripten":          # Pyodide, in the browser
    import micropip
    await micropip.install(
        str(mo.notebook_location() / "public" / "chaoslib-0.1.0-py3-none-any.whl")
    )
else:                                      # locally: marimo edit, pytest
    sys.path.insert(0, str(mo.notebook_dir().parent))

from chaoslib import integrate, plotting, systems
```

The import line the student reads never branches. Use `mo.notebook_location()` for
anything that must resolve in the browser; `mo.notebook_dir()` is the local-only
branch.

### 3. Export to WASM — all chapters into ONE directory

`make notebooks` runs, for every `notebooks/ch*.py`:

```bash
marimo export html-wasm --sandbox --mode run --no-show-code \
  notebooks/chNN_slug.py -o site/static/nb/chNN_slug.html
```

**The `-o …/chNN_slug.html` form is load-bearing.** In
`marimo/_cli/export/commands.py`, when `-o` ends in `.html` the *parent directory*
becomes `out_dir`, and `out_dir` is what receives both `export_assets()` and
`export_public_folder()`. So exporting every chapter into `site/static/nb/` produces
**one** `assets/` directory and **one** `public/` holding the wheel, shared by all
chapters, and `mo.notebook_location() / "public" / …` resolves to `/nb/public/…`.

The alternative form, `-o site/static/nb/chNN_slug/`, produces a *self-contained
directory per chapter*: ~27 MB and ~700 files each. At 31 chapters that is roughly
800 MB, against the ~30 MB the shared form costs. Verified empirically: one notebook
and two notebooks both produce 28 MB / 728 asset files.

Two consequences follow, and both are already handled in the `Makefile`:

- **`make clean` runs before a full export.** marimo merges into an existing `assets/`
  (`shutil.copytree(..., dirs_exist_ok=True)`) rather than replacing it, so chunks
  from an older marimo version would accumulate indefinitely.
- **Per-chapter heavy data cannot use `public/`**, because there is now only one
  shared `public/`. Put precomputed fields in `site/static/data/chNN_slug.npz` and
  fetch them by URL.

`--mode run --no-show-code` gives a read-and-interact app: sliders live, code hidden.
Use `--mode edit` only for a chapter whose point is editing the code.

`--sandbox` builds the export environment from the notebook's PEP 723 header. That
header pins **marimo exactly** and leaves numpy/scipy/plotly floating — see
*Reproducibility* below.

### 4. Post-processing

Immediately after export, `make notebooks`:

1. removes the stray `CLAUDE.md` that `marimo export` drops into every output
   directory;
2. runs `scripts/patch_chunk_reload.py` over the exported HTML.

The second one matters more here than in a per-chapter layout. Every export produces
fresh content-hashed JS chunk names, and with a single shared `assets/` directory
*every* chapter's chunk names change on *every* deploy. A browser holding a stale
cached page then requests a chunk the new deploy no longer serves, and fails with
`Failed to fetch dynamically imported module`. The injected handler reloads once,
guarded by a `sessionStorage` flag so a genuinely broken deploy cannot reload-loop.

### 5. Hugo embeds the bundle

`site/static/nb/` sits under Hugo's `static/`, so it is copied verbatim into
`site/public/nb/`. Each chapter page embeds its notebook with one line:

```
{{< marimo src="/nb/chNN_slug.html" >}}
```

The shortcode renders a **button, not an iframe**. The iframe is created in JavaScript
on click, so Pyodide — about 10 MB, 10–40 s cold boot — never loads until the reader
asks for it. Without that, every chapter page would take half a minute to become
usable.

Equations are rendered to MathML **at build time** by `transform.ToMath` via the
passthrough render hook. No maths JavaScript ships, and a malformed equation fails
the build rather than rendering as raw TeX in the reader's browser.

### 6. Deploy

`.github/workflows/deploy.yml`, on every push to `main`: run the tests (a hard gate),
export every notebook, patch the exports, build Hugo, upload and deploy to GitHub
Pages. Build output is never committed — `site/static/nb/` and `site/public/` are
git-ignored.

Because the repo is `anichaospred/anichaospred.github.io`, Pages serves it at the
domain root, so `baseURL = "https://anichaospred.github.io/"` sits in `hugo.toml` and
needs no command-line override. Content still uses the
`strings.TrimPrefix "/" | relURL` idiom in the shortcode, so nothing breaks if the
book is ever served from a subpath.

## The performance budget

Pyodide is single-threaded and roughly 3–10× slower than native CPython. Every
notebook design decision traces back to that:

- **Keep trajectories and grids modest.** A few thousand steps of a 3-variable system
  is instant; a 40-variable Lyapunov spectrum over 500 time units is not.
- **Precompute outside the loop.** Build grids, masks and operators once.
- **Expose the speed/detail trade-off.** A resolution or ensemble-size slider is a
  standard control, not a cop-out.
- **Precompute genuinely heavy runs offline** and ship the fields for the widget to
  scrub through — in `site/static/data/`, not `notebooks/public/`.
- **Lazy-load.** Pyodide does not boot until the reader clicks Run.

## Dependencies: why SciPy and Plotly, and why they float

`chaoslib` depends on NumPy, **SciPy** and **Plotly**. This is a deliberate departure
from the sibling GFD textbook, which is pure NumPy + matplotlib:

- **SciPy** for `solve_ivp` (adaptive integration where tolerance matters),
  `scipy.special.ellipk` (the exact pendulum period), `scipy.optimize` (curve fitting
  and the 4D-Var minimisation).
- **Plotly** for rotatable 3-D attractors. Being able to grab the Lorenz attractor and
  turn it is the single most valuable interaction in this subject, and a static
  projection is not a substitute.

Both have Pyodide wheels, and both were already running in the browser in the two
notebooks this book was seeded from.

**Why the runtime deps float.** Each notebook's PEP 723 header pins `marimo` exactly
but leaves `numpy`, `scipy` and `plotly` unpinned. marimo's version determines which
Pyodide build the export bundles (`PYODIDE_VERSION` in
`marimo/_pyodide/pyodide_constraints.py`; `marimo==0.23.9` → Pyodide 0.27.7), and that
Pyodide ships its own numpy/scipy builds. Pinning exact versions for those in the
header would ask micropip to install a build the bundled Pyodide does not have, and
the chapter would fail to boot with no error visible at export time.
`requirements.txt` pins them exactly, but that governs the *authoring* environment
only — pytest and local `marimo edit`.

So: **pinning marimo is what makes the reader's environment reproducible.** It is the
one version that matters.

## Testing and CI

- `make test` runs `pytest -q tests/` — 90 correctness tests over `chaoslib`, anchored
  to analytic identities and published values. It must pass before every commit and it
  gates the deploy.
- `make notebooks` doubles as an integration test: a fully reactive notebook executes
  end-to-end during export, so a clean export is real evidence. Run-gated cells are
  only parse-checked.
- For faster feedback while writing a chapter,
  `marimo export html --sandbox notebooks/chNN.py -o <scratch>.html` *executes every
  cell* — grep the output for `marimo-error`. `html-wasm` only bundles; it will
  happily produce a page whose cells all raise.
