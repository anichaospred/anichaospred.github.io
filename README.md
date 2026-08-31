# An Interactive Chaos and Predictability Textbook

Source for **<https://anichaospred.github.io>** — a browser-runnable textbook on chaos
and predictability in weather and climate, built from [marimo](https://marimo.io)
notebooks exported to WebAssembly and served as a [Hugo](https://gohugo.io) site.

See [`chaos-book/PLAN.md`](chaos-book/PLAN.md) for the full outline and the order
chapters are being written in.

Sibling project: the [interactive GFD textbook](https://anigfd.github.io).

## Layout

```
chaos-book/
  PLAN.md          the book's spine: parts, chapters, and the knob for each
  NOTATION.md      mandatory symbols and sign conventions (mounted as the site's /notation/)
  chaoslib/        shared Pyodide-safe numerics -- all chapters import from here
  notebooks/       one marimo notebook per chapter, plain .py
  tests/           correctness tests for chaoslib (90, anchored to exact identities)
  site/            Hugo: layouts, content, static
  docs/            architecture, authoring guide, chaoslib reference
  scripts/         export post-processing
```

Build products — `site/static/nb/`, `site/public/`, the `chaoslib` wheel — are
git-ignored and rebuilt by CI on every push.

## Quick start

```bash
cd chaos-book
pip install -r requirements.txt

make test                      # chaoslib correctness suite -- must pass
make notebooks                 # export every chapter to WASM (clean re-export)
make nb-one NB=ch06_lorenz63   # export just one chapter, while iterating
make serve                     # hugo server, with the notebooks embedded
```

Editing a notebook directly:

```bash
marimo edit chaos-book/notebooks/ch06_lorenz63.py
```

## How it works, in one paragraph

`chaoslib` is packaged as a wheel into `notebooks/public/`, which marimo copies into the
export directory. **Every chapter exports into the same directory**, so they share one
`assets/` folder and one `public/` holding that wheel — about 30 MB for the whole book,
against roughly 27 MB *per chapter* if each were self-contained. Each notebook's import
cell branches on `sys.platform == "emscripten"`: in the browser it `micropip`-installs
the bundled wheel, locally it imports from the checkout. Hugo copies the exports verbatim
and each chapter page embeds its notebook behind a button, so Pyodide does not boot until
the reader asks. Equations are rendered to MathML at build time, so no maths JavaScript
ships and a malformed equation fails the build.

Full detail in [`chaos-book/docs/architecture.md`](chaos-book/docs/architecture.md);
the checklist for adding a chapter is in
[`chaos-book/docs/authoring.md`](chaos-book/docs/authoring.md).

## Status

Two chapters are live (4, the pendulum; 6, Lorenz 63); the other 29 are stubs carrying an
abstract and the planned knob, so the book's shape is visible and no link is dead.
`chaoslib` has 10 modules and 90 passing tests. `PLAN.md` §5 lists the next chapters in
priority order.

## Reproducibility

`requirements.txt` pins the authoring environment; each notebook's PEP 723 header pins
**marimo exactly** and lets NumPy/SciPy/Plotly float. That is deliberate: marimo's version
determines which Pyodide build every export bundles, and that Pyodide ships its own NumPy
and SciPy. Pinning marimo is therefore what makes the *reader's* browser environment
reproducible — and pinning the others would ask micropip for builds the bundled Pyodide
does not have.
