---
title: "How these notebooks work"
---

## Nothing to install

Click **Load the interactive notebook** on any chapter page and Python starts running
inside your browser. There is no server doing the computing: the page downloads a
WebAssembly build of CPython ([Pyodide](https://pyodide.org)) together with NumPy, SciPy
and Plotly, and every number you see is computed on your own machine.

The first load fetches roughly 10 MB and takes 10–40 seconds depending on your connection.
That is why the notebooks sit behind a button rather than loading automatically — otherwise
every chapter page would take half a minute to become usable. Once loaded, moving a slider
is instant.

## What you can and cannot do

The notebooks are exported in **run** mode: the controls are live and the code is visible
but not editable. You can explore every parameter the chapter exposes, but you cannot
rewrite a cell in place.

To edit and extend them, clone the repository and run marimo locally:

```bash
git clone https://github.com/anichaospred/anichaospred.github.io
cd anichaospred.github.io/chaos-book
pip install -r requirements.txt
marimo edit notebooks/ch06_lorenz63.py
```

Locally the notebooks import `chaoslib` straight from the checkout; in the browser they
`micropip`-install it from a wheel bundled with the export. The import line is identical
either way.

## If a notebook fails to load

Two things to try, in order:

1. **Reload the page.** Each deploy rebuilds the JavaScript bundles with new content-hashed
   filenames, so a page cached from an earlier deploy can ask for a file that no longer
   exists. The pages carry a handler that reloads once automatically when this happens, but
   a manual reload does the same job.
2. **Check that WebAssembly is enabled.** Some managed or hardened browser configurations
   disable it. The notebooks cannot run without it, and there is no server-side fallback.

Notebooks are memory-hungry by browser standards. If a tab becomes unresponsive, close the
other notebook tabs — each one holds its own Python runtime.

## Performance, and why the models are small

Pyodide is single-threaded and roughly 3–10× slower than native CPython. Every model in the
book is chosen to run comfortably inside that budget: three variables, or forty, not a
million. That is a genuine constraint, but it is also the pedagogical point — the whole
argument of [the hierarchy-of-models chapter]({{< relref "part1/ch03_model-hierarchy.md" >}})
is that the small systems are where predictability is *understood*, and the large ones are
where it is *measured*.

Where a computation genuinely cannot run live, the chapter ships precomputed fields and the
widget scrubs through them. Chapters say when they do this.
