---
title: "What the notebooks depend on"
---

## In your browser

Nothing you have to install. Each notebook bundles:

| Component | Version | Role |
|---|---|---|
| [marimo](https://marimo.io) | 0.23.9 (pinned) | reactive notebook runtime and WASM export |
| [Pyodide](https://pyodide.org) | 0.27.7 | CPython 3.12 compiled to WebAssembly |
| NumPy, SciPy | as shipped by that Pyodide build | numerics |
| [Plotly](https://plotly.com/python/) | as shipped by that Pyodide build | interactive figures, including rotatable 3-D attractors |
| `chaoslib` | 0.1.0 | this book's own shared numerics, installed from a bundled wheel |

**The marimo pin is the load-bearing one.** marimo's version determines which Pyodide build
every export bundles, and that Pyodide ships its own NumPy and SciPy. So pinning marimo
pins *your* browser environment, not just the authors' — which is what makes a figure in
this book reproducible. The other runtime versions are deliberately left to float, because
requesting a specific NumPy that the bundled Pyodide does not have would break the notebook
with no error visible at build time.

## `chaoslib`

All the numerics live in one small library rather than being repeated across notebooks, so
that the equation printed in a chapter is provably the equation being stepped, and one test
suite covers all of it.

| Module | Provides |
|---|---|
| `systems` | right-hand sides and Jacobians: Lorenz 63 and 96, pendulum and double pendulum, logistic and Hénon maps |
| `integrate` | vectorised fixed-step RK4 for ensembles; adaptive integration for single trajectories |
| `lyapunov` | Lyapunov spectra, finite-time exponents, doubling times, Kaplan–Yorke dimension, KS entropy |
| `errorgrowth` | saturation and the logistic error-growth model |
| `dimension` | correlation dimension from a sampled trajectory |
| `information` | entropy, relative entropy, mutual information |
| `adjoint` | tangent linear and adjoint propagators; singular vectors |
| `assimilate` | Kalman filter, 3D-Var, 4D-Var, EnKF, localisation |
| `ensemble` | ensemble construction, spread–skill, rank histograms, CRPS, Brier score |
| `plotting` | the book's semantic colour palette and figure styling |

The library carries 90 tests, anchored wherever possible to identities that hold *exactly*
rather than approximately — the Lyapunov exponents summing to the trace of the Jacobian, the
adjoint identity, energy conservation, $\det\mathbf{M} = e^{\tau\,\mathrm{tr}\mathbf{J}}$ —
and otherwise to published values with the source named. Two real bugs were caught this way
during construction, and both would have produced plausible-looking figures.

## If you want to run it locally

```bash
git clone https://github.com/anichaospred/anichaospred.github.io
cd anichaospred.github.io/chaos-book
pip install -r requirements.txt
make test        # the chaoslib suite
marimo edit notebooks/ch06_lorenz63.py
```

`requirements.txt` pins the *authoring* environment exactly — that governs the test suite
and local editing, not what readers receive.
