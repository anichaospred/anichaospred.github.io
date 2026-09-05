#!/usr/bin/env python3
"""Precompute chapter 5's one expensive result: the intermittency scaling law.

Everything else in the chapter runs live. That is worth stating explicitly,
because it is unusual for this book and it is not obvious from the physics:

* the cobweb, the bifurcation diagram and the Lyapunov curve are all vectorised
  over the parameter axis, so a thousand values of :math:`r` cost the same as
  one -- 4 ms and 32 ms respectively for a 1400-point sweep;
* even the full superstable cascade is cheap. Locating :math:`R_8` needs 256 map
  compositions per function evaluation, but a function evaluation is 256
  multiplications, and the bracketing scan converges in one pass. All three map
  families together take 57 ms, so Feigenbaum universality is computed in front
  of the reader rather than quoted at them.

**The exception is intermittency.** Laminar-phase detection is irreducibly
serial: whether iterate :math:`n` continues a laminar run depends on the run
length so far, so the parameter axis cannot be vectorised the way the others
can. Six parameters times 400,000 scalar iterations is about 1.2 s natively and
of order 12 s under Pyodide, for a curve that is identical for every reader.

Long runs are not a luxury here. Approaching :math:`r_c`, laminar phases grow
(mean 12 to 67 iterations across the range sampled) while their *number* falls,
so the closest parameter yields the fewest samples of the longest phases --
exactly where the mean is hardest to pin down. 400,000 iterations keeps the
sample count above 4,000 at every parameter.

Run from chaos-book/:
    python3 scripts/generate_ch05_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import maps, systems  # noqa: E402


def _scalar(name: str, value, fmt: str = ".6f") -> None:
    """Emit one float, spelling a non-finite one as `float("nan")`.

    A bare `nan` in the generated data is valid Python only where numpy is in
    scope under that name, which in a marimo data cell it is not -- so it
    becomes a NameError in the reader's browser. It is invisible to
    `grep marimo-error` and does not change the exporter's exit code; only
    stderr reports it. `tests/test_notebooks.py` now catches it in the notebook,
    and this catches it at the source.
    """
    value = float(value)
    if not np.isfinite(value):
        print(f'{name} = float("nan")')
    else:
        print(f"{name} = {format(value, fmt)}")

# Distances below r_c = 1 + 2 sqrt(2), spread over a 33-fold range so the
# power-law fit has leverage. The closest is limited by how often a 400,000-step
# run still samples a laminar phase of the relevant length.
EPSILONS = (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4, 6.0e-5)

N_ITER = 400_000


def main() -> None:
    r_c = maps.period_three_threshold()
    print(f"# r_c = 1 + 2*sqrt(2) = {r_c:.13f}")
    print(f"# {N_ITER} iterations per parameter, laminar tolerance 0.02, period 3")
    means: list[float] = []
    counts: list[int] = []
    for eps in EPSILONS:
        lengths = maps.laminar_phases(
            systems.logistic_map, r_c - eps, n_iter=N_ITER, n_discard=5_000
        )
        means.append(float(lengths.mean()))
        counts.append(int(lengths.size))
        print(
            f"#   r_c - r = {eps:.2e}   {lengths.size:5d} phases   "
            f"mean = {lengths.mean():7.2f}   longest = {lengths.max():5.0f}"
        )

    slope, intercept = np.polyfit(np.log(EPSILONS), np.log(means), 1)
    print()
    print("LAMINAR_EPS = (" + ", ".join(f"{v:g}" for v in EPSILONS) + ")")
    print("LAMINAR_MEAN = (" + ", ".join(f"{v:.4f}" for v in means) + ")")
    print("LAMINAR_COUNT = (" + ", ".join(str(v) for v in counts) + ")")
    print("# LAMINAR_SLOPE: theory says -1/2")
    _scalar("LAMINAR_SLOPE", slope, ".4f")
    _scalar("LAMINAR_INTERCEPT", intercept, ".4f")
    print(f"LAMINAR_N_ITER = {N_ITER}")


if __name__ == "__main__":
    main()
