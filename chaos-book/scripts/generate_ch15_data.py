#!/usr/bin/env python3
"""Precompute chapter 15's fixed figure: how far a tangent linear model can be trusted.

The validity-window panel sweeps the linear-vs-nonlinear discrepancy over a grid
of lead times and perturbation amplitudes, and then locates, for each amplitude,
the lead time at which the linear prediction first misses by 10%. That costs about
7 seconds natively -- roughly 40 in Pyodide -- and it has no knob: it is the same
picture for every reader.

The chapter's *interactive* figure is the validation curve, which does respond to
the tau and amplitude sliders and costs a fraction of a second.

Run from chaos-book/:
    python3 scripts/generate_ch15_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import adjoint, integrate, systems  # noqa: E402

DT = 0.005
X0 = np.array([1.0, 1.0, 20.0])
TAUS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
AMPS = (1e-6, 1e-4, 1e-2, 1e-1)
CROSS_AMPS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
THRESHOLD = 0.10


def relative_error(tau: float, amp: float, seed: int = 0) -> float:
    """How far the linear prediction misses the true nonlinear difference.

    Both are measured over the same interval from the same base state, with the
    same random perturbation direction, so the only difference is linearisation.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    n = max(2, int(round(tau / DT)) + 1)
    grid = np.linspace(0.0, tau, n)
    base = integrate.rk4(systems.lorenz63, X0, grid)[-1]
    nonlinear = integrate.rk4(systems.lorenz63, X0 + amp * direction, grid)[-1] - base
    propagator = adjoint.tangent_linear_propagator(
        systems.lorenz63, systems.lorenz63_jacobian, X0, tau, dt=DT
    )
    linear = propagator @ (amp * direction)
    scale = max(float(np.linalg.norm(nonlinear)), 1e-300)
    return float(np.linalg.norm(nonlinear - linear) / scale)


def main() -> int:
    grid = {}
    for amp in AMPS:
        grid[amp] = [relative_error(t, amp) for t in TAUS]
        print(f"  amp={amp:.0e} done", file=sys.stderr)

    crossings = {}
    for amp in CROSS_AMPS:
        found = float("nan")
        for tau in np.arange(1.0, 30.0, 0.5):
            if relative_error(float(tau), amp) > THRESHOLD:
                found = float(tau)
                break
        crossings[amp] = found
        print(f"  crossing amp={amp:.0e} -> tau={found}", file=sys.stderr)

    print("# Section 2, precomputed: relative error of the linear prediction over a")
    print("# grid of lead times and amplitudes, and the lead time at which it first")
    print(f"# exceeds {THRESHOLD:.0%}. Costs ~7 s natively (~40 s in Pyodide) and has no")
    print("# knob, so it is computed once by scripts/generate_ch15_data.py.")
    print(f"VALIDITY_TAUS = {TAUS!r}")
    print("VALIDITY_ERROR = {")
    for amp in AMPS:
        vals = ", ".join(f"{v:.4e}" for v in grid[amp])
        print(f"    {amp!r}: ({vals}),")
    print("}")
    print("VALIDITY_CROSSING = {")
    for amp in CROSS_AMPS:
        print(f"    {amp!r}: {crossings[amp]!r},")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
