#!/usr/bin/env python3
"""Precompute chapter 16's fixed figure: optimal growth against the Lyapunov estimate.

Averaging the leading singular value over base points on the attractor is what
makes this curve readable at all: at a *single* base point sigma_1(tau) is not even
monotonic in tau, because a longer window can include a contracting stretch. The
average over 33 decorrelated points is monotonic and converges properly.

That sweep costs about 10 seconds natively -- a minute in Pyodide -- and has no
knob. The chapter's interactive figures use the propagator at the reader's chosen
tau, which costs milliseconds.

Run from chaos-book/:
    python3 scripts/generate_ch16_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import adjoint, integrate, systems  # noqa: E402

DT = 0.005
TAUS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
LAMBDA1 = 0.9056  # chapter 7, pinned in chaoslib's tests


def base_points(n_wanted: int = 33) -> np.ndarray:
    """Decorrelated points on the attractor."""
    grid = integrate.trajectory_grid(150.0, 0.01)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid)
    stride = max(1, (traj.shape[0] - 2000) // n_wanted)
    return traj[2000::stride][:n_wanted]


def main() -> int:
    states = base_points()
    geo, lo, hi = [], [], []
    for tau in TAUS:
        sigmas = np.array([
            float(adjoint.singular_vectors(
                adjoint.tangent_linear_propagator(
                    systems.lorenz63, systems.lorenz63_jacobian, x, tau, dt=DT
                ), 1
            )[0][0])
            for x in states
        ])
        # Geometric mean: these are multiplicative amplifications spanning orders
        # of magnitude, and an arithmetic mean would be dominated by one outlier.
        geo.append(float(np.exp(np.mean(np.log(sigmas)))))
        lo.append(float(sigmas.min()))
        hi.append(float(sigmas.max()))
        print(f"  tau={tau:5.2f} geo-mean sigma_1={geo[-1]:10.3f} "
              f"range [{lo[-1]:.2f}, {hi[-1]:.2f}]", file=sys.stderr)

    print("# Section 2, precomputed: the leading singular value of the L63 propagator,")
    print(f"# geometric-mean over {len(states)} decorrelated base points on the attractor,")
    print("# with the range across those points. Averaging is what makes the curve")
    print("# monotonic -- at a single base point sigma_1(tau) is not. Costs ~10 s")
    print("# natively (~60 s in Pyodide) and has no knob, so it is computed once by")
    print("# scripts/generate_ch16_data.py.")
    print(f"AMP_TAUS = {TAUS!r}")
    print("AMP_GEOMEAN = (" + ", ".join(f"{v:.4f}" for v in geo) + ")")
    print("AMP_MIN = (" + ", ".join(f"{v:.4f}" for v in lo) + ")")
    print("AMP_MAX = (" + ", ".join(f"{v:.4f}" for v in hi) + ")")
    print(f"AMP_N_POINTS = {len(states)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
