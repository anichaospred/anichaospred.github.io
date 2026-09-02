#!/usr/bin/env python3
"""Precompute chapter 11's two Lyapunov sweeps.

A full Lorenz 96 spectrum is the most expensive object in this book. The
Benettin algorithm carries :math:`N` tangent vectors alongside the trajectory
and re-orthonormalises at every step, so the cost is dominated not by the QR
decomposition but by evaluating the :math:`N \\times N` Jacobian
:math:`T/\\Delta t` times. At :math:`N = 40` and :math:`T = 200` that is 3.1 s
natively -- roughly 30 s under Pyodide, for one parameter value. Restricting to
the leading few exponents saves almost nothing (measured: 0.62 s against 0.83 s
at :math:`T = 50`), because the trajectory and its Jacobian are needed either
way.

So both sweeps are precomputed and embedded in the notebook, and the chapter's
live figures are the ones that are actually cheap: a Hovmoller diagram is a
plain integration (137 ms at :math:`N = 40` over 30 time units) and the spatial
diagnostics on top of it are FFTs.

**The forcing sweep** gives the full spectrum at each :math:`F`, from which the
notebook reads the leading exponent, the number of positive exponents, the
Kolmogorov-Sinai entropy and the Kaplan-Yorke dimension. The grid is dense
between 3 and 6 because that is where chaos sets in.

**The domain sweep** gives the full spectrum at each :math:`N` with
:math:`F = 8`, which is the chapter's headline: the spectrum is intensive while
the entropy and dimension are extensive.

Run from chaos-book/:
    python3 scripts/generate_ch11_data.py        # ~4 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import lyapunov, systems  # noqa: E402

DT = 0.01
T_FINAL = 300.0
T_TRANSIENT = 40.0

# Dense through the onset of chaos, sparse either side of it.
FORCINGS = (
    0.5, 0.8, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 4.75,
    5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0,
)

# Domain sizes at F = 8. The smallest that still supports the preferred
# wavenumber is around N = 10; below that the ring cannot hold the instability.
DOMAIN_SIZES = (12, 16, 20, 24, 30, 36, 40, 48, 60, 80)


def spectrum(n: int, forcing: float, t_final: float = T_FINAL) -> np.ndarray:
    state = systems.lorenz96_uniform_state(forcing, n)
    state[n // 2] += 0.01
    return lyapunov.lyapunov_spectrum(
        systems.lorenz96,
        systems.lorenz96_jacobian,
        state,
        dt=DT,
        t_final=t_final,
        t_transient=T_TRANSIENT,
        forcing=forcing,
    )


def _emit(name: str, values, fmt: str = ".4f") -> None:
    print(f"{name} = (" + ", ".join(format(v, fmt) for v in values) + ")")


def forcing_sweep() -> None:
    print(f"# --- forcing sweep, N=40, dt={DT}, T={T_FINAL} ---")
    lam1, npos, hks, dky, resid = [], [], [], [], []
    for forcing in FORCINGS:
        started = time.perf_counter()
        s = spectrum(40, forcing)
        lam1.append(float(s[0]))
        npos.append(int((s > 0.0).sum()))
        hks.append(float(lyapunov.ks_entropy(s)))
        dky.append(float(lyapunov.kaplan_yorke_dimension(s)))
        resid.append(abs(float(s.sum()) + 40.0))
        print(
            f"#   F={forcing:5.2f}  lam1={s[0]:+7.4f}  n_pos={npos[-1]:2d}  "
            f"h_KS={hks[-1]:7.3f}  D_KY={dky[-1]:6.2f}  "
            f"|sum+N|={resid[-1]:.1e}  ({time.perf_counter() - started:.1f}s)"
        )
    _emit("F_GRID", FORCINGS, "g")
    _emit("F_LAMBDA1", lam1)
    print("F_NPOS = (" + ", ".join(str(v) for v in npos) + ")")
    _emit("F_HKS", hks, ".3f")
    _emit("F_DKY", dky, ".2f")
    print(f"# max |sum(lambda) + N| over the sweep: {max(resid):.1e}")

    # The spectrum itself at the reference parameters, for the shape figure.
    reference = spectrum(40, 8.0)
    print("\n# The full spectrum at the reference parameters N=40, F=8.")
    _emit("SPECTRUM_N40_F8", reference, ".4f")


def domain_sweep() -> None:
    print(f"\n# --- domain sweep, F=8, dt={DT}, T={T_FINAL} ---")
    lam1, npos, hks, dky, shapes = [], [], [], [], {}
    for n in DOMAIN_SIZES:
        started = time.perf_counter()
        s = spectrum(n, 8.0)
        lam1.append(float(s[0]))
        npos.append(int((s > 0.0).sum()))
        hks.append(float(lyapunov.ks_entropy(s)))
        dky.append(float(lyapunov.kaplan_yorke_dimension(s)))
        if n in (16, 40, 80):
            shapes[n] = s
        print(
            f"#   N={n:3d}  lam1={s[0]:+7.4f}  n_pos={npos[-1]:2d}  "
            f"h_KS={hks[-1]:7.3f}  h_KS/N={hks[-1] / n:.4f}  "
            f"D_KY={dky[-1]:6.2f}  D_KY/N={dky[-1] / n:.4f}  "
            f"({time.perf_counter() - started:.1f}s)"
        )
    print("N_GRID = (" + ", ".join(str(v) for v in DOMAIN_SIZES) + ")")
    _emit("N_LAMBDA1", lam1)
    print("N_NPOS = (" + ", ".join(str(v) for v in npos) + ")")
    _emit("N_HKS", hks, ".3f")
    _emit("N_DKY", dky, ".2f")

    # Fitted densities -- the extensivity statement, as two numbers.
    for label, values in (("HKS", hks), ("DKY", dky), ("NPOS", npos)):
        slope, intercept = np.polyfit(DOMAIN_SIZES, values, 1)
        print(f"{label}_DENSITY = {slope:.5f}   # intercept {intercept:+.3f}")

    print("\n# Three spectra for the shape-collapse panel.")
    for n, s in shapes.items():
        _emit(f"SHAPE_N{n}", s, ".4f")


if __name__ == "__main__":
    forcing_sweep()
    domain_sweep()
