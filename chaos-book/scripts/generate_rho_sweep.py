#!/usr/bin/env python3
"""Precompute chapter 7's two fixed figures: lambda_1(rho), and transient chaos.

Chapter 7 plots lambda_1(rho) across the transition to chaos. Computing it live
is not affordable: each rho needs its own Benettin run, and a usable sweep costs
about 80 seconds in Pyodide -- far past the chapter's budget. It is also the same
curve for every reader, so there is nothing to gain from recomputing it.

So it is computed here, once, at higher accuracy than a live run could afford,
and pasted into the notebook as a literal array. The chapter then computes the
*full spectrum* live at the reader's chosen rho (about 10 seconds) and marks that
point on this curve -- the global picture for free, the local detail live.

The transient-chaos panel (Section 5) is the same story: three rho values times
five integration lengths plus three long trajectory runs costs about 46 seconds
natively, i.e. more than four minutes in Pyodide, and it has no knob at all.

Run from chaos-book/:
    python3 scripts/generate_rho_sweep.py             # print the rho-sweep literal
    python3 scripts/generate_rho_sweep.py --transient # print the transient literal
    python3 scripts/generate_rho_sweep.py --check     # re-verify a few points
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import lyapunov, systems  # noqa: E402

RHO_MIN, RHO_MAX, N_RHO = 0.5, 120.0, 141
T_FINAL, T_TRANSIENT, DT = 200.0, 30.0, 0.01
X0 = np.array([1.0, 1.0, 20.0])


def sweep() -> tuple[np.ndarray, np.ndarray]:
    rhos = np.linspace(RHO_MIN, RHO_MAX, N_RHO)
    lam = np.empty_like(rhos)
    for i, rho in enumerate(rhos):
        lam[i] = lyapunov.lyapunov_spectrum(
            systems.lorenz63,
            systems.lorenz63_jacobian,
            X0,
            dt=DT,
            t_final=T_FINAL,
            t_transient=T_TRANSIENT,
            n_exponents=1,
            rho=float(rho),
        )[0]
        print(f"  rho={rho:7.2f}  lambda_1={lam[i]:+.4f}", file=sys.stderr)
    return rhos, lam


TRANSIENT_RHOS = (22.7, 23.6, 28.0)
TRANSIENT_LENGTHS = (50.0, 100.0, 200.0, 400.0, 800.0)
SETTLE_T = 900.0          # long run used to ask "has it stopped moving?"
SETTLE_TAIL = 0.1         # fraction of the run inspected at the end


def transient_table() -> tuple[dict[float, list[float]], dict[float, float]]:
    """lambda_1 against integration length, plus whether the run settles.

    The second measurement is the decisive one and costs one line: the range of
    the state over the tail of a long run is zero if and only if the trajectory
    has landed on a fixed point.
    """
    from chaoslib import integrate

    lam: dict[float, list[float]] = {}
    settled: dict[float, float] = {}
    for rho in TRANSIENT_RHOS:
        lam[rho] = []
        for t_final in TRANSIENT_LENGTHS:
            lam[rho].append(float(lyapunov.lyapunov_spectrum(
                systems.lorenz63, systems.lorenz63_jacobian, X0, dt=DT,
                t_final=t_final, t_transient=30.0, n_exponents=1, rho=rho,
            )[0]))
            print(f"  rho={rho:6.2f} T={t_final:6.0f} lambda_1="
                  f"{lam[rho][-1]:+.4f}", file=sys.stderr)
        grid = integrate.trajectory_grid(SETTLE_T, DT)
        traj = integrate.rk4(systems.lorenz63, X0, grid, rho=rho)
        tail = traj[-int(len(traj) * SETTLE_TAIL):]
        settled[rho] = float(np.ptp(tail, axis=0).max())
        print(f"  rho={rho:6.2f} settled-range={settled[rho]:.4g}", file=sys.stderr)
    return lam, settled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="recompute a handful of points and report the spread")
    parser.add_argument("--transient", action="store_true",
                        help="print the Section 5 transient-chaos literal")
    args = parser.parse_args()

    if args.transient:
        lam, settled = transient_table()
        print("# Section 5, precomputed: lambda_1 against integration length at three")
        print("# rho values, and whether a 900-MTU run has stopped moving. Costs ~46 s")
        print("# natively (>4 min in Pyodide) and has no knob, so it is computed once by")
        print("# scripts/generate_rho_sweep.py --transient.")
        print(f"TRANSIENT_LENGTHS = {TRANSIENT_LENGTHS!r}")
        print("TRANSIENT_LAMBDA1 = {")
        for rho in TRANSIENT_RHOS:
            vals = ", ".join(f"{v:+.4f}" for v in lam[rho])
            print(f"    {rho}: ({vals}),")
        print("}")
        print("TRANSIENT_SETTLED_RANGE = {")
        for rho in TRANSIENT_RHOS:
            print(f"    {rho}: {settled[rho]:.6g},")
        print("}")
        return 0

    if args.check:
        for rho in (0.5, 15.0, 28.0, 100.0):
            vals = [
                lyapunov.lyapunov_spectrum(
                    systems.lorenz63, systems.lorenz63_jacobian, X0, dt=DT,
                    t_final=T_FINAL, t_transient=t, n_exponents=1, rho=rho,
                )[0]
                for t in (30.0, 31.0, 32.0)
            ]
            print(f"rho={rho:6.1f}  lambda_1 = {np.mean(vals):+.4f} "
                  f"± {np.ptp(vals) / 2:.4f} (spread over 3 transients)")
        return 0

    rhos, lam = sweep()
    print(f"# lambda_1(rho) for Lorenz 63, sigma=10, beta=8/3.")
    print(f"# Generated by scripts/generate_rho_sweep.py: rho in "
          f"[{RHO_MIN}, {RHO_MAX}], {N_RHO} points, Benettin with "
          f"dt={DT}, T={T_FINAL:.0f}, transient={T_TRANSIENT:.0f}.")
    print("RHO_GRID = (")
    for i in range(0, len(rhos), 8):
        print("    " + " ".join(f"{v:g}," for v in rhos[i:i + 8]))
    print(")")
    print("LAMBDA1_GRID = (")
    for i in range(0, len(lam), 8):
        print("    " + " ".join(f"{v:.4f}," for v in lam[i:i + 8]))
    print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
