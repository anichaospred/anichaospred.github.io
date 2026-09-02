"""chaoslib -- shared primitives for the Interactive Chaos and Predictability textbook.

Every chapter notebook imports from here rather than re-implementing numerics, so
the equation printed in the text is provably the equation being stepped, and one
test suite covers all of it.

Example
-------
>>> import numpy as np
>>> from chaoslib import systems, integrate, lyapunov
>>> t = integrate.trajectory_grid(t_final=40.0, dt=0.01)
>>> traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), t)
>>> spectrum = lyapunov.lyapunov_spectrum(
...     systems.lorenz63, systems.lorenz63_jacobian, traj[-1], t_final=200.0
... )

Modules
-------
systems      right-hand sides and Jacobians of the low-order models
integrate    fixed-step RK4 (ensembles) and an adaptive solver (single runs)
lyapunov     Lyapunov spectra, finite-time exponents, doubling times, dimension
errorgrowth  saturation and the logistic error-growth model
dimension    fractal dimension estimated from a sampled trajectory
information  entropy, relative entropy and mutual information as predictability
maps         bifurcation cascades, Feigenbaum universality, map Lyapunov exponents
spatial      spectra, phase speeds and correlation lengths for fields on a ring
adjoint      tangent-linear and adjoint propagators; singular vectors
assimilate   3D-Var, 4D-Var, Kalman filter and EnKF on the low-order models
ensemble     ensemble construction and probabilistic verification scores
plotting     the book's semantic colour palette and Plotly styling helpers

Pyodide note
------------
Everything here runs in the browser under Pyodide: NumPy, SciPy and Plotly only,
no compiled extensions of our own, and no threading. Keep it that way -- a
dependency that Pyodide cannot install breaks every chapter at once.
"""

from importlib import metadata as _metadata

from chaoslib import (
    adjoint,
    assimilate,
    dimension,
    ensemble,
    errorgrowth,
    information,
    integrate,
    lyapunov,
    maps,
    plotting,
    spatial,
    systems,
)

try:
    __version__ = _metadata.version("chaoslib")
except _metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "0.1.0"

__all__ = [
    "adjoint",
    "assimilate",
    "dimension",
    "ensemble",
    "errorgrowth",
    "information",
    "integrate",
    "lyapunov",
    "maps",
    "plotting",
    "spatial",
    "systems",
    "__version__",
]
