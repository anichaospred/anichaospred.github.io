r"""One-dimensional rotating shallow water: balance, gravity waves, and the CFL limit.

The smallest system that contains Richardson's problem. It has a slow, balanced
mode and a fast inertia-gravity mode, and the whole history of numerical weather
prediction turns on telling them apart:

.. math::
    \partial_t u + u\,\partial_x u - f v &= -g\,\partial_x h \\
    \partial_t v + u\,\partial_x v + f u &= 0 \\
    \partial_t h + \partial_x (h u) &= 0

with :math:`h` the **total** depth, so :math:`\int h\,\mathrm{d}x` is the mass.
Derivatives are spectral on a periodic domain, which makes the linear dispersion
relation exact to round-off rather than to the order of a finite-difference
stencil -- the tests use that.

Three properties this module exists to expose, all of them measured in
chapter 2:

* the exact dispersion relation :math:`\omega^2 = f^2 + gHk^2`, so the fast mode
  has a floor at :math:`f` and a period of hours at synoptic wavelengths;
* **geostrophic balance is an exact steady solution here**: with :math:`u \equiv 0`
  and :math:`f v = g\,\partial_x h`, every tendency vanishes *identically*, not
  asymptotically. That is one dimension being kind -- in two dimensions, or with
  :math:`f` varying and advection able to feed back, balance is only asymptotic
  and a real initialisation scheme has a residual. The chapter says so;
* the stable timestep is set by :math:`\sqrt{gH}` and barely notices the wind,
  which is why filtering the fast mode out was worth a factor of ten in 1950.

Units are SI throughout: metres, seconds, metres per second. The free surface
:math:`h` plays the part of surface pressure, and
:func:`surface_pressure_change` fixes the convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "shallow_water_grid",
    "shallow_water_1d",
    "shallow_water_state",
    "shallow_water_split",
    "geostrophic_balance",
    "gravity_wave_speed",
    "inertia_gravity_frequency",
    "rossby_radius",
    "shallow_water_mass",
    "shallow_water_energy",
    "surface_pressure_change",
    "cfl_timestep",
]


def shallow_water_grid(n: int = 256, length: float = 1.0e7) -> dict:
    r"""Wavenumbers and spacing for a periodic domain of ``n`` points.

    Returns ``x``, ``k`` (the ``rfft`` wavenumbers), ``dx``, ``n`` and
    ``length``. Built once and reused: rebuilding it inside a time loop is the
    usual way to make a spectral solver needlessly slow.

    The default is 256 points on 10,000 km, so :math:`\Delta x \approx 39` km.
    """
    n = int(n)
    dx = float(length) / n
    return {
        "n": n,
        "length": float(length),
        "dx": dx,
        "x": np.arange(n, dtype=float) * dx,
        "k": 2.0 * np.pi * np.fft.rfftfreq(n, d=dx),
    }


def _ddx(field: Array, grid: dict) -> Array:
    """Spectral :math:`\\partial_x` along the last axis."""
    return np.fft.irfft(
        1j * grid["k"] * np.fft.rfft(field, axis=-1), n=grid["n"], axis=-1
    )


def shallow_water_1d(
    t: float,
    state: Array,
    grid: dict,
    coriolis: float = 1.0e-4,
    gravity: float = 9.81,
) -> Array:
    r"""Right-hand side of the 1-D rotating shallow-water equations.

    The state is ``concatenate([u, v, h])``, each of length ``grid["n"]``, with
    :math:`h` the total depth. Leading axes are ensemble members, as elsewhere.

    Pass ``grid`` through :func:`chaoslib.integrate.rk4`'s keyword arguments::

        run = integrate.rk4(shallow_water_1d, state, times, grid=grid)

    Note which term does what. The :math:`h` equation is a flux divergence, so
    **the height tendency responds only to the divergent part of the flow**: a
    state with :math:`u \equiv 0` has :math:`\partial_t h = 0` however badly
    unbalanced its :math:`v` is. Chapter 2 measures that, because "the pressure
    tendency is small" is therefore not the same statement as "the state is
    balanced", and conflating the two is a live error and not only a historical
    one.
    """
    state = np.asarray(state, dtype=float)
    n = grid["n"]
    u = state[..., 0:n]
    v = state[..., n : 2 * n]
    h = state[..., 2 * n : 3 * n]
    return np.concatenate(
        [
            -u * _ddx(u, grid) + coriolis * v - gravity * _ddx(h, grid),
            -u * _ddx(v, grid) - coriolis * u,
            -_ddx(h * u, grid),
        ],
        axis=-1,
    )


def shallow_water_state(u: Array, v: Array, h: Array) -> Array:
    """Assemble ``concatenate([u, v, h])``."""
    return np.concatenate(
        [
            np.asarray(u, dtype=float),
            np.asarray(v, dtype=float),
            np.asarray(h, dtype=float),
        ],
        axis=-1,
    )


def shallow_water_split(state: Array, grid: dict) -> tuple[Array, Array, Array]:
    """Split a state into ``(u, v, h)``; works on a trajectory too."""
    arr = np.asarray(state, dtype=float)
    n = grid["n"]
    return arr[..., 0:n], arr[..., n : 2 * n], arr[..., 2 * n : 3 * n]


def geostrophic_balance(
    height: Array,
    grid: dict,
    coriolis: float = 1.0e-4,
    gravity: float = 9.81,
) -> Array:
    r"""The balanced state carrying a given height field: :math:`u = 0`,
    :math:`v = (g/f)\,\partial_x h`.

    **This is an exact steady solution**, and the algebra is worth seeing
    because it is one line: with :math:`u \equiv 0` the :math:`u` equation reads
    :math:`fv - g\partial_x h`, which the balance kills identically; the
    :math:`v` equation reads :math:`-u\partial_x v - fu = 0`; and the :math:`h`
    equation reads :math:`-\partial_x(hu) = 0`. Nothing was linearised and no
    amplitude was assumed small, so the tendency is zero to round-off for any
    height field -- which the tests assert at :math:`10^{-15}` relative.

    That exactness is a property of one dimension on an :math:`f`-plane, not of
    balance in general. It makes the demonstration in chapter 2 clean and makes
    it *easier* than the real problem, where nonlinear normal-mode initialisation
    and digital filtering exist precisely because no exact balanced manifold is
    available.
    """
    return shallow_water_state(
        np.zeros_like(np.asarray(height, dtype=float)),
        (gravity / coriolis) * _ddx(np.asarray(height, dtype=float), grid),
        height,
    )


def gravity_wave_speed(depth: float = 8000.0, gravity: float = 9.81) -> float:
    r"""External gravity-wave speed :math:`c = \sqrt{gH}`.

    At :math:`H = 8000` m this is 280 m/s, the number that made Richardson's
    arithmetic hopeless and that the filtered equations of 1950 removed.
    """
    return float(np.sqrt(float(gravity) * float(depth)))


def inertia_gravity_frequency(
    wavenumber: Array,
    depth: float = 8000.0,
    coriolis: float = 1.0e-4,
    gravity: float = 9.81,
) -> Array:
    r"""Exact dispersion relation of the linearised system,

    .. math:: \omega^2 = f^2 + gH k^2 .

    Two features carry the chapter. There is a **floor**: no inertia-gravity
    wave is slower than :math:`f`, a period of about 17 hours at mid-latitudes,
    so the fast mode is never slow. And :math:`\omega` rises without bound with
    :math:`k`, so refining the grid makes the stiffness worse rather than
    better.

    Measured against the model to five significant figures (chapter 2,
    section 2).
    """
    k = np.asarray(wavenumber, dtype=float)
    return np.sqrt(float(coriolis) ** 2 + float(gravity) * float(depth) * k**2)


def rossby_radius(
    depth: float = 8000.0, coriolis: float = 1.0e-4, gravity: float = 9.81
) -> float:
    r"""Rossby radius of deformation :math:`L_R = \sqrt{gH}/f`.

    The scale at which rotation and stratification balance, and so the scale
    above which geostrophic balance is a good description. 2800 km for the
    external mode at mid-latitudes -- comparable with a synoptic system, which
    is why the balanced and unbalanced parts of an analysis are not cleanly
    separated by scale.
    """
    return gravity_wave_speed(depth, gravity) / float(coriolis)


def shallow_water_mass(state: Array, grid: dict) -> Array:
    r"""Total mass :math:`\int h\,\mathrm{d}x`, which the flux form conserves
    **exactly**.

    A spectral derivative of any periodic field has zero mean by construction,
    so :math:`\partial_t \int h = -\int \partial_x(hu) = 0` to round-off at
    every step and not merely on average. A test asserts it at
    :math:`10^{-12}` relative over a six-hour run, which makes it a sharp check
    that the flux form has not been quietly rewritten as
    :math:`-u\partial_x h - h\partial_x u`.
    """
    _, _, h = shallow_water_split(state, grid)
    return h.sum(axis=-1) * grid["dx"]


def shallow_water_energy(
    state: Array, grid: dict, gravity: float = 9.81
) -> Array:
    r"""Total energy :math:`\int \left(\tfrac12 h(u^2+v^2)
    + \tfrac12 g h^2\right)\mathrm{d}x`.

    Conserved by the continuous equations; here it drifts at the integrator's
    order, so it diagnoses the timestep rather than the physics.
    """
    u, v, h = shallow_water_split(state, grid)
    return (
        0.5 * h * (u**2 + v**2) + 0.5 * float(gravity) * h**2
    ).sum(axis=-1) * grid["dx"]


def surface_pressure_change(
    height_change: Array, depth: float = 8000.0, mean_pressure: float = 1000.0
) -> Array:
    r"""Convert a free-surface change to a surface-pressure change, in hPa.

    In a shallow-water model the free surface *is* the surface pressure, up to
    :math:`\rho g`. The convention here is that the mean depth carries the mean
    surface pressure, so :math:`\Delta p = p_0\,\Delta h / H` -- one metre of
    water is 0.125 hPa at :math:`H = 8000` m.

    This is a **scaling convention, not a physical conversion**: shallow water
    has one layer and the atmosphere does not. It exists so that the numbers in
    chapter 2 can be compared with the figure Richardson reported, and any
    conclusion that depends on the constant rather than on the ratio should be
    treated as a statement about this model.
    """
    return float(mean_pressure) * np.asarray(
        height_change, dtype=float
    ) / float(depth)


def cfl_timestep(
    grid: dict,
    depth: float = 8000.0,
    wind: float = 0.0,
    gravity: float = 9.81,
    courant: float = 0.87,
) -> float:
    r"""Largest stable timestep, :math:`\Delta t = C\,\Delta x/(c + U)`.

    The default ``courant`` is **measured, not theoretical**: for this spectral
    RK4 discretisation the threshold is :math:`C\,\Delta x/c` with
    :math:`C` between 0.84 and 0.91 across a factor of four in :math:`c`
    (an 8 % spread, mean 0.89; chapter 2, section 5). The default sits just
    below the mean so that it is a usable step rather than the marginal one.

    The wind enters weakly, which is the substantive point -- **the timestep is
    set by the wave and not by the weather**, so a model carrying the external
    gravity mode pays for a signal that carries no forecast information.

    Pass ``depth=0`` to get the filtered-model limit, set by the wind alone.
    """
    speed = gravity_wave_speed(depth, gravity) + abs(float(wind))
    if speed <= 0.0:
        return float("inf")
    return float(courant) * grid["dx"] / speed
