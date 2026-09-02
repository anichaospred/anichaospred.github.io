r"""Diagnostics for systems that have a *space*, not just a state.

Lorenz 63 has three variables and no geometry: there is no meaningful question
about the spatial scale of an error in it. Lorenz 96 has forty sites on a ring,
and that changes the available questions. An error now has a **wavelength**, a
structure can **propagate**, and energy can move between scales. This module
holds the diagnostics that answer those questions, and they are as much use in
chapter 12's scale-dependent error growth as in chapter 11's description of the
attractor.

Conventions. A *field* is an array whose **last axis is space** and whose
leading axis (if present) is time -- the same convention as
:func:`chaoslib.systems.lorenz96`, so a trajectory from
:func:`chaoslib.integrate.rk4` can be passed straight in. Space is periodic and
uniformly spaced, with :math:`N` sites indexed :math:`k = 0 \ldots N-1`;
wavenumber :math:`m` means :math:`m` full waves around the ring, so
:math:`\theta_m = 2\pi m/N` radians per site. Speeds are in **sites per unit
time**, and their sign follows :func:`chaoslib.systems.lorenz96_dispersion`:
positive means propagation toward increasing :math:`k`.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

__all__ = [
    "spatial_power_spectrum",
    "dominant_wavenumber",
    "phase_speed",
    "spatial_correlation",
    "correlation_length",
]


def spatial_power_spectrum(
    field: Array, remove_mean: bool = True
) -> tuple[Array, Array]:
    r"""Time-averaged power spectrum in space.

    Returns ``(wavenumbers, power)`` where ``wavenumbers`` runs
    :math:`0 \ldots \lfloor N/2 \rfloor` and ``power`` is
    :math:`\langle |\hat x_m|^2 \rangle` averaged over the leading (time) axis.

    ``remove_mean`` subtracts the space-time mean first, which is almost always
    what is wanted: in Lorenz 96 the mean state is close to the forcing
    :math:`F`, so :math:`m = 0` otherwise carries far more power than every
    other wavenumber combined and flattens the rest of the spectrum into
    invisibility.

    The average is over *power*, not over the complex coefficients. Averaging
    the coefficients would give nearly zero for a propagating wave whose phase
    decorrelates, which is a statement about the phase and not about the energy.
    """
    x = np.asarray(field, dtype=float)
    if remove_mean:
        x = x - x.mean()
    n = x.shape[-1]
    coefficients = np.fft.rfft(x, axis=-1)
    power = np.abs(coefficients) ** 2
    if power.ndim > 1:
        power = power.reshape(-1, power.shape[-1]).mean(axis=0)
    return np.arange(power.size), power * (2.0 / n**2)


def dominant_wavenumber(field: Array) -> int:
    r"""The wavenumber carrying the most power, excluding :math:`m = 0`.

    For Lorenz 96 at :math:`N = 40, F = 8` this is the sense in which the
    system has a characteristic wavelength -- there is one, and it is close to
    but not identical with the wavenumber that linear theory selects at the
    onset of instability (see
    :func:`chaoslib.systems.lorenz96_critical_forcing`).
    """
    m, power = spatial_power_spectrum(field)
    return int(m[1:][np.argmax(power[1:])])


def phase_speed(field: Array, wavenumber: int, dt: float) -> float:
    r"""Phase speed of one Fourier component, in sites per unit time.

    The complex coefficient :math:`\hat x_m(t)` is extracted at every time,
    its phase unwrapped, and :math:`\omega = d\phi/dt` fitted by least squares.
    The phase speed follows from :math:`\theta_m = 2\pi m/N`:

    .. math:: c = -\omega/\theta_m ,

    negative meaning propagation toward decreasing :math:`k`. The sign
    convention matches :func:`chaoslib.systems.lorenz96_dispersion`: a mode
    :math:`\exp[i(\theta k + \omega t)]` holds constant phase along
    :math:`k = -\omega t/\theta`.

    **Unwrapping is the fragile step.** It assumes the phase advances by less
    than :math:`\pi` between samples, so ``dt`` must resolve the oscillation:
    with :math:`\omega \approx 3` rad per time unit, ``dt = 0.01`` gives 0.03
    rad per step and is ample, while ``dt = 1`` would alias badly and return a
    confidently wrong answer. There is no way to detect this from the result,
    so check that ``dt`` is small against :math:`2\pi/\omega`.

    A propagating structure in a chaotic field also has a finite lifetime, and
    the fitted :math:`\omega` is an average over many structures rather than a
    property of one. Expect it to differ from the linear-theory value: at
    :math:`F = 8` the nonlinear waves travel roughly three times slower than
    the growth-rate calculation about the uniform state suggests.
    """
    x = np.asarray(field, dtype=float)
    if x.ndim != 2:
        raise ValueError("phase_speed needs a (time, space) field")
    m = int(wavenumber)
    if m <= 0:
        raise ValueError("wavenumber must be positive; m = 0 does not propagate")
    n = x.shape[-1]
    coefficients = np.fft.rfft(x - x.mean(), axis=-1)[:, m]
    phase = np.unwrap(np.angle(coefficients))
    t = np.arange(phase.size) * float(dt)
    omega = float(np.polyfit(t, phase, 1)[0])
    return -omega / (2.0 * np.pi * m / n)


def spatial_correlation(field: Array) -> Array:
    r"""Time-averaged autocorrelation as a function of separation in sites.

    Returns an array of length :math:`\lfloor N/2 \rfloor + 1`, normalised so
    that separation zero gives 1. Computed through the power spectrum
    (Wiener-Khinchin) rather than by explicit shifting, which makes it exact for
    a periodic domain rather than merely approximate near the wrap-around.
    """
    x = np.asarray(field, dtype=float)
    x = x - x.mean()
    n = x.shape[-1]
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    if power.ndim > 1:
        power = power.reshape(-1, power.shape[-1]).mean(axis=0)
    correlation = np.fft.irfft(power, n=n)
    return correlation[: n // 2 + 1] / correlation[0]


def correlation_length(field: Array) -> float:
    r"""Separation at which the spatial autocorrelation first crosses zero.

    A blunt but robust measure of the width of a structure, interpolated
    linearly between the two bracketing sites. Reported in sites. Returns
    ``nan`` if the correlation never crosses zero, which for a field with a
    dominant wavenumber it always does.

    Preferred here over an e-folding scale because it needs no assumption that
    the correlation decays exponentially -- in Lorenz 96 it does not, it
    oscillates with the dominant wavelength.
    """
    c = spatial_correlation(field)
    below = np.nonzero(c < 0.0)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    c0, c1 = c[i - 1], c[i]
    return float((i - 1) + c0 / (c0 - c1))
