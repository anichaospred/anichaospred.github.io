r"""Two-dimensional turbulence, pseudospectral, and its energy spectrum.

Everything up to chapter 13 has been a system with a *characteristic scale*.
Lorenz 96's spectrum has a peak (chapter 11 measured it at wavenumber 9);
Lorenz 63 has no space at all. A turbulent flow has neither a characteristic
scale nor a finite number of active ones: it has a **range**, over which the
energy spectrum is a power law, and that changes the predictability question in
a way chapter 12 anticipated and this module lets one look at directly.

The equations are two-dimensional Navier-Stokes in vorticity form,

.. math::
    \frac{\partial \zeta}{\partial t} = -J(\psi, \zeta) + \nu\,\nabla^2\zeta,
    \qquad \nabla^2\psi = -\zeta,

solved on a doubly periodic square by transforming to Fourier space, evaluating
the advection term as products in physical space, and truncating the result
(the "pseudospectral" method). Aliasing from the quadratic term is removed by
the two-thirds rule, so only wavenumbers up to :math:`N/3` carry information.

**Why two dimensions and not three.** Not economy -- though a 3-D run at any
useful Reynolds number is far out of this book's budget -- but because the
large-scale atmosphere *is* quasi-two-dimensional, and because the two cases
give opposite answers about predictability. Two-dimensional flow conserves
enstrophy as well as energy, which forbids the forward energy cascade of three
dimensions and gives a steeper spectrum; and by the relation in
:func:`turnover_time`, a steeper spectrum means a *scale-independent* eddy
timescale, which by chapter 12's criterion means **no finite predictability
limit**. Chapter 14 works that through.

**What this module will not do.** It will not produce a convincing inertial
range. Chapter 14 measures the failure rather than describing it: at 64, 128 and
256 grid points the widest stretch of spectrum whose local slope stays within
0.4 of :math:`-3` is *zero octaves*, because a decaying flow at those
resolutions has its energy peak and its dissipation range adjacent with nothing
between. Published two-dimensional spectra use 1024 points and upward, forced
and run for many eddy turnover times. Treat what comes out of here as a fluid
that behaves qualitatively like turbulence -- vortex merger, filamentation, an
upscale error cascade -- and not as a source of spectral exponents.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]
Complex = NDArray[np.complexfloating]

__all__ = [
    "spectral_grid",
    "vorticity_tendency",
    "advance_vorticity",
    "energy",
    "enstrophy",
    "energy_spectrum",
    "local_spectral_slope",
    "turnover_time",
    "random_vorticity",
    "vorticity_field",
    "band_perturbation",
]


def spectral_grid(n: int, length: float = 2.0 * np.pi) -> dict:
    r"""Wavenumbers, the inverse Laplacian, and the dealiasing mask.

    Returns a dict with ``kx``, ``ky`` (broadcast to the half-spectrum shape of
    :func:`numpy.fft.rfft2`), ``k2``, ``inverse_k2`` (zero at the mean mode),
    ``mask`` (the two-thirds rule) and ``n``.

    Built once and reused: the wavenumber arrays and the mask are the same at
    every step, and rebuilding them inside a time loop is the classic way to
    make a spectral solver ten times slower than it needs to be.
    """
    n = int(n)
    scale = 2.0 * np.pi / float(length)
    kx = np.fft.fftfreq(n, d=1.0 / n) * scale
    ky = np.fft.rfftfreq(n, d=1.0 / n) * scale
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid**2 + ky_grid**2
    inverse = np.zeros_like(k2)
    np.divide(1.0, k2, out=inverse, where=k2 > 0.0)
    cutoff = n // 3
    mask = (np.abs(kx_grid) <= cutoff * scale) & (np.abs(ky_grid) <= cutoff * scale)
    return dict(
        kx=kx_grid, ky=ky_grid, k2=k2, inverse_k2=inverse, mask=mask, n=n
    )


def vorticity_tendency(
    vorticity_hat: Complex,
    grid: dict,
    viscosity: float = 0.0,
    hyper_order: int = 1,
) -> Complex:
    r"""Spectral tendency of :math:`\zeta`: advection plus dissipation.

    The Jacobian :math:`J(\psi,\zeta) = u\,\partial_x\zeta + v\,\partial_y\zeta`
    is formed in physical space and transformed back, then the two-thirds mask
    removes the aliased content. ``hyper_order`` raises the dissipation to
    :math:`\nu(k^2)^{p}`, which confines it to the smallest scales.
    """
    stream_hat = grid["inverse_k2"] * vorticity_hat
    u = np.fft.irfft2(1j * grid["ky"] * stream_hat)
    v = -np.fft.irfft2(1j * grid["kx"] * stream_hat)
    zeta_x = np.fft.irfft2(1j * grid["kx"] * vorticity_hat)
    zeta_y = np.fft.irfft2(1j * grid["ky"] * vorticity_hat)
    advection = np.fft.rfft2(u * zeta_x + v * zeta_y) * grid["mask"]
    dissipation = float(viscosity) * (grid["k2"] ** int(hyper_order)) * vorticity_hat
    return -advection - dissipation


def advance_vorticity(
    vorticity_hat: Complex,
    grid: dict,
    dt: float,
    n_steps: int,
    viscosity: float = 0.0,
    hyper_order: int = 1,
) -> Complex:
    """RK4 in spectral space for ``n_steps``, returning only the final state.

    The interior is not stored: a 256-point field is 2 MB per snapshot in
    complex double precision, and a chapter that wants a time series should call
    this repeatedly with a small ``n_steps``.
    """
    state = np.asarray(vorticity_hat).copy()
    for _ in range(int(n_steps)):
        k1 = vorticity_tendency(state, grid, viscosity, hyper_order)
        k2 = vorticity_tendency(state + 0.5 * dt * k1, grid, viscosity, hyper_order)
        k3 = vorticity_tendency(state + 0.5 * dt * k2, grid, viscosity, hyper_order)
        k4 = vorticity_tendency(state + dt * k3, grid, viscosity, hyper_order)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def _hermitian_weights(grid: dict) -> Array:
    """Multiplicity of each stored mode under the rfft2 half-spectrum.

    ``rfft2`` keeps only non-negative ``ky``, so every column except the first
    and (for even ``n``) the last stands for two physical modes. Getting this
    wrong is a factor-of-two error in every energy and enstrophy this module
    reports, and it would not show up in a conservation test, because the same
    wrong weight appears on both sides.
    """
    weights = np.ones_like(grid["k2"])
    last = -1 if grid["n"] % 2 == 0 else None
    weights[:, 1:last] = 2.0
    return weights


def energy(vorticity_hat: Complex, grid: dict) -> float:
    r"""Domain-averaged kinetic energy, :math:`\tfrac12\langle|u|^2\rangle`.

    In spectral terms :math:`\tfrac12\sum |\hat\zeta|^2/k^2`. Conserved exactly
    by the inviscid equations, which is one of this module's two test anchors.
    """
    weights = _hermitian_weights(grid)
    total = 0.5 * np.sum(
        weights * grid["inverse_k2"] * np.abs(vorticity_hat) ** 2
    )
    return float(total) / grid["n"] ** 4


def enstrophy(vorticity_hat: Complex, grid: dict) -> float:
    r"""Domain-averaged enstrophy, :math:`\tfrac12\langle\zeta^2\rangle`.

    The second inviscid invariant, and the one three-dimensional flow does not
    have. Its conservation is what forbids the forward energy cascade in two
    dimensions and so what makes the two cases differ.
    """
    weights = _hermitian_weights(grid)
    return float(0.5 * np.sum(weights * np.abs(vorticity_hat) ** 2)) / grid["n"] ** 4


def energy_spectrum(vorticity_hat: Complex, grid: dict) -> tuple[Array, Array]:
    r"""Shell-averaged energy spectrum :math:`E(k)`, on integer :math:`k`.

    Returns ``(wavenumbers, spectrum)`` with ``wavenumbers`` running
    :math:`0 \ldots N/3`, beyond which the shells would be sampled only in the
    corners of the retained region and their averages would be meaningless.

    **The shell sum is therefore not always the energy.** The two-thirds mask is
    a *square* in :math:`(k_x, k_y)`, so modes survive out to
    :math:`|k| = \sqrt2\,N/3` in its corners, and those are not represented in
    any shell. The shortfall is negligible when the field is resolved -- below
    0.02 % at :math:`N \ge 64` -- and becomes serious when the spectral peak
    approaches the truncation: measured 12.4 % at :math:`N = 32` with the peak
    at :math:`k = 10 = N/3`. That makes the ratio of the shell sum to
    :func:`energy` a useful resolution diagnostic in its own right, and the
    tests assert both the agreement when resolved and the failure when not.
    """
    magnitude = np.sqrt(grid["k2"])
    weights = _hermitian_weights(grid)
    density = (
        0.5 * weights * grid["inverse_k2"] * np.abs(vorticity_hat) ** 2
        / grid["n"] ** 4
    )
    binned = np.rint(magnitude).astype(int)
    highest = grid["n"] // 3
    spectrum = np.zeros(highest + 1, dtype=float)
    for shell in range(1, highest + 1):
        spectrum[shell] = float(density[binned == shell].sum())
    return np.arange(highest + 1), spectrum


def local_spectral_slope(
    wavenumbers: Array, spectrum: Array
) -> Array:
    r"""Local slope :math:`-d\ln E/d\ln k`, the honest spectral diagnostic.

    A power law shows a plateau here and nothing else does. Chapter 8 made the
    same argument for fractal dimension and chapter 9 for error growth: fitting
    a slope over a range chosen after the fact will return a number whatever the
    data does, and the local slope is what says whether the number means
    anything.

    Returns ``nan`` where the spectrum is empty. The sign convention is such
    that a :math:`k^{-3}` range reads :math:`+3`.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    out = np.full(spectrum.shape, np.nan)
    usable = (spectrum > 0.0) & (wavenumbers > 0.0)
    if int(usable.sum()) < 3:
        return out
    out[usable] = -np.gradient(
        np.log(spectrum[usable]), np.log(wavenumbers[usable])
    )
    return out


def turnover_time(wavenumbers: Array, spectrum: Array) -> Array:
    r"""Eddy turnover time at each wavenumber, :math:`\tau(k)`.

    Taking the velocity of an eddy of size :math:`1/k` as
    :math:`u_k^2 \sim k E(k)`, its turnover time is
    :math:`\tau(k) \sim 1/(k u_k) = [k^3 E(k)]^{-1/2}`.

    **This is the bridge to chapter 12, and its consequence is worth stating
    before it is measured.** If :math:`E(k)\sim k^{-p}` then
    :math:`\tau \sim k^{(p-3)/2}`, so the growth rate at scale :math:`k` goes as
    :math:`k^{(3-p)/2}` and chapter 12's exponent is

    .. math:: \alpha = \frac{3-p}{2}.

    Three-dimensional turbulence has :math:`p = 5/3` and hence
    :math:`\alpha = 2/3`, which is exactly the Kolmogorov case where chapter 12
    measured a **finite** predictability horizon of 1.4466. Two-dimensional
    turbulence has :math:`p = 3` in its enstrophy range and hence
    :math:`\alpha = 0` -- the scale-independent case, where chapter 12 measured
    growth without bound, 0.281 per octave out to 128 octaves.

    So whether a flow has an intrinsic predictability limit is decided by its
    spectral slope, and the two turbulent cases fall on opposite sides.

    Note that :math:`\alpha = (3-p)/2` is an **algebraic identity** given this
    estimate of :math:`\tau`, not an independent measurement -- the slope of the
    returned array against :math:`\ln k` is fixed by the slope of the spectrum
    it was computed from. Chapter 14 says so, and tests the physics by watching
    an error cascade instead.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    out = np.full(spectrum.shape, np.nan)
    usable = (spectrum > 0.0) & (wavenumbers > 0.0)
    out[usable] = 1.0 / np.sqrt(
        wavenumbers[usable] ** 3 * spectrum[usable]
    )
    return out


def random_vorticity(
    grid: dict, peak: float = 10.0, seed: int | None = 0
) -> Complex:
    r"""A random field with a spectrum peaked at wavenumber ``peak``.

    The standard initial condition for decaying two-dimensional turbulence:
    amplitude :math:`\propto k^3/(1 + (k/k_0)^8)`, random phases, zero mean,
    normalised to unit enstrophy so that runs at different resolutions are
    comparable.
    """
    rng = np.random.default_rng(seed)
    magnitude = np.sqrt(grid["k2"])
    amplitude = magnitude**3 / (1.0 + (magnitude / float(peak)) ** 8)
    phase = rng.uniform(0.0, 2.0 * np.pi, magnitude.shape)
    field_hat = amplitude * np.exp(1j * phase)
    field_hat[0, 0] = 0.0
    # Round-trip so the field is real and Hermitian-consistent, then mask.
    field = np.fft.irfft2(field_hat)
    field_hat = np.fft.rfft2(field - field.mean()) * grid["mask"]
    return field_hat / np.sqrt(enstrophy(field_hat, grid))


def vorticity_field(vorticity_hat: Complex) -> Array:
    """The physical vorticity, for plotting."""
    return np.fft.irfft2(np.asarray(vorticity_hat))


def band_perturbation(
    grid: dict, centre: float, amplitude: float, width: float = 0.5,
    seed: int | None = 0,
) -> Complex:
    r"""A random perturbation confined to :math:`|k| \in` ``centre`` ± ``width``.

    What "seeding error at one scale" means for a field: the perturbation has
    support in one narrow shell and nowhere else, so its later spread across
    the spectrum is the cascade and not the initial condition. Normalised to
    ``amplitude`` in the enstrophy norm.
    """
    rng = np.random.default_rng(seed)
    magnitude = np.sqrt(grid["k2"])
    shell = (magnitude >= centre - width) & (magnitude <= centre + width)
    phase = rng.uniform(0.0, 2.0 * np.pi, magnitude.shape)
    perturbation = np.zeros_like(grid["k2"], dtype=complex)
    perturbation[shell] = np.exp(1j * phase[shell])
    perturbation = perturbation * grid["mask"]
    scale = np.sqrt(enstrophy(perturbation, grid))
    if not scale > 0.0:
        raise ValueError(
            f"no modes in the shell |k| = {centre} +- {width}; the grid reaches "
            f"{grid['n'] // 3}"
        )
    return float(amplitude) * perturbation / scale
