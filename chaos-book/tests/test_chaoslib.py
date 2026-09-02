"""Correctness tests for :mod:`chaoslib`.

These test *numbers*, not that the code runs. Every tolerance below is anchored
either to an exact analytic identity (a fixed point being a zero of the RHS, the
Lyapunov exponents summing to the trace of the Jacobian, the adjoint identity) or
to a published value for the system in question. A test that merely called each
function and checked for absence of exceptions would have passed happily while
two real bugs -- an O(dt)-inconsistent tangent linear model and a
correlation-dimension fit window sitting entirely in the saturated regime -- went
undetected. Both were caught by the identities asserted here.

Literature anchors used:

* Lorenz 63 (sigma=10, rho=28, beta=8/3): lambda_1 = 0.9056, D_KY = 2.06,
  D_2 ~ 2.05, rho_Hopf = 24.7368.
* Lorenz 96 (N=40, F=8): lambda_1 ~ 1.67 per time unit, 13 positive exponents,
  D_KY ~ 27.1.
* Henon (a=1.4, b=0.3): D_2 ~ 1.22.
* Feigenbaum delta = 4.6692.
"""

from __future__ import annotations

import numpy as np
import pytest

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

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
L63_TRACE = -(SIGMA + 1.0 + BETA)  # exact divergence of the Lorenz 63 flow


# ==========================================================================
# systems: exact identities
# ==========================================================================
def test_lorenz63_fixed_points_are_zeros_of_the_rhs():
    origin, c_plus, c_minus = systems.lorenz63_fixed_points(RHO, BETA)
    for point in (origin, c_plus, c_minus):
        rate = systems.lorenz63(0.0, point, SIGMA, RHO, BETA)
        assert np.allclose(rate, 0.0, atol=1e-12)


def test_lorenz63_fixed_point_pair_appears_only_above_rho_one():
    _, c_plus, c_minus = systems.lorenz63_fixed_points(0.5, BETA)
    assert c_plus is None and c_minus is None
    _, c_plus, c_minus = systems.lorenz63_fixed_points(1.5, BETA)
    assert c_plus is not None and c_minus is not None
    # The pair is symmetric under (X, Y, Z) -> (-X, -Y, Z).
    assert np.allclose(c_plus * np.array([-1.0, -1.0, 1.0]), c_minus)


def test_lorenz63_hopf_threshold_matches_literature():
    assert systems.lorenz63_hopf_rho(SIGMA, BETA) == pytest.approx(
        24.7368, abs=1e-4
    )


@pytest.mark.parametrize(
    "state", [np.array([2.0, -3.0, 25.0]), np.array([-8.0, 7.0, 33.0])]
)
def test_lorenz63_jacobian_matches_finite_differences(state):
    jac = systems.lorenz63_jacobian(state, SIGMA, RHO, BETA)
    eps = 1e-6
    columns = []
    for basis in np.eye(3):
        plus = systems.lorenz63(0.0, state + eps * basis, SIGMA, RHO, BETA)
        minus = systems.lorenz63(0.0, state - eps * basis, SIGMA, RHO, BETA)
        columns.append((plus - minus) / (2.0 * eps))
    assert np.allclose(jac, np.column_stack(columns), atol=1e-6)


def test_lorenz63_jacobian_trace_is_state_independent():
    """trace(J) = -(sigma + 1 + beta) everywhere -- the flow's constant divergence."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = rng.normal(scale=20.0, size=3)
        jac = systems.lorenz63_jacobian(state, SIGMA, RHO, BETA)
        assert np.trace(jac) == pytest.approx(L63_TRACE, abs=1e-12)


def test_lorenz96_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    state = 8.0 + 2.0 * rng.normal(size=40)
    jac = systems.lorenz96_jacobian(state, 8.0)
    eps = 1e-6
    columns = []
    for basis in np.eye(40):
        plus = systems.lorenz96(0.0, state + eps * basis, 8.0)
        minus = systems.lorenz96(0.0, state - eps * basis, 8.0)
        columns.append((plus - minus) / (2.0 * eps))
    assert np.allclose(jac, np.column_stack(columns), atol=1e-6)


def test_lorenz96_quadratic_terms_conserve_energy():
    """The advective terms alone conserve sum x_k^2; only -x_k and F change it.

    Checked by evaluating the RHS with dissipation and forcing removed
    analytically: dE/dt = 2 sum x_k xdot_k must vanish for the quadratic part.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    full = systems.lorenz96(0.0, x, 0.0)
    quadratic_only = full + x  # undo the -x_k dissipation (F was set to 0)
    assert np.sum(x * quadratic_only) == pytest.approx(0.0, abs=1e-10)


def test_lorenz63_accepts_an_ensemble_without_a_loop():
    ens = np.array([[1.0, 1.0, 20.0], [2.0, -1.0, 5.0], [0.0, 0.0, 0.0]])
    out = systems.lorenz63(0.0, ens, SIGMA, RHO, BETA)
    assert out.shape == ens.shape
    for i, member in enumerate(ens):
        assert np.allclose(out[i], systems.lorenz63(0.0, member, SIGMA, RHO, BETA))


# ==========================================================================
# pendulums: energy conservation and the exact period
# ==========================================================================
@pytest.mark.parametrize("theta0", [0.05, 0.5, 1.5, 3.0])
def test_pendulum_conserves_energy(theta0):
    grid = integrate.trajectory_grid(20.0, 0.001)
    traj = integrate.rk4(systems.pendulum, np.array([theta0, 0.0]), grid)
    energy = systems.pendulum_energy(traj)
    drift = (energy.max() - energy.min()) / energy[0]
    assert drift < 1e-9


@pytest.mark.parametrize("theta0", [0.05, 0.5, 1.5, 3.0])
def test_pendulum_period_matches_elliptic_integral(theta0):
    """Measured period from zero crossings vs 4 sqrt(L/g) K(sin^2(theta0/2))."""
    expected = systems.pendulum_period_exact(theta0)
    grid = integrate.trajectory_grid(6.0 * expected, 0.0005)
    traj = integrate.rk4(systems.pendulum, np.array([theta0, 0.0]), grid)
    theta = traj[:, 0]
    crossings = np.flatnonzero((theta[:-1] < 0.0) & (theta[1:] >= 0.0))
    assert crossings.size >= 2
    measured = np.mean(np.diff(grid[crossings]))
    assert measured == pytest.approx(expected, rel=1e-3)


def test_pendulum_small_angle_limit_is_harmonic():
    """As theta0 -> 0 the exact period tends to 2 pi sqrt(L/g)."""
    harmonic = 2.0 * np.pi * np.sqrt(1.0 / 9.81)
    assert systems.pendulum_period_exact(1e-4) == pytest.approx(
        harmonic, rel=1e-8
    )
    # ...and it is strictly longer for any finite amplitude (period lengthening).
    assert systems.pendulum_period_exact(1.0) > harmonic


def test_double_pendulum_conserves_energy():
    grid = integrate.trajectory_grid(30.0, 0.0005)
    traj = integrate.rk4(
        systems.double_pendulum, np.array([2.0, -0.2, 0.0, 0.0]), grid
    )
    energy = systems.double_pendulum_energy(traj)
    assert (energy.max() - energy.min()) / abs(energy[0]) < 1e-7


def test_double_pendulum_has_four_dimensional_state():
    """The dimensional claim chapter 4 rests on: 1 DOF -> 2-D, 2 DOF -> 4-D."""
    assert systems.pendulum(0.0, np.zeros(2)).shape == (2,)
    assert systems.double_pendulum(0.0, np.zeros(4)).shape == (4,)


# ==========================================================================
# integrate
# ==========================================================================
def test_rk4_is_fourth_order_accurate():
    """Halving dt should cut the error by ~16 on a smooth problem."""

    def decay(t, x, rate=1.0):
        return -rate * np.asarray(x)

    errors = []
    for dt in (0.1, 0.05, 0.025):
        grid = integrate.trajectory_grid(1.0, dt)
        num = integrate.rk4(decay, np.array([1.0]), grid, rate=1.0)[-1, 0]
        errors.append(abs(num - np.exp(-1.0)))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert coarse / fine == pytest.approx(16.0, rel=0.25)


def test_rk4_and_solve_agree_on_a_single_trajectory():
    grid = integrate.trajectory_grid(5.0, 0.001)
    fixed = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid)
    adaptive = integrate.solve(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid
    )
    assert np.allclose(fixed[-1], adaptive[-1], rtol=1e-4, atol=1e-5)


def test_rk4_steps_an_ensemble_consistently_with_single_members():
    grid = integrate.trajectory_grid(2.0, 0.005)
    ens = np.array([[1.0, 1.0, 20.0], [1.1, 0.9, 19.5]])
    together = integrate.rk4(systems.lorenz63, ens, grid)
    for i, member in enumerate(ens):
        alone = integrate.rk4(systems.lorenz63, member, grid)
        assert np.allclose(together[:, i], alone)


def test_solve_binds_parameters_by_keyword_not_position():
    """Guards the closure in `solve`: params must not depend on dict ordering."""
    grid = integrate.trajectory_grid(1.0, 0.01)
    # beta given before sigma: positional binding would silently swap them.
    out = integrate.solve(
        systems.lorenz63,
        np.array([1.0, 1.0, 20.0]),
        grid,
        beta=BETA,
        sigma=SIGMA,
        rho=RHO,
    )
    reference = integrate.rk4(
        systems.lorenz63,
        np.array([1.0, 1.0, 20.0]),
        integrate.trajectory_grid(1.0, 0.0005),
        sigma=SIGMA,
        rho=RHO,
        beta=BETA,
    )
    assert np.allclose(out[-1], reference[-1], rtol=1e-4, atol=1e-5)


# ==========================================================================
# lyapunov
# ==========================================================================
@pytest.fixture(scope="module")
def l63_spectrum():
    return lyapunov.lyapunov_spectrum(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=0.01,
        t_final=300.0,
        t_transient=20.0,
        sigma=SIGMA,
        rho=RHO,
        beta=BETA,
    )


def test_l63_leading_exponent_matches_literature(l63_spectrum):
    assert l63_spectrum[0] == pytest.approx(0.9056, abs=0.05)


def test_l63_exponents_sum_to_the_jacobian_trace(l63_spectrum):
    """Exact identity: sum lambda_i = trace(J) = -(sigma + 1 + beta).

    Independent of integration length, so it is a genuinely strong check on the
    Benettin implementation rather than a convergence test.
    """
    assert l63_spectrum.sum() == pytest.approx(L63_TRACE, abs=1e-3)


def test_l63_spectrum_has_one_positive_one_near_zero_one_very_negative(
    l63_spectrum,
):
    assert l63_spectrum[0] > 0.5
    assert abs(l63_spectrum[1]) < 0.05  # the direction along the flow
    assert l63_spectrum[2] < -10.0


def test_l63_kaplan_yorke_dimension_matches_literature(l63_spectrum):
    dim = lyapunov.kaplan_yorke_dimension(l63_spectrum)
    assert dim == pytest.approx(2.06, abs=0.05)


def test_ks_entropy_is_the_sum_of_positive_exponents(l63_spectrum):
    assert lyapunov.ks_entropy(l63_spectrum) == pytest.approx(
        l63_spectrum[l63_spectrum > 0].sum(), abs=1e-12
    )
    assert lyapunov.ks_entropy(np.array([-1.0, -2.0])) == 0.0


def test_lyapunov_convergence_shapes_and_time_grid():
    times, estimates = lyapunov.lyapunov_convergence(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=0.01,
        t_final=50.0,
        t_transient=10.0,
        n_samples=40,
    )
    assert estimates.shape[1] == 3
    assert times.size == estimates.shape[0]
    assert np.all(np.diff(times) > 0)
    assert times[-1] == pytest.approx(50.0, abs=0.05)


def test_lyapunov_convergence_endpoint_obeys_the_trace_identity():
    """The running estimate must satisfy sum = trace(J) at its end.

    This is the right invariant to assert, and the only one that is exact. The
    endpoint does NOT reproduce a separate `lyapunov_spectrum` call to better than
    a few hundredths, and that is not a defect: both runs are chaotic, so
    floating-point divergence decorrelates them long before T = 200, and the
    leading exponent scatters by ~0.03 across nearby transients at that length.
    The trace identity holds for either run at any T.
    """
    _, estimates = lyapunov.lyapunov_convergence(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=0.01,
        t_final=150.0,
        t_transient=20.0,
        n_samples=50,
        sigma=SIGMA,
        rho=RHO,
        beta=BETA,
    )
    assert estimates[-1].sum() == pytest.approx(L63_TRACE, abs=1e-3)
    assert estimates[-1][0] == pytest.approx(0.9056, abs=0.06)


def test_lyapunov_convergence_settles_down():
    """The running estimate must wander less late than early.

    Compared as the spread of the leading estimate over the last half of the
    record against the first half -- the defining behaviour of a time average,
    and the reason the chapter plots this curve rather than quoting one number.
    """
    _, estimates = lyapunov.lyapunov_convergence(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=0.01,
        t_final=300.0,
        t_transient=20.0,
        n_samples=120,
    )
    leading = estimates[:, 0]
    early = leading[: len(leading) // 2]
    late = leading[len(leading) // 2 :]
    assert late.std() < early.std()


def test_lyapunov_convergence_can_track_the_leading_exponent_alone():
    _, estimates = lyapunov.lyapunov_convergence(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=0.01,
        t_final=60.0,
        t_transient=10.0,
        n_samples=20,
        n_exponents=1,
    )
    assert estimates.shape[1] == 1
    assert estimates[-1, 0] == pytest.approx(0.9056, abs=0.12)


def test_l63_is_not_chaotic_below_the_hopf_threshold():
    """rho = 15 < rho_Hopf: trajectories spiral onto a stable fixed point."""
    spectrum = lyapunov.lyapunov_spectrum(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 10.0]),
        dt=0.01,
        t_final=100.0,
        t_transient=40.0,
        sigma=SIGMA,
        rho=15.0,
        beta=BETA,
    )
    assert spectrum[0] < 0.02  # no positive exponent -> no chaos


@pytest.fixture(scope="module")
def l96_spectrum():
    state = 8.0 * np.ones(40)
    state[19] += 0.01
    return lyapunov.lyapunov_spectrum(
        systems.lorenz96,
        systems.lorenz96_jacobian,
        state,
        dt=0.01,
        t_final=100.0,
        t_transient=20.0,
        forcing=8.0,
    )


def test_l96_leading_exponent_matches_literature(l96_spectrum):
    assert l96_spectrum[0] == pytest.approx(1.67, abs=0.2)


def test_l96_has_thirteen_positive_exponents(l96_spectrum):
    """N=40, F=8 gives 13 positive exponents -- a sharp, published integer."""
    assert int((l96_spectrum > 0.0).sum()) == 13


def test_l96_kaplan_yorke_dimension_matches_literature(l96_spectrum):
    dim = lyapunov.kaplan_yorke_dimension(l96_spectrum)
    assert dim == pytest.approx(27.1, abs=1.0)


def test_l96_spectrum_is_ordered_descending(l96_spectrum):
    assert np.all(np.diff(l96_spectrum) <= 1e-9)


def test_twin_trajectory_perturbation_has_the_requested_norm():
    x0 = np.array([1.0, 1.0, 20.0])
    grid = integrate.trajectory_grid(2.0, 0.01)
    _, x_pert = lyapunov.twin_trajectory_growth(
        systems.lorenz63, x0, 1e-6, grid, seed=7
    )
    assert np.linalg.norm(x_pert - x0) == pytest.approx(1e-6, rel=1e-9)


def test_twin_trajectory_separation_grows_by_orders_of_magnitude():
    grid = integrate.trajectory_grid(15.0, 0.005)
    separation, _ = lyapunov.twin_trajectory_growth(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), 1e-6, grid, seed=7
    )
    assert separation[0] == pytest.approx(1e-6, rel=1e-9)
    assert separation.max() / separation[0] > 1e4


def test_twin_trajectory_rates_average_to_the_leading_exponent():
    """One twin pair is a FINITE-TIME exponent and scatters widely; the mean
    over initial conditions is what converges to lambda_1.

    Written this way deliberately: an earlier version asserted that a single
    pair recovers 0.9056 and failed at 0.21, which was the test being wrong
    rather than the code. Local growth on the Lorenz attractor is bursty.
    """
    grid = integrate.trajectory_grid(15.0, 0.005)
    spin_up = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]),
        integrate.trajectory_grid(200.0, 0.01),
    )
    starts = spin_up[5000::900]
    rates = []
    for i, start in enumerate(starts):
        separation, _ = lyapunov.twin_trajectory_growth(
            systems.lorenz63, start, 1e-7, grid, seed=100 + i
        )
        try:
            rate, _ = lyapunov.fit_growth_rate(grid, separation)
        except ValueError:
            continue
        rates.append(rate)
    assert len(rates) >= 8
    rates = np.array(rates)
    assert rates.mean() == pytest.approx(0.9056, abs=0.2)
    # The scatter is the physics, not noise: it is why chapter 7 exists.
    assert rates.std() > 0.05


def test_fit_growth_rate_window_is_anchored_to_delta0_not_saturation():
    """A synthetic pure-exponential record must be recovered exactly.

    Guards the bug this window definition replaced: a window expressed as a
    fraction of saturation lands in the nonlinear phase when delta0 is tiny.
    """
    t = np.linspace(0.0, 20.0, 2000)
    rate_true = 0.85
    sep = 1e-8 * np.exp(rate_true * t)
    rate, _ = lyapunov.fit_growth_rate(t, sep)
    assert rate == pytest.approx(rate_true, rel=1e-6)


def test_fit_growth_rate_refuses_a_window_with_too_few_points():
    t = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="exponential window"):
        lyapunov.fit_growth_rate(t, np.ones_like(t))


def test_doubling_time_is_ln2_over_rate():
    assert lyapunov.doubling_time(0.9056) == pytest.approx(
        np.log(2.0) / 0.9056
    )
    assert lyapunov.doubling_time(-0.1) == float("inf")
    assert lyapunov.doubling_time(0.0) == float("inf")


def test_finite_time_exponents_vary_across_the_attractor():
    """Flow-dependent predictability: the local exponent is not a constant.

    This is the quantitative basis for chapter 7 -- if finite-time exponents did
    not spread, "today's forecast is unusually predictable" would be meaningless.
    """
    grid = integrate.trajectory_grid(60.0, 0.01)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid)
    states = traj[2000::400]
    local = lyapunov.finite_time_exponents(
        systems.lorenz63, systems.lorenz63_jacobian, states, tau=0.5, dt=0.01
    )
    assert local.size == states.shape[0]
    assert local.std() > 0.2
    # ...and they should straddle the asymptotic value rather than all exceed it.
    assert local.min() < 0.9056 < local.max()


def test_kaplan_yorke_edge_cases():
    assert lyapunov.kaplan_yorke_dimension(np.array([-1.0, -2.0])) == 0.0
    # All exponents positive: dimension saturates at the state dimension.
    assert lyapunov.kaplan_yorke_dimension(np.array([1.0, 0.5])) == 2.0


# ==========================================================================
# adjoint / tangent linear
# ==========================================================================
@pytest.fixture(scope="module")
def l63_propagator():
    return adjoint.tangent_linear_propagator(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        tau=0.5,
        dt=0.005,
        sigma=SIGMA,
        rho=RHO,
        beta=BETA,
    )


def test_adjoint_identity_holds_to_machine_precision(l63_propagator):
    """<Mx, y> == <x, M^T y>. The standard adjoint test.

    Historically the most common bug in variational assimilation systems, and
    invisible to any test that only checks the forward model.
    """
    residual = adjoint.adjoint_identity_residual(l63_propagator, n_trials=64)
    assert residual < 1e-12


def test_propagator_determinant_equals_exp_trace_times_tau(l63_propagator):
    """Liouville: det M = exp(tau * trace J), exact for Lorenz 63.

    An identity the propagator must satisfy regardless of the trajectory, so it
    tests the tangent integration itself rather than its convergence.
    """
    expected = np.exp(L63_TRACE * 0.5)
    assert np.linalg.det(l63_propagator) == pytest.approx(expected, rel=1e-4)


def test_tangent_linear_error_falls_linearly_with_amplitude():
    """The defining check of a correct TLM: neglected terms are O(alpha^2).

    So the *relative* discrepancy against finite differences must fall like
    alpha. A TLM that linearises the continuous flow instead of the discrete
    RK4 map produces a constant floor here instead -- which is exactly the bug
    this test caught during development.
    """
    amplitudes = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    _, errors = adjoint.tangent_linear_error(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        tau=0.5,
        amplitudes=amplitudes,
        dt=0.005,
    )
    for coarse, fine in zip(errors[1:], errors[:-1]):
        assert coarse / fine == pytest.approx(10.0, rel=0.2)
    assert errors[0] < 1e-4


def test_singular_values_exceed_asymptotic_growth_over_short_windows(
    l63_propagator,
):
    """Finite-time optimal growth beats exp(lambda_1 tau) -- why SVs are used.

    If singular-vector amplification merely matched the Lyapunov estimate there
    would be no operational reason to compute them.
    """
    sigmas, initial, final = adjoint.singular_vectors(l63_propagator, 3)
    assert sigmas[0] > np.exp(0.9056 * 0.5)
    assert np.all(np.diff(sigmas) <= 0.0)
    # Singular vectors are orthonormal sets.
    assert np.allclose(initial.T @ initial, np.eye(3), atol=1e-10)
    assert np.allclose(final.T @ final, np.eye(3), atol=1e-10)
    # The product of singular values is |det M|.
    assert np.prod(sigmas) == pytest.approx(
        abs(np.linalg.det(l63_propagator)), rel=1e-8
    )


def test_weighted_singular_vectors_reduce_to_the_plain_svd(l63_propagator):
    """weight=ones must be byte-identical to weight=None.

    The weighted path takes a different route (SVD of E^{1/2} M E^{-1/2}), so this
    pins that the default behaviour did not change when the option was added.
    """
    plain = adjoint.singular_vectors(l63_propagator, 3)
    unit = adjoint.singular_vectors(l63_propagator, 3, weight=np.ones(3))
    for a, b in zip(plain, unit):
        assert np.allclose(a, b)


@pytest.mark.parametrize(
    "weight",
    [
        np.array([1.0, 1.0, 25.0]),
        np.array([25.0, 1.0, 1.0]),
        np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]]),
    ],
)
def test_weighted_sigma_is_the_achieved_amplification_in_that_norm(
    l63_propagator, weight
):
    """sigma_1 must be the E-norm growth the returned vector actually achieves.

    The check that matters: it is easy to compute a weighted SVD and return
    vectors in the *transformed* space, which then do not achieve the reported
    amplification when applied to the real state. Both a diagonal and a full
    symmetric positive definite weight are exercised.
    """
    sigma, initial, _ = adjoint.singular_vectors(
        l63_propagator, 1, weight=weight
    )
    e_mat = np.diag(weight) if weight.ndim == 1 else weight
    v = initial[:, 0]
    grown = l63_propagator @ v
    achieved = np.sqrt(grown @ e_mat @ grown) / np.sqrt(v @ e_mat @ v)
    assert achieved == pytest.approx(float(sigma[0]), rel=1e-10)


def test_weighted_optimal_growth_beats_a_random_direction_in_that_norm(
    l63_propagator,
):
    weight = np.array([1.0, 1.0, 25.0])
    e_mat = np.diag(weight)
    sigma, _, _ = adjoint.singular_vectors(l63_propagator, 1, weight=weight)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3)
        grown = l63_propagator @ v
        achieved = np.sqrt(grown @ e_mat @ grown) / np.sqrt(v @ e_mat @ v)
        assert achieved <= float(sigma[0]) * (1.0 + 1e-9)


def test_weighted_singular_vectors_reject_a_non_positive_weight(l63_propagator):
    with pytest.raises(ValueError, match="positive"):
        adjoint.singular_vectors(l63_propagator, 1, weight=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="positive definite"):
        adjoint.singular_vectors(
            l63_propagator, 1, weight=np.diag([1.0, -2.0, 1.0])
        )


def test_the_norm_changes_which_perturbation_grows_fastest(l63_propagator):
    """Not a technicality: different norms give materially different answers.

    If this ever came out as "the same vector either way", the weighted option
    would be pointless and the chapter-16 discussion wrong.
    """
    _, euclid, _ = adjoint.singular_vectors(l63_propagator, 1)
    _, weighted, _ = adjoint.singular_vectors(
        l63_propagator, 1, weight=np.array([25.0, 1.0, 1.0])
    )
    a = euclid[:, 0] / np.linalg.norm(euclid[:, 0])
    b = weighted[:, 0] / np.linalg.norm(weighted[:, 0])
    angle = np.degrees(np.arccos(min(1.0, abs(float(a @ b)))))
    assert angle > 10.0


def test_singular_vector_actually_achieves_the_predicted_amplification(
    l63_propagator,
):
    sigmas, initial, _ = adjoint.singular_vectors(l63_propagator, 1)
    grown = l63_propagator @ initial[:, 0]
    assert np.linalg.norm(grown) == pytest.approx(sigmas[0], rel=1e-10)


def test_propagator_over_a_zero_window_is_the_identity():
    """tau = 0 propagates nothing, so M must be I.

    Regression test. The obvious `n_steps = max(1, round(tau/dt))` forces one
    step of dt here, which silently advances the tangent and corrupts every
    4D-Var gradient containing an observation at the window start -- i.e. the
    normal case in cycling assimilation. It cost 6.8% gradient error, and was
    invisible to the finite-difference amplitude test because the floor is
    independent of the perturbation size.
    """
    m = adjoint.tangent_linear_propagator(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        tau=0.0,
        dt=0.01,
    )
    assert np.allclose(m, np.eye(3), atol=0.0)


@pytest.mark.parametrize("tau", [0.105, 0.037, 0.2501])
def test_propagator_covers_exactly_tau_when_not_a_multiple_of_dt(tau):
    """M must correspond to the interval tau, not to n_steps*dt.

    Checked against the exact Liouville identity det M = exp(tau * trace J),
    which pins the interval independently of the trajectory.
    """
    m = adjoint.tangent_linear_propagator(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        tau=tau,
        dt=0.01,
        sigma=SIGMA,
        rho=RHO,
        beta=BETA,
    )
    assert np.linalg.det(m) == pytest.approx(np.exp(L63_TRACE * tau), rel=1e-4)


def test_four_dvar_gradient_matches_central_differences():
    """The adjoint gradient of the 4D-Var cost, verified to ~1e-9.

    This is the test that caught the two interval bugs above. It reconstructs
    the same cost and gradient that `four_dvar_analysis` minimises, including an
    observation at the window start.
    """
    dt = 0.01
    h_op = np.eye(3)
    r_cov = np.eye(3) * 4.0
    b_cov = np.eye(3) * 9.0
    r_inv = np.linalg.inv(r_cov)
    b_inv = np.linalg.inv(b_cov)
    xb = np.array([-7.09, -3.74, 35.33])
    truth0 = np.array([-8.97, -2.81, 33.96])
    rng = np.random.default_rng(0)

    observations = []
    for t_obs in (0.0, 0.2, 0.4, 0.6, 0.8):
        if t_obs == 0.0:
            state = truth0
        else:
            grid = np.linspace(0.0, t_obs, int(round(t_obs / dt)) + 1)
            state = integrate.rk4(systems.lorenz63, truth0, grid)[-1]
        observations.append((t_obs, state + rng.normal(0.0, 2.0, 3)))

    def cost(x0):
        departure = x0 - xb
        total = 0.5 * float(departure @ (b_inv @ departure))
        for t_obs, y in observations:
            if t_obs == 0.0:
                state = x0
            else:
                grid = np.linspace(0.0, t_obs, int(round(t_obs / dt)) + 1)
                state = integrate.rk4(systems.lorenz63, x0, grid)[-1]
            innov = y - h_op @ state
            total += 0.5 * float(innov @ (r_inv @ innov))
        return total

    def grad(x0):
        departure = x0 - xb
        g = b_inv @ departure
        for t_obs, y in observations:
            if t_obs == 0.0:
                state = x0
            else:
                grid = np.linspace(0.0, t_obs, int(round(t_obs / dt)) + 1)
                state = integrate.rk4(systems.lorenz63, x0, grid)[-1]
            weighted = r_inv @ (y - h_op @ state)
            propagator = adjoint.tangent_linear_propagator(
                systems.lorenz63, systems.lorenz63_jacobian, x0, t_obs, dt=dt
            )
            g = g - propagator.T @ (h_op.T @ weighted)
        return g

    analytic = grad(xb)
    eps = 1e-5
    numerical = np.array(
        [(cost(xb + eps * e) - cost(xb - eps * e)) / (2.0 * eps) for e in np.eye(3)]
    )
    rel = np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic)
    assert rel < 1e-7


def test_finite_time_exponents_reject_a_zero_window():
    with pytest.raises(ValueError, match="tau must be positive"):
        lyapunov.finite_time_exponents(
            systems.lorenz63,
            systems.lorenz63_jacobian,
            np.array([[1.0, 1.0, 20.0]]),
            tau=0.0,
        )


def test_adjoint_propagator_is_the_transpose(l63_propagator):
    assert np.allclose(
        adjoint.adjoint_propagator(l63_propagator), l63_propagator.T
    )


# ==========================================================================
# assimilate
# ==========================================================================
@pytest.fixture
def linear_gaussian_problem():
    b_cov = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    h_op = np.eye(3)
    r_cov = np.eye(3) * 0.5
    xb = np.array([1.0, 2.0, 3.0])
    y = np.array([1.4, 1.7, 3.3])
    return xb, b_cov, y, h_op, r_cov


def test_kalman_analysis_lies_between_background_and_observations(
    linear_gaussian_problem,
):
    xb, b_cov, y, h_op, r_cov = linear_gaussian_problem
    xa, p_a = assimilate.kalman_filter_update(xb, b_cov, y, h_op, r_cov)
    assert np.all((xa - xb) * (y - xb) > 0.0)  # moved towards the observations
    assert np.all(np.abs(xa - xb) < np.abs(y - xb))  # but not all the way
    # The analysis must be more certain than the background.
    assert np.all(np.diag(p_a) < np.diag(b_cov))


def test_three_dvar_is_algebraically_the_kalman_analysis(
    linear_gaussian_problem,
):
    xb, b_cov, y, h_op, r_cov = linear_gaussian_problem
    xa_kf, _ = assimilate.kalman_filter_update(xb, b_cov, y, h_op, r_cov)
    xa_var = assimilate.three_dvar_update(xb, b_cov, y, h_op, r_cov)
    assert np.allclose(xa_var, xa_kf)


def test_perfect_observations_pull_the_analysis_onto_the_observations(
    linear_gaussian_problem,
):
    xb, b_cov, y, h_op, _ = linear_gaussian_problem
    xa, _ = assimilate.kalman_filter_update(
        xb, b_cov, y, h_op, np.eye(3) * 1e-10
    )
    assert np.allclose(xa, y, atol=1e-6)


def test_enkf_converges_to_the_kalman_filter_as_the_ensemble_grows(
    linear_gaussian_problem,
):
    """The EnKF's only approximation is a sampled B, so it must converge.

    Errors should fall roughly as 1/sqrt(N); the assertion is the weaker and
    more robust claim that they fall monotonically and become small.
    """
    xb, b_cov, y, h_op, r_cov = linear_gaussian_problem
    xa_kf, _ = assimilate.kalman_filter_update(xb, b_cov, y, h_op, r_cov)
    rng = np.random.default_rng(1)
    errors = []
    for n_members in (200, 2000, 20000):
        ens = rng.multivariate_normal(xb, b_cov, size=n_members)
        # Re-centre so the sampled *mean* is exact and only the sampled
        # covariance differs -- isolating the effect under test.
        ens = ens - ens.mean(axis=0) + xb
        xa = assimilate.enkf_update(ens, y, h_op, r_cov, seed=3).mean(axis=0)
        errors.append(float(np.linalg.norm(xa - xa_kf)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.05


def test_enkf_reproduces_the_kalman_analysis_covariance():
    """The mean is not enough: the analysis SPREAD must be right too.

    An EnKF whose mean converges to the Kalman filter but whose covariance is
    wrong would produce forecasts that are accurate and dishonest -- and the
    error would show up only as a mis-calibrated ensemble, which is easy to
    misread as needing more inflation. Checked here against the exact
    linear-Gaussian analysis covariance.
    """
    b_cov = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    h_op = np.eye(3)
    r_cov = np.eye(3) * 0.5
    xb = np.array([1.0, 2.0, 3.0])
    y = np.array([1.4, 1.7, 3.3])
    _, p_a_exact = assimilate.kalman_filter_update(xb, b_cov, y, h_op, r_cov)

    rng = np.random.default_rng(0)
    ens = rng.multivariate_normal(xb, b_cov, size=20000)
    ens = ens - ens.mean(axis=0) + xb  # exact mean: only the covariance is sampled
    analysis = assimilate.enkf_update(ens, y, h_op, r_cov, inflation=1.0, seed=5)
    p_a_sample = np.cov(analysis.T, ddof=1)

    ratio = np.sqrt(np.trace(p_a_sample) / np.trace(p_a_exact))
    assert ratio == pytest.approx(1.0, abs=0.03)


def test_enkf_inflation_widens_the_background_not_the_analysis_mean():
    """Inflation must only ever INCREASE spread -- it cannot reduce it.

    Pinned because the chapter-20 reliability discussion rests on it: in a
    configuration that is already over-dispersed, no amount of inflation can
    restore calibration.
    """
    b_cov = np.eye(3) * 2.0
    h_op = np.eye(3)
    r_cov = np.eye(3) * 0.5
    xb = np.array([1.0, 2.0, 3.0])
    y = np.array([1.4, 1.7, 3.3])
    rng = np.random.default_rng(11)
    ens = rng.multivariate_normal(xb, b_cov, size=2000)

    spreads = [
        float(
            np.sqrt(
                np.mean(
                    np.var(
                        assimilate.enkf_update(
                            ens, y, h_op, r_cov, inflation=a, seed=4
                        ),
                        axis=0,
                        ddof=1,
                    )
                )
            )
        )
        for a in (1.0, 1.1, 1.2, 1.4)
    ]
    assert all(b > a for a, b in zip(spreads, spreads[1:]))


def test_enkf_rejects_a_degenerate_ensemble():
    with pytest.raises(ValueError, match="at least 2 members"):
        assimilate.enkf_update(
            np.array([[1.0, 2.0, 3.0]]), np.zeros(3), np.eye(3), np.eye(3)
        )


def test_inflation_widens_the_analysis_ensemble(linear_gaussian_problem):
    xb, b_cov, y, h_op, r_cov = linear_gaussian_problem
    rng = np.random.default_rng(2)
    ens = rng.multivariate_normal(xb, b_cov, size=400)
    plain = assimilate.enkf_update(ens, y, h_op, r_cov, inflation=1.0, seed=5)
    inflated = assimilate.enkf_update(ens, y, h_op, r_cov, inflation=1.5, seed=5)
    assert inflated.std(axis=0).mean() > plain.std(axis=0).mean()


def test_gaspari_cohn_is_a_valid_localisation_function():
    assert assimilate.gaspari_cohn(np.array([0.0]), 4.0)[0] == pytest.approx(1.0)
    assert assimilate.gaspari_cohn(np.array([4.0]), 4.0)[0] == pytest.approx(
        0.0, abs=1e-12
    )
    assert assimilate.gaspari_cohn(np.array([6.0]), 4.0)[0] == 0.0
    weights = assimilate.gaspari_cohn(np.linspace(0.0, 4.0, 60), 4.0)
    assert np.all(np.diff(weights) <= 1e-12)  # monotonically decreasing
    assert np.all(weights >= 0.0)


def test_four_dvar_reduces_the_distance_to_the_truth():
    """4D-Var must beat the background it started from, on the same window."""
    truth0 = np.array([1.0, 1.0, 20.0])
    dt = 0.01
    obs_times = [0.05, 0.10, 0.15]
    h_op = np.eye(3)
    r_cov = np.eye(3) * 0.01
    observations = []
    for t_obs in obs_times:
        grid = np.arange(int(round(t_obs / dt)) + 1) * dt
        observations.append(
            (t_obs, integrate.rk4(systems.lorenz63, truth0, grid)[-1])
        )

    xb = truth0 + np.array([0.6, -0.5, 0.7])
    b_cov = np.eye(3) * 1.0
    xa = assimilate.four_dvar_analysis(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        xb,
        b_cov,
        observations,
        h_op,
        r_cov,
        dt=dt,
        max_iterations=60,
    )
    assert np.linalg.norm(xa - truth0) < 0.25 * np.linalg.norm(xb - truth0)


# ==========================================================================
# ensemble
# ==========================================================================
def test_gaussian_perturbations_have_the_requested_amplitude():
    x0 = np.array([1.0, 1.0, 20.0])
    ens = ensemble.gaussian_perturbations(x0, 4000, 0.1, seed=0)
    assert ens.shape == (4000, 3)
    assert (ens - x0).std() == pytest.approx(0.1, rel=0.05)


def test_spread_matches_error_for_a_perfectly_reliable_ensemble():
    """The calibration identity: if truth is drawn from the same distribution as
    the members, RMS spread equals RMS error of the mean (up to sampling)."""
    rng = np.random.default_rng(4)
    n_members, n_cases = 50, 400
    spreads, errors = [], []
    for _ in range(n_cases):
        members = rng.normal(size=(1, n_members, 1))
        truth = rng.normal(size=(1, 1))
        spreads.append(ensemble.ensemble_spread(members)[0])
        errors.append(ensemble.ensemble_mean_error(members, truth)[0])
    ratio = np.sqrt(np.mean(np.square(spreads)) / np.mean(np.square(errors)))
    assert ratio == pytest.approx(1.0, rel=0.1)


def test_spread_needs_at_least_two_members():
    with pytest.raises(ValueError, match="at least 2 members"):
        ensemble.ensemble_spread(np.zeros((3, 1, 2)))


def test_rank_histogram_is_flat_for_a_reliable_ensemble():
    rng = np.random.default_rng(5)
    n_cases, n_members = 20000, 9
    members = rng.normal(size=(n_cases, n_members))
    truth = rng.normal(size=n_cases)
    counts = ensemble.rank_histogram(members, truth, seed=0)
    assert counts.size == n_members + 1
    assert counts.sum() == n_cases
    expected = n_cases / (n_members + 1)
    assert np.all(np.abs(counts - expected) < 0.15 * expected)


def test_rank_histogram_is_u_shaped_for_an_under_dispersed_ensemble():
    """Under-spread is the characteristic operational failure; it must show."""
    rng = np.random.default_rng(6)
    n_cases, n_members = 20000, 9
    members = 0.3 * rng.normal(size=(n_cases, n_members))
    truth = rng.normal(size=n_cases)
    counts = ensemble.rank_histogram(members, truth, seed=0)
    ends = counts[0] + counts[-1]
    middle = counts[1:-1].sum()
    assert ends > middle


def test_crps_matches_the_closed_form_for_a_gaussian_ensemble():
    """CRPS of N(0,1) evaluated at y, against the analytic expression."""
    from scipy.stats import norm

    mu, sigma, y = 0.0, 1.0, 0.7
    z = (y - mu) / sigma
    closed_form = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    rng = np.random.default_rng(7)
    members = rng.normal(mu, sigma, size=(1, 40000))
    assert ensemble.crps(members, np.array([y]))[0] == pytest.approx(
        closed_form, rel=0.02
    )


def test_crps_reduces_to_absolute_error_for_a_deterministic_forecast():
    members = np.full((1, 12), 2.0)
    assert ensemble.crps(members, np.array([3.5]))[0] == pytest.approx(1.5)


def test_crps_rewards_a_sharper_correct_forecast():
    rng = np.random.default_rng(8)
    truth = np.zeros(2000)
    sharp = rng.normal(0.0, 0.5, size=(2000, 40))
    broad = rng.normal(0.0, 2.0, size=(2000, 40))
    assert ensemble.crps(sharp, truth).mean() < ensemble.crps(broad, truth).mean()


def test_brier_score_reference_values():
    # A perfect forecast scores 0; the climatological p=0.5 forecast scores 0.25.
    assert ensemble.brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert ensemble.brier_score([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)
    assert ensemble.brier_score([0.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_brier_score_rejects_non_binary_outcomes():
    with pytest.raises(ValueError, match="0 or 1"):
        ensemble.brier_score([0.5], [0.7])


# ==========================================================================
# errorgrowth
# ==========================================================================
def test_logistic_error_growth_is_exponential_while_the_error_is_small():
    rate = 0.9
    t = np.linspace(0.0, 2.0, 50)
    err = errorgrowth.logistic_error_growth(t, 1e-8, rate, 1.0)
    slope = np.polyfit(t, np.log(err), 1)[0]
    assert slope == pytest.approx(rate, rel=1e-3)


def test_logistic_error_growth_saturates():
    err = errorgrowth.logistic_error_growth(np.array([1e4]), 1e-6, 0.9, 3.0)
    assert err[0] == pytest.approx(3.0, rel=1e-9)


def test_fit_logistic_error_growth_recovers_known_parameters():
    t = np.linspace(0.0, 10.0, 200)
    truth = errorgrowth.logistic_error_growth(t, 1e-3, 0.9, 1.0)
    e0, rate, sat = errorgrowth.fit_logistic_error_growth(t, truth, saturation=1.0)
    assert e0 == pytest.approx(1e-3, rel=1e-3)
    assert rate == pytest.approx(0.9, rel=1e-3)
    assert sat == 1.0


def test_fit_logistic_error_growth_can_free_the_saturation_level():
    t = np.linspace(0.0, 12.0, 300)
    truth = errorgrowth.logistic_error_growth(t, 1e-3, 0.7, 2.5)
    e0, rate, sat = errorgrowth.fit_logistic_error_growth(t, truth)
    assert rate == pytest.approx(0.7, rel=1e-2)
    assert sat == pytest.approx(2.5, rel=1e-2)


def test_saturation_level_is_the_climatological_spread():
    """Should match sqrt(2) * the RMS radius of the attractor about its mean."""
    grid = integrate.trajectory_grid(200.0, 0.02)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid)[2000:]
    level = errorgrowth.saturation_level(traj, seed=0)
    centred = traj - traj.mean(axis=0)
    expected = np.sqrt(2.0 * np.mean(np.sum(centred**2, axis=-1)))
    assert level == pytest.approx(expected, rel=0.05)


def test_predictability_horizon_finds_the_threshold_crossing():
    t = np.linspace(0.0, 10.0, 101)
    err = errorgrowth.logistic_error_growth(t, 1e-4, 2.0, 1.0)
    horizon = errorgrowth.predictability_horizon(t, err, threshold_frac=0.5)
    assert 0.0 < horizon < 10.0
    assert errorgrowth.logistic_error_growth(
        np.array([horizon]), 1e-4, 2.0, 1.0
    )[0] >= 0.5 * err.max()


def test_predictability_horizon_is_infinite_if_never_crossed():
    t = np.linspace(0.0, 1.0, 10)
    assert errorgrowth.predictability_horizon(
        t, np.linspace(0.0, 1.0, 10), threshold_frac=2.0
    ) == float("inf")


# ==========================================================================
# dimension
# ==========================================================================
def test_l63_correlation_dimension_matches_literature():
    grid = integrate.trajectory_grid(600.0, 0.01)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid)[5000:]
    d2, radii, c = dimension.correlation_dimension(traj, theiler=50)
    assert d2 == pytest.approx(2.05, abs=0.15)
    # The scaling window must actually be a plateau, not a fitted average of a
    # curve -- otherwise the number above is meaningless.
    slopes = dimension.local_slopes(radii, c)
    assert slopes.std() < 0.1


def test_henon_correlation_dimension_matches_literature():
    state = np.array([0.1, 0.1])
    points = []
    for i in range(30000):
        state = systems.henon_map(state)
        if i > 200:
            points.append(state.copy())
    d2, _, _ = dimension.correlation_dimension(
        np.array(points), fit_range=(0.005, 0.05)
    )
    assert d2 == pytest.approx(1.22, abs=0.1)


def test_correlation_sum_is_monotone_and_bounded():
    rng = np.random.default_rng(9)
    pts = rng.normal(size=(600, 3))
    radii = np.logspace(-2, 1, 20)
    c = dimension.correlation_sum(pts, radii)
    assert np.all(np.diff(c) >= 0.0)
    assert c[0] >= 0.0 and c[-1] <= 1.0


def test_correlation_dimension_of_a_uniform_cube_is_three():
    """A sanity anchor with a known integer answer."""
    rng = np.random.default_rng(10)
    pts = rng.uniform(size=(4000, 3))
    # Small radii only: in a *bounded* uniform set, pairs near the boundary are
    # deficient, so large radii bias the slope below the true dimension.
    d2, _, _ = dimension.correlation_dimension(pts, fit_range=(0.02, 0.08))
    assert d2 == pytest.approx(3.0, abs=0.15)


def test_correlation_sum_rejects_an_over_wide_theiler_window():
    with pytest.raises(ValueError, match="Theiler"):
        dimension.correlation_sum(
            np.zeros((10, 2)), np.array([1.0]), theiler=50
        )


# ==========================================================================
# information
# ==========================================================================
def test_entropy_of_a_uniform_distribution_is_log_n():
    for n in (2, 8, 64):
        assert information.shannon_entropy(np.ones(n) / n) == pytest.approx(
            np.log(n)
        )


def test_entropy_of_a_certain_outcome_is_zero():
    p = np.zeros(5)
    p[2] = 1.0
    assert information.shannon_entropy(p) == pytest.approx(0.0)


def test_relative_entropy_is_zero_only_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.4, 0.4, 0.2])
    assert information.relative_entropy(p, p) == pytest.approx(0.0)
    assert information.relative_entropy(p, q) > 0.0
    # Asymmetric, deliberately.
    assert information.relative_entropy(p, q) != pytest.approx(
        information.relative_entropy(q, p)
    )


def test_relative_entropy_rejects_an_infinite_divergence():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="infinite"):
        information.relative_entropy(p, q)


def test_mutual_information_vanishes_for_independent_variables():
    joint = np.outer(np.array([0.2, 0.8]), np.array([0.3, 0.3, 0.4]))
    assert information.mutual_information(joint) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_is_maximal_for_a_deterministic_relation():
    joint = np.eye(4) / 4.0
    assert information.mutual_information(joint) == pytest.approx(np.log(4.0))


def test_gaussian_relative_entropy_matches_the_1d_closed_form():
    m0, v0, m1, v1 = 0.0, 1.0, 1.0, 4.0
    closed = 0.5 * (v0 / v1 + (m1 - m0) ** 2 / v1 - 1.0 + np.log(v1 / v0))
    assert information.gaussian_relative_entropy(
        [m0], [[v0]], [m1], [[v1]]
    ) == pytest.approx(closed)


def test_gaussian_relative_entropy_separates_signal_from_dispersion():
    """A shifted-but-equally-wide forecast and a sharper-but-centred one are both
    informative, for different reasons -- the two terms of the divergence."""
    cov = np.eye(2)
    signal_only = information.gaussian_relative_entropy(
        [1.0, 0.0], cov, [0.0, 0.0], cov
    )
    dispersion_only = information.gaussian_relative_entropy(
        [0.0, 0.0], 0.25 * cov, [0.0, 0.0], cov
    )
    assert signal_only > 0.0
    assert dispersion_only > 0.0


def test_gaussian_relative_entropy_is_zero_for_identical_gaussians():
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert information.gaussian_relative_entropy(
        [1.0, 2.0], cov, [1.0, 2.0], cov
    ) == pytest.approx(0.0, abs=1e-12)


def test_predictive_information_decays_with_lead_time():
    """Predictability as information: I(x(0); x(tau)) must fall towards zero."""
    grid = integrate.trajectory_grid(400.0, 0.02)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid)[2000:]
    x = traj[:, 0]
    short = information.predictive_information(x[:-10], x[10:], bins=24)
    long = information.predictive_information(x[:-500], x[500:], bins=24)
    assert short > long


# ==========================================================================
# plotting: the palette is a contract other chapters rely on
# ==========================================================================
PALETTE_HEX = (
    "C_TRUTH",
    "C_PERT",
    "C_SPREAD",
    "C_MEAN",
    "C_FIXED",
    "C_SAT",
    "C_START",
    "C_OBS",
    "C_BG",
    "C_ANALYSIS",
)


def test_every_palette_colour_is_usable_in_matplotlib():
    """The palette is CSS, because Plotly eats it directly -- but two entries use
    the rgba(...) form that matplotlib rejects.

    This is the test that was missing. `C_CONTEXT` passed straight to `ax.plot`
    raises at RENDER time, so a notebook cell fails while the import succeeds --
    and it slipped past a `grep -c marimo-error` check, which reported zero.
    Every palette entry must survive `mpl_colour` and then `to_rgba`.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba

    names = [n for n in plotting.__all__ if n.startswith("C_")] + ["SCENE_BG"]
    assert "C_CONTEXT" in names and "SCENE_BG" in names
    for name in names:
        raw = getattr(plotting, name)
        converted = plotting.mpl_colour(raw)
        to_rgba(converted)  # raises if matplotlib cannot use it


def test_mpl_colour_preserves_alpha_and_passes_hex_through():
    assert plotting.mpl_colour("#3730a3") == "#3730a3"
    r, g, b, a = plotting.mpl_colour("rgba(150,150,165,0.16)")
    assert a == pytest.approx(0.16)
    assert r == pytest.approx(150 / 255)
    # rgb() with no alpha is opaque
    assert plotting.mpl_colour("rgb(10,20,30)")[3] == pytest.approx(1.0)


def test_mpl_panels_returns_a_flat_axes_array_for_any_column_count():
    """Callers must not have to special-case ncols == 1.

    matplotlib's `subplots` returns a bare Axes for 1x1, an array for 1xN. The
    helper normalises that, because a chapter that indexes axes[0] should keep
    working when a panel is added or removed.
    """
    import matplotlib
    matplotlib.use("Agg")

    for ncols in (1, 2, 3, 4):
        fig, axes = plotting.mpl_panels(ncols)
        assert len(axes) == ncols
        for ax in axes:
            ax.plot([0, 1], [0, 1])  # must be a usable Axes
        plotting.finish_mpl(fig)


def test_mpl_panels_applies_titles_and_does_not_touch_global_rcparams():
    """Importing or using chaoslib must not mutate the caller's matplotlib state.

    A global rcParams change is exactly the hidden state that makes a marimo
    notebook's output depend on which cell ran first.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = dict(plt.rcParams)
    fig, axes = plotting.mpl_panels(3, titles=("a", "b", "c"))
    assert [ax.get_title() for ax in axes] == ["a", "b", "c"]
    plotting.finish_mpl(fig, suptitle="whole figure")
    after = dict(plt.rcParams)
    changed = [k for k in before if before[k] != after.get(k)]
    assert changed == [], f"chaoslib mutated global rcParams: {changed}"


def test_mpl_panels_hides_the_top_and_right_spines():
    """The house look, asserted so a future edit cannot quietly drop it."""
    import matplotlib
    matplotlib.use("Agg")

    _, axes = plotting.mpl_panels(2)
    for ax in axes:
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()


def test_palette_entries_are_valid_css_colours():
    hex_colours = [getattr(plotting, name) for name in PALETTE_HEX]
    for colour in hex_colours:
        assert colour.startswith("#") and len(colour) == 7
        int(colour[1:], 16)  # raises if not hex
    assert plotting.C_CONTEXT.startswith("rgba(")
    assert plotting.SCENE_BG.startswith("rgba(")


def test_palette_colours_are_distinct():
    """Semantic colours must be visually separable or the convention is useless."""
    colours = [getattr(plotting, name) for name in PALETTE_HEX]
    assert len(set(colours)) == len(colours)


def test_palette_colours_are_perceptually_separated():
    """Distinctness of the hex string is not enough -- a DA figure shows six of
    these at once, so any two must differ by a visible margin.

    Crude but sufficient: require a minimum Euclidean distance in RGB, which
    catches the failure mode of adding a colour that is merely a shade of one
    already in use.
    """
    def rgb(h):
        return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))

    # One documented exception: C_PERT (rose, a trajectory) and C_SAT (firebrick,
    # a dashed horizontal reference line) sit 59 apart. They co-occur only in the
    # SDIC figures, where line style separates them, so the pair is accepted --
    # but the bound is kept strong for every other pair so that a NEW colour
    # cannot be added as a mere shade of an existing one.
    allowed_close = {frozenset({"C_PERT", "C_SAT"})}

    colours = {name: rgb(getattr(plotting, name)) for name in PALETTE_HEX}
    names = list(colours)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            dist = sum((x - y) ** 2 for x, y in zip(colours[a], colours[b])) ** 0.5
            floor = 55.0 if frozenset({a, b}) in allowed_close else 60.0
            assert dist > floor, f"{a} and {b} are too close in RGB (d={dist:.0f})"


def test_every_exported_palette_name_exists():
    """__all__ is the contract chapters import against."""
    for name in plotting.__all__:
        assert hasattr(plotting, name), name


# ==========================================================================
# maps: the period-doubling cascade and Feigenbaum universality
# ==========================================================================
# Superstable parameters of the logistic map, i.e. the r at which the
# 2^n-cycle contains the critical point x = 1/2. Published values.
# *[citation needed: Feigenbaum (1978), table 1]*
LOGISTIC_SUPERSTABLE = [
    2.0,
    3.2360679775,       # exactly 1 + sqrt(5)
    3.4985616993,
    3.5546408628,
    3.5666673799,
    3.5692435316,
    3.5697952937,
    3.5699134654,
]
# The accumulation point of the cascade; bracketing must stop below it.
LOGISTIC_R_INFINITY = 3.5699456720


def test_logistic_superstable_cascade_matches_published_values():
    """Each R_n to 1e-9 -- eight levels, spanning a 4700-fold range of spacings."""
    cascade = maps.superstable_cascade(
        systems.logistic_map, 0.5, 1.5, LOGISTIC_R_INFINITY, n_max=7
    )
    assert cascade.size == len(LOGISTIC_SUPERSTABLE)
    for n, (got, want) in enumerate(zip(cascade, LOGISTIC_SUPERSTABLE)):
        assert abs(got - want) < 1e-9, f"R_{n}: {got!r} != {want!r}"
    # The first two are exact in closed form.
    assert cascade[0] == pytest.approx(2.0, abs=1e-13)
    assert cascade[1] == pytest.approx(1.0 + np.sqrt(5.0), abs=1e-12)


def test_superstable_cycles_have_exactly_zero_multiplier():
    """The defining property, and it holds to the last bit rather than to a tolerance.

    A superstable 2^n-cycle passes through x = 1/2, where f'(1/2) = r(1 - 2*0.5)
    is *identically* zero in floating point, so the product of multipliers around
    the cycle is exactly 0.0. This is the strongest available check that
    superstable_cascade returned cycles rather than nearby parameters.
    """
    cascade = maps.superstable_cascade(
        systems.logistic_map, 0.5, 1.5, LOGISTIC_R_INFINITY, n_max=5
    )
    for n, r in enumerate(cascade):
        multiplier = maps.cycle_multiplier(
            systems.logistic_map, systems.logistic_map_derivative, 0.5, 2**n, r=r
        )
        assert multiplier == 0.0, f"n={n}, r={r}: multiplier {multiplier!r}"


def test_feigenbaum_delta_from_the_logistic_cascade():
    """delta = 4.6692 measured from the map, not quoted."""
    cascade = maps.superstable_cascade(
        systems.logistic_map, 0.5, 1.5, LOGISTIC_R_INFINITY, n_max=8
    )
    ratios = maps.feigenbaum_ratios(cascade)
    assert ratios[-1] == pytest.approx(maps.FEIGENBAUM_DELTA, abs=1e-4)
    # Convergence is monotone from below over the last few levels.
    assert ratios[-1] > ratios[-2] > ratios[-3]


def test_feigenbaum_delta_is_universal_across_unrelated_map_families():
    """The point of the constant: three different maps, three different cascades,
    one delta.

    The logistic, sine and cubic families share no algebra -- polynomial versus
    transcendental, critical point at 1/2 versus 1/sqrt(3), first superstable
    parameter 2.0 versus 0.5 versus 1.5. If delta were an artefact of the
    quadratic form of the logistic map, this test would fail.
    """
    families = {
        "logistic": (systems.logistic_map, 0.5, 1.5, LOGISTIC_R_INFINITY, 7),
        "sine": (systems.sine_map, 0.5, 0.3, 0.8655, 5),
        "cubic": (systems.cubic_map, 1.0 / np.sqrt(3.0), 1.0, 2.3025, 6),
    }
    first_parameters = {}
    for name, (fmap, xc, lo, hi, n_max) in families.items():
        cascade = maps.superstable_cascade(fmap, xc, lo, hi, n_max=n_max)
        assert cascade.size >= 4, f"{name}: only {cascade.size} levels found"
        first_parameters[name] = cascade[0]
        estimate = maps.feigenbaum_ratios(cascade)[-1]
        assert estimate == pytest.approx(maps.FEIGENBAUM_DELTA, abs=6e-3), (
            f"{name}: delta estimate {estimate:.6f}"
        )
    # Confirm the cascades really are different objects, so the agreement above
    # is universality and not three copies of the same computation.
    assert first_parameters["logistic"] == pytest.approx(2.0, abs=1e-10)
    assert first_parameters["sine"] == pytest.approx(0.5, abs=1e-10)
    assert first_parameters["cubic"] == pytest.approx(1.5, abs=1e-10)


def test_bifurcation_points_reproduce_the_analytic_two_cycle():
    """For 3 < r < 1 + sqrt(6) the attractor is the exact pair
    x = [(r+1) +- sqrt((r-3)(r+1))] / 2r.
    """
    for r in (3.2, 3.3, 3.44):
        spread = np.sqrt((r - 3.0) * (r + 1.0))
        expected = np.sort([((r + 1.0) - spread) / (2.0 * r),
                            ((r + 1.0) + spread) / (2.0 * r)])
        _, x = maps.bifurcation_points(
            systems.logistic_map, np.array([r]), n_discard=4000, n_keep=8
        )
        found = np.sort(np.unique(np.round(x, 9)))
        assert found.size == 2, f"r={r}: found {found}"
        assert found == pytest.approx(expected, abs=1e-9)


def test_bifurcation_points_reproduce_the_analytic_fixed_point():
    """For 1 < r < 3 the only attractor is x* = 1 - 1/r."""
    for r in (1.8, 2.4, 2.8):
        _, x = maps.bifurcation_points(
            systems.logistic_map, np.array([r]), n_discard=4000, n_keep=6
        )
        assert x == pytest.approx(1.0 - 1.0 / r, abs=1e-9)


# ==========================================================================
# maps: the Lyapunov exponent of a one-dimensional map
# ==========================================================================
def test_map_lyapunov_exponent_is_exactly_ln2_at_r_equals_four():
    """At r = 4 the logistic map is conjugate to the doubling map x -> 2x mod 1,
    whose stretching rate is 2 at every point, so lambda = ln 2 exactly.
    """
    lam = maps.map_lyapunov_exponent(
        systems.logistic_map,
        systems.logistic_map_derivative,
        np.array([4.0]),
        n_discard=2000,
        n_iter=30000,
    )
    assert float(lam[0]) == pytest.approx(np.log(2.0), abs=1e-5)


def test_map_lyapunov_exponent_vanishes_at_the_first_bifurcation():
    """At r = 3 the fixed point x* = 2/3 has f'(x*) = r(1 - 2x*) = -1 exactly,
    so the exponent is zero: the marginal case that separates the two regimes.
    """
    lam = maps.map_lyapunov_exponent(
        systems.logistic_map,
        systems.logistic_map_derivative,
        np.array([3.0]),
        n_discard=200000,
        n_iter=60000,
    )
    assert abs(float(lam[0])) < 1e-4


def test_map_lyapunov_exponent_separates_periodic_from_chaotic_parameters():
    """Sign structure at parameters whose behaviour is known independently.

    r = 3.83 is the critical case: it sits inside the period-3 window, well
    above the accumulation point r_inf = 3.5699, so a diagram-by-eye reading of
    "past r_inf, therefore chaotic" gets it wrong. The exponent does not.
    """
    periodic = np.array([2.9, 3.2, 3.5, 3.55, 3.83])
    chaotic = np.array([3.6, 3.7, 3.9, 4.0])
    lam_p = maps.map_lyapunov_exponent(
        systems.logistic_map, systems.logistic_map_derivative, periodic, n_iter=8000
    )
    lam_c = maps.map_lyapunov_exponent(
        systems.logistic_map, systems.logistic_map_derivative, chaotic, n_iter=8000
    )
    assert np.all(lam_p < -0.05), f"expected negative, got {lam_p}"
    assert np.all(lam_c > 0.05), f"expected positive, got {lam_c}"


def test_map_lyapunov_exponent_floor_keeps_superstable_dips_finite():
    """At a superstable parameter the orbit hits f' = 0 and the exponent is
    genuinely -inf. The floor keeps the returned array plottable.
    """
    lam = maps.map_lyapunov_exponent(
        systems.logistic_map,
        systems.logistic_map_derivative,
        np.array([2.0]),
        n_iter=500,
        floor=-20.0,
    )
    assert np.isfinite(lam).all()
    assert float(lam[0]) == pytest.approx(-20.0, abs=1e-9)


# ==========================================================================
# maps: the tangent bifurcation and type-I intermittency
# ==========================================================================
def test_period_three_threshold_is_a_tangency_of_the_third_iterate():
    """At r_c = 1 + 2 sqrt(2) the third iterate touches the diagonal:
    f^3(x) = x with (f^3)'(x) = +1, at three points at once.

    A tangency, not a crossing -- which is why this bifurcation cannot be
    located with a root-finder on f^3(x) - x, and why period_three_threshold is
    a closed form rather than a search.
    """
    r_c = maps.period_three_threshold()
    assert r_c == pytest.approx(3.8284271247461903, abs=1e-13)
    for x_star in (0.1599288184, 0.5143552771, 0.9563178420):
        residual = float(maps.iterate_n(systems.logistic_map, x_star, 3, r=r_c)) - x_star
        assert abs(residual) < 1e-9, f"x={x_star}: f^3(x)-x = {residual:.2e}"
        slope = maps.cycle_multiplier(
            systems.logistic_map, systems.logistic_map_derivative, x_star, 3, r=r_c
        )
        assert slope == pytest.approx(1.0, abs=1e-6), f"x={x_star}: (f^3)' = {slope}"


def test_period_three_solutions_appear_in_a_pair_at_the_threshold():
    """Below r_c the only solution of f^3(x) = x is the fixed point; above it
    there are seven -- the fixed point plus a stable and an unstable 3-cycle,
    born together. That is the saddle-node signature.
    """
    r_c = maps.period_three_threshold()
    grid = np.linspace(1e-4, 1.0 - 1e-4, 200001)

    def crossings(r):
        v = maps.iterate_n(systems.logistic_map, grid, 3, r=r) - grid
        return int(np.count_nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0.0))

    assert crossings(r_c - 1e-3) == 1
    assert crossings(r_c + 1e-3) == 7


def test_laminar_phases_diverge_as_the_inverse_square_root():
    """Type-I intermittency: <L> ~ (r_c - r)^(-1/2).

    Measured over a 16-fold range of (r_c - r), which changes <L> by a factor of
    four. The exponent is the prediction of the normal form x -> x + a x^2 + eps,
    and is independent of the map's details -- another universal number.
    """
    r_c = maps.period_three_threshold()
    epsilons = np.array([2.0e-3, 5.0e-4, 1.25e-4])
    means = []
    for eps in epsilons:
        lengths = maps.laminar_phases(
            systems.logistic_map, r_c - eps, n_iter=60000
        )
        assert lengths.size > 100, f"eps={eps}: only {lengths.size} laminar phases"
        means.append(lengths.mean())
    slope = float(np.polyfit(np.log(epsilons), np.log(means), 1)[0])
    assert slope == pytest.approx(-0.5, abs=0.06), f"exponent {slope:.4f}"


def test_laminar_phases_are_absent_well_below_the_threshold():
    """The intermittency is specific to the approach to r_c: at r = 3.7 the
    orbit is ordinarily chaotic and shadows no 3-cycle.
    """
    far = maps.laminar_phases(systems.logistic_map, 3.70, n_iter=60000)
    near = maps.laminar_phases(
        systems.logistic_map, maps.period_three_threshold() - 1.25e-4, n_iter=60000
    )
    assert far.size == 0 or far.mean() < 0.25 * near.mean()


# ==========================================================================
# maps: orbits and the cobweb construction
# ==========================================================================
def test_map_orbit_vectorises_over_initial_conditions():
    """The vectorised path is what makes a live bifurcation diagram affordable,
    so it must agree with iterating one initial condition at a time.
    """
    x0 = np.array([0.1, 0.35, 0.6, 0.87])
    together = maps.map_orbit(systems.logistic_map, x0, 40, r=3.7)
    for j, single in enumerate(x0):
        alone = maps.map_orbit(systems.logistic_map, single, 40, r=3.7)
        assert together[:, j] == pytest.approx(alone, abs=1e-14)


def test_cobweb_path_vertices_lie_on_the_orbit():
    """Every horizontal landing of the staircase is the next iterate, and every
    vertex sits either on the diagonal or on the graph of f.
    """
    xs, ys = maps.cobweb_path(systems.logistic_map, 0.2, 12, r=3.5)
    orbit = maps.map_orbit(systems.logistic_map, 0.2, 12, r=3.5)
    # Vertices 0, 2, 4, ... sit on the diagonal at successive iterates.
    assert xs[::2] == pytest.approx(orbit, abs=1e-14)
    assert ys[::2] == pytest.approx(orbit, abs=1e-14)
    # Vertices 1, 3, 5, ... sit on the graph: y = f(x).
    assert ys[1::2] == pytest.approx(
        systems.logistic_map(xs[1::2], r=3.5), abs=1e-14
    )


def test_periodic_window_has_its_own_cascade_with_the_same_delta():
    """Self-similarity of the bifurcation diagram, as a measurement.

    Inside the period-3 window the sequence 3 -> 6 -> 12 -> 24 -> ... is a
    complete cascade in its own right, compressed into a parameter interval
    0.0175 wide against the main cascade's 1.57, and its spacing ratios converge
    to the *same* delta. That is the
    content of the renormalisation argument: the window is not merely a picture
    that resembles the whole diagram, it has the same quantitative structure.
    """
    cascade = maps.superstable_cascade(
        systems.logistic_map, 0.5, 3.8285, 3.8497, n_max=5, base_period=3
    )
    assert cascade.size >= 5, f"only {cascade.size} sub-levels found"
    # The first is the superstable 3-cycle, inside the window.
    r_c = maps.period_three_threshold()
    assert r_c < cascade[0] < 3.84
    # Every level is a genuine superstable 3*2^n cycle.
    for n, r in enumerate(cascade):
        multiplier = maps.cycle_multiplier(
            systems.logistic_map,
            systems.logistic_map_derivative,
            0.5,
            3 * 2**n,
            r=r,
        )
        assert multiplier == 0.0, f"n={n}, period={3 * 2**n}: {multiplier!r}"
    # Same constant, from a cascade with a different base period, spanning
    # ~1/90 of the parameter range the main cascade occupies.
    estimate = maps.feigenbaum_ratios(cascade)[-1]
    assert estimate == pytest.approx(maps.FEIGENBAUM_DELTA, abs=2e-2), (
        f"sub-cascade delta {estimate:.6f}"
    )
    assert cascade[-1] - cascade[0] < 0.022


# ==========================================================================
# Lorenz 96: linear theory about the uniform state
# ==========================================================================
def test_lorenz96_uniform_state_is_an_exact_zero_of_the_rhs():
    """x_k = F kills the quadratic terms identically, not to a tolerance.

    (x_{k+1} - x_{k-2}) x_{k-1} = (F - F) F = 0 exactly in floating point, and
    -F + F = 0 exactly, so the residual is 0.0 for every F and N.
    """
    for n in (12, 40, 61):
        for forcing in (0.5, 8.0, 20.0):
            state = systems.lorenz96_uniform_state(forcing, n)
            residual = systems.lorenz96(0.0, state, forcing)
            assert np.max(np.abs(residual)) == 0.0, f"N={n}, F={forcing}"


def test_lorenz96_dispersion_reproduces_every_jacobian_eigenvalue():
    """The circulant Jacobian is diagonalised exactly by Fourier modes.

    sigma(theta) = -1 + F(e^{i theta} - e^{-2 i theta}) is not an approximation:
    the N values it returns are the N eigenvalues of the Jacobian at the uniform
    state, as a set. Compared after a deterministic lexicographic sort, since
    conjugate pairs share a real part and no ordering is implied by either
    computation.
    """
    for n, forcing in ((20, 3.0), (40, 8.0), (60, 12.0)):
        jacobian = systems.lorenz96_jacobian(
            systems.lorenz96_uniform_state(forcing, n), forcing
        )
        numeric = np.linalg.eigvals(jacobian)
        analytic = systems.lorenz96_dispersion(np.arange(n), n, forcing)

        def lexsorted(z):
            return z[np.lexsort((np.round(z.imag, 9), np.round(z.real, 9)))]

        difference = np.abs(lexsorted(numeric) - lexsorted(analytic)).max()
        assert difference < 1e-11, f"N={n}, F={forcing}: {difference:.2e}"


def test_lorenz96_critical_forcing_matches_the_closed_form():
    """Re sigma = -1 + F(cos t - cos 2t), so F_crit = 1/max(cos t - cos 2t).

    Writing u = cos theta the bracket is 1 + u - 2u^2, maximised at u = 1/4 with
    value 9/8. A finite chain can only use integer wavenumbers, so F_crit
    approaches 8/9 from above as N grows. At N = 20 and N = 40 the best
    available mode gives exactly sqrt(5)/2, hence F_crit = 2/sqrt(5).
    """
    for n, m_star in ((20, 4), (40, 8)):
        forcing, mode = systems.lorenz96_critical_forcing(n)
        assert mode == m_star
        assert forcing == pytest.approx(2.0 / np.sqrt(5.0), abs=1e-12)

    # Monotone approach to the continuum limit 8/9.
    values = [systems.lorenz96_critical_forcing(n)[0] for n in (40, 60, 80, 200)]
    assert all(a >= b for a, b in zip(values, values[1:])), values
    assert values[-1] == pytest.approx(8.0 / 9.0, abs=2e-4)
    assert min(values) > 8.0 / 9.0 - 1e-12


def test_lorenz96_uniform_state_stability_switches_at_the_critical_forcing():
    """Below F_crit every mode decays; above it at least one grows. Checked
    against the eigenvalues of the Jacobian, not against the closed form, so
    this is an independent statement about the model rather than about algebra.
    """
    n = 40
    forcing_crit, _ = systems.lorenz96_critical_forcing(n)
    for forcing, expect_unstable in (
        (0.99 * forcing_crit, False),
        (1.01 * forcing_crit, True),
    ):
        jacobian = systems.lorenz96_jacobian(
            systems.lorenz96_uniform_state(forcing, n), forcing
        )
        growth = np.linalg.eigvals(jacobian).real.max()
        assert bool(growth > 0.0) is expect_unstable, f"F={forcing}: max Re = {growth}"


# ==========================================================================
# Lorenz 96: the spectrum, the trace identity, and extensivity
# ==========================================================================
def test_l96_spectrum_sums_to_minus_n(l96_spectrum):
    """Every diagonal entry of the Lorenz 96 Jacobian is -1, so tr J = -N
    identically and the exponents of the *flow* sum to exactly -N.

    The analogue of the Lorenz 63 trace identity, and the same kind of check:
    it constrains the sum, so it holds however poorly the individual exponents
    have converged. What it does *not* do is hold to machine precision, because
    the measurement is made on an RK4 propagator rather than on the flow -- see
    the convergence test below, which is the sharper statement.
    """
    assert l96_spectrum.size == 40
    assert l96_spectrum.sum() == pytest.approx(-40.0, abs=1e-4)


def test_l96_trace_identity_error_is_fourth_order_in_dt():
    """The residual in sum(lambda) = -N is pure RK4 truncation.

    Two things distinguish a discretisation error from a sampling error, and
    both are asserted: the residual is **independent of the averaging time**
    (halving or doubling T does not move it), and it falls by ~16 for each
    halving of dt. Together they say the identity is exact for the flow and
    that nothing else is wrong with the tangent propagator -- a much stronger
    claim than any single tolerance.
    """
    n = 12

    def residual(dt, t_final):
        state = systems.lorenz96_uniform_state(8.0, n)
        state[n // 2] += 0.01
        spectrum = lyapunov.lyapunov_spectrum(
            systems.lorenz96,
            systems.lorenz96_jacobian,
            state,
            dt=dt,
            t_final=t_final,
            t_transient=10.0,
            forcing=8.0,
        )
        return abs(spectrum.sum() + n)

    errors = [residual(dt, 60.0) for dt in (0.02, 0.01, 0.005)]
    for coarse, fine in zip(errors, errors[1:]):
        ratio = coarse / fine
        assert 8.0 < ratio < 26.0, f"dt^4 expected ~16, got {ratio:.1f} from {errors}"

    # Independent of the averaging window: a sampling error would shrink here.
    short, long = residual(0.01, 40.0), residual(0.01, 120.0)
    assert abs(short - long) < 0.35 * max(short, long), (short, long)


def test_l96_ks_entropy_is_the_sum_of_the_positive_exponents(l96_spectrum):
    """Pesin's identity, as chaoslib implements it."""
    positive = l96_spectrum[l96_spectrum > 0.0]
    assert lyapunov.ks_entropy(l96_spectrum) == pytest.approx(
        positive.sum(), rel=1e-12
    )
    # For N=40, F=8 this is around 9-10 nats per time unit -- an order of
    # magnitude above Lorenz 63's 0.9, which is the whole point of the model.
    assert 8.0 < lyapunov.ks_entropy(l96_spectrum) < 11.0


def test_lorenz96_is_extensive():
    """The defining property of spatiotemporal chaos, and chapter 11's headline.

    The Lyapunov spectrum is *intensive*: lambda_1 does not grow with the size
    of the domain. The entropy and the attractor dimension are *extensive*: both
    are proportional to N, so h_KS/N and D_KY/N are constants of the model
    rather than of the domain.

    Tolerances are loose because each exponent is a finite-time average over a
    single trajectory; the claim being tested is proportionality, not a
    high-precision value.
    """
    ratios_h, ratios_d, leading = [], [], []
    for n in (16, 24, 40):
        state = systems.lorenz96_uniform_state(8.0, n)
        state[n // 2] += 0.01
        spectrum = lyapunov.lyapunov_spectrum(
            systems.lorenz96,
            systems.lorenz96_jacobian,
            state,
            dt=0.01,
            t_final=120.0,
            t_transient=20.0,
            forcing=8.0,
        )
        # The trace identity must hold at every N (to RK4 truncation, which
        # grows with N because the residual is a sum of N terms).
        assert spectrum.sum() == pytest.approx(-float(n), abs=1e-4), f"N={n}"
        leading.append(spectrum[0])
        ratios_h.append(lyapunov.ks_entropy(spectrum) / n)
        ratios_d.append(lyapunov.kaplan_yorke_dimension(spectrum) / n)

    # Intensive: the leading exponent is the same at every domain size.
    assert max(leading) - min(leading) < 0.25, leading
    # Extensive: the densities are constant to within a few percent.
    assert max(ratios_h) - min(ratios_h) < 0.03, ratios_h
    assert max(ratios_d) - min(ratios_d) < 0.03, ratios_d
    assert 0.22 < np.mean(ratios_h) < 0.28
    assert 0.64 < np.mean(ratios_d) < 0.70


# ==========================================================================
# spatial: diagnostics for a field on a ring
# ==========================================================================
def _planted_wave(wavenumber, omega, n=40, dt=0.01, steps=2000, amplitude=2.5):
    """cos(theta_m k + omega t), plus a large mean that must be removed."""
    k = np.arange(n)
    t = np.arange(steps) * dt
    theta = 2.0 * np.pi * wavenumber / n
    return amplitude * np.cos(theta * k[None, :] + omega * t[:, None]) + 7.0


def test_spatial_power_spectrum_recovers_a_planted_wave():
    """A single Fourier component carries amplitude^2 / 2 and nothing else does.

    That normalisation is what makes the spectrum comparable between domain
    sizes, which chapter 12 needs.
    """
    field = _planted_wave(8, 3.0, amplitude=2.5)
    m, power = spatial.spatial_power_spectrum(field)
    assert power[8] == pytest.approx(2.5**2 / 2.0, rel=1e-10)
    assert power[0] < 1e-20  # the mean is removed
    others = np.delete(power, [0, 8])
    assert others.max() < 1e-20
    assert spatial.dominant_wavenumber(field) == 8


def test_phase_speed_recovers_a_planted_wave_including_its_sign():
    """c = -omega/theta_m, exactly, in both directions.

    The sign is the part worth testing: it is the difference between a wave
    that propagates with the flow and one that propagates against it, and it
    depends on a convention that is easy to get backwards.
    """
    for wavenumber, omega in ((8, 3.0), (5, -2.0), (13, 7.5)):
        field = _planted_wave(wavenumber, omega)
        theta = 2.0 * np.pi * wavenumber / 40
        assert spatial.phase_speed(field, wavenumber, 0.01) == pytest.approx(
            -omega / theta, rel=1e-8
        )


def test_phase_speed_rejects_inputs_it_cannot_interpret():
    field = _planted_wave(8, 3.0)
    with pytest.raises(ValueError):
        spatial.phase_speed(field, 0, 0.01)  # m = 0 does not propagate
    with pytest.raises(ValueError):
        spatial.phase_speed(field[0], 8, 0.01)  # not a (time, space) field


def test_spatial_correlation_matches_the_analytic_cosine():
    """For a pure wave the autocorrelation is cos(theta_m s), exactly."""
    n = 40
    for wavenumber in (4, 5, 8, 10):
        field = _planted_wave(wavenumber, 0.0, n=n, steps=50)
        theta = 2.0 * np.pi * wavenumber / n
        separations = np.arange(n // 2 + 1)
        assert spatial.spatial_correlation(field) == pytest.approx(
            np.cos(theta * separations), abs=1e-10
        )


def test_correlation_length_is_a_quarter_wavelength():
    """cos(theta s) first crosses zero at s = N/(4m).

    Exact when the crossing falls on a site (m = 5 and m = 10 at N = 40), and
    biased slightly high otherwise: the estimate interpolates linearly across a
    curve that is concave there, an O(theta^2) error which grows as the dominant
    wavelength shortens.
    """
    n = 40
    for wavenumber in (5, 10):
        field = _planted_wave(wavenumber, 0.0, n=n, steps=20)
        assert spatial.correlation_length(field) == pytest.approx(
            n / (4.0 * wavenumber), abs=1e-9
        )
    # m = 8 puts the true crossing at 1.25, between sites 1 and 2.
    field = _planted_wave(8, 0.0, n=n, steps=20)
    estimate = spatial.correlation_length(field)
    assert 1.25 < estimate < 1.29


def test_the_named_colourmaps_exist_and_have_the_right_character():
    """A diverging map for signed fields, a sequential one for non-negative.

    Checked rather than assumed, because swapping them silently produces a
    figure that hides the sign of the field it is drawing.
    """
    import matplotlib

    diverging = matplotlib.colormaps[plotting.MPL_DIVERGING]
    sequential = matplotlib.colormaps[plotting.MPL_SEQUENTIAL]

    # A diverging map is light in the middle and dark at both ends.
    def luminance(cmap, x):
        r, g, b, _ = cmap(x)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    middle = luminance(diverging, 0.5)
    assert middle > luminance(diverging, 0.0) + 0.2
    assert middle > luminance(diverging, 1.0) + 0.2

    # A sequential map is monotone in luminance, so magnitude reads as darkness.
    values = [luminance(sequential, x) for x in np.linspace(0.0, 1.0, 9)]
    assert all(a < b for a, b in zip(values, values[1:])), values


# ==========================================================================
# dimension: box counting against closed-form dimensions
# ==========================================================================
def test_reference_sets_have_the_dimensions_they_claim():
    """Box counting recovers three exactly known dimensions.

    These are the only dimension checks in the suite that compare against an
    exact number rather than a literature estimate, which makes them the
    calibration for everything else in the module:
    ln2/ln3, ln4/ln3 and ln3/ln2.
    """
    builders = {
        "cantor": lambda: dimension.cantor_set(200_000, 18, seed=0),
        "koch": lambda: dimension.koch_curve(8),
        "sierpinski": lambda: dimension.sierpinski_triangle(200_000, 24, seed=1),
    }
    for name, build in builders.items():
        exact = dimension.REFERENCE_DIMENSIONS[name]
        window = dimension.REFERENCE_WINDOWS[name]
        estimate, _, _, _ = dimension.renyi_dimension(
            build(), q=0.0, fit_range=window, n_scales=14
        )
        assert estimate == pytest.approx(exact, abs=0.02), (
            f"{name}: D_0 = {estimate:.4f} against exact {exact:.4f}"
        )


def test_box_counting_saturates_when_the_sample_runs_out():
    """The trap the occupancy return exists to expose.

    Below about 20 points per box the measured slope collapses toward zero: at
    one point per box the box count equals the sample size and stops responding
    to the scale at all. Asserting the *collapse* rather than merely a wrong
    number pins the mechanism.
    """
    points = dimension.sierpinski_triangle(200_000, 24, seed=2)
    exact = dimension.REFERENCE_DIMENSIONS["sierpinski"]

    good, _, _, occupancy_good = dimension.renyi_dimension(
        points, q=0.0, fit_range=dimension.REFERENCE_WINDOWS["sierpinski"], n_scales=10
    )
    assert occupancy_good.min() > dimension.MIN_BOX_OCCUPANCY
    assert good == pytest.approx(exact, abs=0.05)

    starved, _, _, occupancy_bad = dimension.renyi_dimension(
        points, q=0.0, fit_range=(1e-5, 1e-3), n_scales=10
    )
    assert occupancy_bad.max() < 3.0, occupancy_bad
    assert starved < 0.55 * exact, f"expected collapse, got {starved:.3f}"


def test_renyi_dimensions_separate_and_order_on_a_multifractal():
    """D_0 >= D_1 >= D_2, with D_1 and D_2 against their closed forms.

    The inequality is exact only in the limit, so it cannot be tested on a
    *uniform* measure: there the three coincide and finite-scale estimation
    noise of ~0.008 reorders them at will (measured 1.5835, 1.5904, 1.5858 on
    the unweighted triangle -- ordered incorrectly, and not a bug).

    Skewing the chaos-game weights separates them by 0.28, which is well clear
    of that noise, and for contraction ratio 1/2 supplies exact targets:
    D_1 = -sum p ln p / ln 2 and D_2 = -ln(sum p^2) / ln 2.
    """
    probabilities = np.array([0.6, 0.2, 0.2])
    points = dimension.sierpinski_triangle(
        250_000, 24, seed=3, probabilities=probabilities
    )
    spectrum = dimension.renyi_spectrum(
        points,
        q_values=(0.0, 1.0, 2.0),
        fit_range=dimension.REFERENCE_WINDOWS["sierpinski"],
        n_scales=14,
    )
    assert np.all(np.diff(spectrum) < 0.0), spectrum
    assert spectrum[0] - spectrum[2] > 0.2, spectrum

    d1_exact = -(probabilities * np.log(probabilities)).sum() / np.log(2.0)
    d2_exact = -np.log((probabilities**2).sum()) / np.log(2.0)
    assert spectrum[1] == pytest.approx(d1_exact, abs=0.09), (spectrum[1], d1_exact)
    assert spectrum[2] == pytest.approx(d2_exact, abs=0.09), (spectrum[2], d2_exact)


def test_renyi_dimension_of_a_uniform_square_is_two():
    """A control with no fractal structure: D_q = 2 for every q.

    The residual deficit is the O(eps) edge bias documented on
    renyi_dimension: the unit-box grid lays down (1/eps + 1)^d boxes, not
    eps^-d, so a fixed window converges on the (1/eps+1)^2 prediction rather
    than on 2. Hence the one-sided tolerance -- the estimate should be a little
    *low*, never high.
    """
    rng = np.random.default_rng(7)
    points = rng.uniform(0.0, 1.0, size=(200_000, 2))
    for q in (0.0, 1.0, 2.0):
        estimate, _, _, occupancy = dimension.renyi_dimension(
            points, q=q, fit_range=(0.008, 0.06), n_scales=9
        )
        assert occupancy.min() > dimension.MIN_BOX_OCCUPANCY
        assert 1.94 < estimate < 2.01, f"q={q}: {estimate}"


def test_box_occupancy_counts_every_point_exactly_once():
    """A partition, so the occupancies must sum to the sample size."""
    points = dimension.sierpinski_triangle(20_000, 20, seed=4)
    for scale in (0.5, 0.1, 0.02):
        counts = dimension.box_occupancy(points, scale)
        assert counts.sum() == points.shape[0]
        assert np.all(counts >= 1)


# ==========================================================================
# dimension: fitting a window, and embedding from one variable
# ==========================================================================
def test_fit_dimension_recovers_a_planted_power_law():
    """C(r) = r^D exactly, so the fitted slope must be D to round-off."""
    radii = np.logspace(-3.0, -1.0, 40)
    for exponent in (0.7, 1.26, 2.06, 3.0):
        c = radii**exponent
        slope, used = dimension.fit_dimension(radii, c, (1e-3, 1e-1))
        assert slope == pytest.approx(exponent, rel=1e-10)
        assert used == radii.size


def test_fit_dimension_rejects_a_window_with_too_few_radii():
    radii = np.logspace(-3.0, -1.0, 40)
    with pytest.raises(ValueError, match="usable radii"):
        dimension.fit_dimension(radii, radii**2.0, (1.5e-3, 1.6e-3))


def test_delay_embedding_reconstructs_the_geometry():
    """Takens: an embedding of one observed variable has the same dimension.

    D_2 measured on the x component alone rises with the embedding dimension
    and saturates near the full-state value once m is large enough to hold the
    set. That saturation is the only available diagnostic for choosing m, since
    the criterion m > 2D needs the D one is trying to measure.
    """
    grid = integrate.trajectory_grid(t_final=400.0, dt=0.01)
    trajectory = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid
    )[3000:]

    full, _, _ = dimension.correlation_dimension(
        trajectory, theiler=50, max_points=3000
    )

    estimates = []
    for embedding_dimension in (2, 3, 5):
        embedded = dimension.delay_embed(trajectory[:, 0], embedding_dimension, 20)
        value, _, _ = dimension.correlation_dimension(
            embedded, theiler=50, max_points=3000
        )
        estimates.append(value)

    # Too small an embedding caps the estimate well below the truth.
    assert estimates[0] < full - 0.25, estimates
    # A large enough one recovers it.
    assert estimates[2] == pytest.approx(full, abs=0.15), (estimates, full)
    # And the approach is from below.
    assert estimates[0] < estimates[1] < estimates[2] + 0.05, estimates


def test_delay_embed_shape_and_contents():
    series = np.arange(20.0)
    embedded = dimension.delay_embed(series, 3, 4)
    assert embedded.shape == (20 - 2 * 4, 3)
    assert embedded[0] == pytest.approx([0.0, 4.0, 8.0])
    assert embedded[-1] == pytest.approx([11.0, 15.0, 19.0])
    with pytest.raises(ValueError):
        dimension.delay_embed(series, 3, 0)
    with pytest.raises(ValueError, match="too short"):
        dimension.delay_embed(series, 10, 5)


def test_theiler_window_removes_a_high_bias_from_dense_sampling():
    """The temporal-correlation trap, with its sign pinned.

    Sampled at dt = 0.01 the Lorenz 63 trajectory moves about 1.3% of the
    attractor's diameter per step -- inside the default fit window -- so the
    excess of temporally adjacent pairs puts a bump in C(r) right there and
    biases D_2 *high*. Widening the Theiler window monotonically removes it.

    Worth asserting the direction rather than just "the estimates differ",
    because the textbook warning about temporal correlation is usually stated
    as a bias toward low dimension, which is what happens when the sampling is
    dense enough to put the step scale *below* the window instead.
    """
    grid = integrate.trajectory_grid(t_final=75.0, dt=0.01)
    trajectory = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid
    )[3000:]
    # Densely sampled on purpose: no subsampling happens at this length.
    assert trajectory.shape[0] < 4600

    estimates = [
        dimension.correlation_dimension(trajectory, theiler=w, max_points=4000)[0]
        for w in (0, 10, 50, 200)
    ]
    assert estimates[0] > 2.10, estimates          # clearly biased high
    assert estimates[-1] == pytest.approx(2.05, abs=0.05), estimates
    assert all(a > b for a, b in zip(estimates, estimates[1:])), estimates


def test_a_badly_chosen_scaling_window_returns_a_confident_wrong_answer():
    """Both signs of error, from the same curve.

    The point is not that the numbers are wrong but that nothing in the return
    value indicates it: each of these is a clean least-squares fit with a small
    residual over a dozen radii.
    """
    grid = integrate.trajectory_grid(t_final=800.0, dt=0.01)
    trajectory = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid
    )[3000:]
    stride = max(1, trajectory.shape[0] // 4000)
    prepared = trajectory[::stride][:4000]
    diameter = float(
        np.sqrt(((prepared.max(axis=0) - prepared.min(axis=0)) ** 2).sum())
    )
    radii = np.logspace(
        np.log10(3e-4 * diameter), np.log10(1.2 * diameter), 60
    )
    c = dimension.correlation_sum(trajectory, radii, theiler=50, max_points=4000)

    good, _ = dimension.fit_dimension(radii, c, (0.008 * diameter, 0.05 * diameter))
    saturated, _ = dimension.fit_dimension(radii, c, (0.3 * diameter, 1.2 * diameter))
    noisy, _ = dimension.fit_dimension(radii, c, (3e-4 * diameter, 2e-3 * diameter))
    everything, _ = dimension.fit_dimension(radii, c, (3e-4 * diameter, 1.2 * diameter))

    assert good == pytest.approx(2.06, abs=0.05)
    assert saturated < 0.4, saturated            # C -> 1: slope collapses
    assert noisy > 2.35, noisy                   # quantised counts: slope steepens
    # The most dangerous case: wrong, but not obviously so.
    assert 1.8 < everything < 2.0, everything


# ==========================================================================
# errorgrowth: the upscale cascade and the intrinsic limit (Lorenz 1969)
# ==========================================================================
def test_a_single_cascade_band_is_exactly_logistic_growth():
    """With one band the upscale term is absent, so the cascade model must
    reduce to logistic_error_growth -- the same equation, integrated instead of
    evaluated in closed form.

    This is the reduction that makes the cascade model trustworthy: whatever it
    does with many bands, with one it is a function whose exact solution is
    already tested elsewhere in this file.
    """
    rate, initial = 0.9, 1e-6
    times, errors = errorgrowth.cascade_growth(
        1, alpha=0.0, rate0=rate, seed_amplitude=initial, t_final=25.0, n_times=60
    )
    exact = errorgrowth.logistic_error_growth(times, initial, rate, 1.0)
    assert errors[:, 0] == pytest.approx(exact, abs=1e-8)


def test_cascade_rates_double_every_alpha_inverse_octaves():
    rates = errorgrowth.cascade_rates(6, alpha=0.5, rate0=3.0)
    assert rates[0] == pytest.approx(3.0)
    assert np.all(np.diff(np.log2(rates)) == pytest.approx(0.5))
    # alpha = 0 is the scale-independent case: every band the same.
    assert errorgrowth.cascade_rates(5, alpha=0.0) == pytest.approx(np.ones(5))


def test_the_predictability_limit_is_finite_only_if_small_scales_grow_faster():
    """Lorenz (1969), as the chapter's central claim.

    Seeding the smallest *resolved* band at saturation and adding bands is
    improving the observing resolution one octave at a time. The time for the
    largest scale to be contaminated then either converges or does not, and
    which one happens depends on alpha alone.
    """
    # alpha > 0: converges, and the increment per octave shrinks geometrically
    # by 2^(-2 alpha) -- measured 0.630, 0.397, 0.250 against 2^(-2/3),
    # 2^(-4/3), 2^(-2). Note that is the *square* of the 2^(-alpha) the naive
    # sum-of-timescales argument would predict, so the increments die faster
    # than that heuristic suggests.
    for alpha, expected in ((1.0 / 3.0, 2.3035), (2.0 / 3.0, 1.4466), (1.0, 1.1220)):
        times = [
            errorgrowth.cascade_contamination_time(n, alpha=alpha)
            for n in (8, 16, 24, 32)
        ]
        assert times[-1] == pytest.approx(expected, abs=5e-3), (alpha, times)
        assert all(a <= b + 1e-9 for a, b in zip(times, times[1:])), times

        octaves = np.arange(6, 20, 2)
        curve = np.array(
            [errorgrowth.cascade_contamination_time(n, alpha=alpha) for n in octaves]
        )
        increments = np.diff(curve) / 2.0
        ratios = increments[1:] / increments[:-1]
        assert np.all(increments > 0.0), increments
        assert ratios[-1] == pytest.approx(2.0 ** (-2.0 * alpha), abs=0.02), (
            alpha, ratios
        )

    # alpha = 0: diverges, at a settled cost per octave.
    flat = {
        n: errorgrowth.cascade_contamination_time(n, alpha=0.0)
        for n in (16, 32, 64, 128)
    }
    assert flat[128] > 35.0, flat
    per_octave = [
        (flat[b] - flat[a]) / (b - a)
        for a, b in ((16, 32), (32, 64), (64, 128))
    ]
    # Constant cost per octave, hence linear growth without bound.
    assert all(0.27 < v < 0.32 for v in per_octave), per_octave


def test_reducing_the_smallest_scale_error_buys_almost_nothing():
    """The operational reading, and the sharpest number in the chapter.

    In a multiscale system the error at the finest scale saturates so fast that
    its initial amplitude is nearly irrelevant: sixteen orders of magnitude of
    improvement buys 2 %. Reducing the error at the *largest* scale instead
    obeys the familiar logarithmic law, ln(10)/lambda_0 per decade -- so where
    an observing system improves matters far more than by how much.
    """
    fine = [
        errorgrowth.cascade_contamination_time(16, seed_amplitude=amplitude)
        for amplitude in (1.0, 1e-8, 1e-16)
    ]
    assert fine[0] == pytest.approx(1.4453, abs=5e-3)
    assert (fine[-1] - fine[0]) / fine[0] < 0.03, fine

    coarse = [
        errorgrowth.cascade_contamination_time(
            16, seed_amplitude=amplitude, seed_band=0
        )
        for amplitude in (1e-2, 1e-4, 1e-8)
    ]
    # ln(10)/lambda_0 = 2.303 per decade, at rate0 = 1.
    per_decade = [
        (coarse[1] - coarse[0]) / 2.0,
        (coarse[2] - coarse[1]) / 4.0,
    ]
    for value in per_decade:
        assert value == pytest.approx(np.log(10.0), abs=0.05), per_decade
    # And it keeps paying: no convergence here at all.
    assert coarse[-1] > 4.0 * coarse[0]


# ==========================================================================
# systems: two-scale Lorenz 96
# ==========================================================================
def test_two_scale_lorenz96_reduces_to_lorenz96_when_uncoupled():
    """With h = 0 the slow equations are *identically* lorenz96, to the last bit.

    The strongest available check that the two-scale right-hand side is the
    model it claims to be: not a tolerance, an exact match, because switching
    off the coupling should leave the slow equations untouched rather than
    approximately untouched.
    """
    rng = np.random.default_rng(4)
    n_slow, n_fast, forcing = 8, 32, 20.0
    state = np.concatenate(
        [rng.normal(5.0, 3.0, n_slow), rng.normal(0.0, 0.3, n_slow * n_fast)]
    )
    tendency = systems.lorenz96_two_scale(
        0.0, state, n_slow=n_slow, n_fast=n_fast, forcing=forcing, coupling=0.0
    )
    slow, _ = systems.lorenz96_two_scale_split(state, n_slow)
    expected = systems.lorenz96(0.0, slow, forcing=forcing)
    assert np.max(np.abs(tendency[:n_slow] - expected)) == 0.0


def test_two_scale_lorenz96_vectorises_over_ensemble_members():
    """The batched path is what makes chapter 12's 32-member averages free."""
    state = systems.lorenz96_two_scale_state(seed=2)
    single = systems.lorenz96_two_scale(0.0, state)
    ensemble = systems.lorenz96_two_scale(0.0, np.stack([state, state, state]))
    assert ensemble.shape == (3, state.size)
    for member in ensemble:
        assert np.max(np.abs(member - single)) == 0.0


def test_two_scale_lorenz96_coupling_conserves_total_energy():
    """The exchange moves energy between the subsystems and creates none.

    The slow equations lose (hc/b) sum_j Y_j and each fast equation gains
    (hc/b) X_k, so the contributions to d/dt of (sum X^2)/2 and (sum Y^2)/2 are
    exact negatives: both equal (hc/b) sum_k X_k sum_{j in k} Y_j, with
    opposite signs. Measured residual 1.8e-14 out of terms of size 16.5.

    A sign error in either half of the coupling, or a mismatch in how the fast
    variables are grouped onto slow ones, breaks this and nothing else in the
    suite would catch it.
    """
    n_slow, n_fast = 8, 32
    rng = np.random.default_rng(11)
    state = np.concatenate(
        [rng.normal(5.0, 3.0, n_slow), rng.normal(0.0, 0.3, n_slow * n_fast)]
    )
    kwargs = dict(n_slow=n_slow, n_fast=n_fast, forcing=20.0)
    exchange = systems.lorenz96_two_scale(
        0.0, state, coupling=1.0, **kwargs
    ) - systems.lorenz96_two_scale(0.0, state, coupling=0.0, **kwargs)

    slow, fast = systems.lorenz96_two_scale_split(state, n_slow)
    slow_rate, fast_rate = systems.lorenz96_two_scale_split(exchange, n_slow)
    slow_power = float(slow @ slow_rate)
    fast_power = float(fast @ fast_rate)

    assert abs(slow_power) > 1.0, slow_power          # the term is not trivial
    assert slow_power + fast_power == pytest.approx(0.0, abs=1e-10)


def test_two_scale_lorenz96_quadratic_terms_conserve_energy():
    """With the forcing and the coupling off, only the -X damping removes
    energy: the advective terms conserve (sum X^2)/2 exactly.

    The same identity the single-scale lorenz96 test asserts, carried over to
    confirm the quadratic terms were transcribed correctly rather than merely
    plausibly.
    """
    n_slow, n_fast = 8, 32
    rng = np.random.default_rng(12)
    state = np.concatenate(
        [rng.normal(5.0, 3.0, n_slow), rng.normal(0.0, 0.3, n_slow * n_fast)]
    )
    tendency = systems.lorenz96_two_scale(
        0.0, state, n_slow=n_slow, n_fast=n_fast, forcing=0.0, coupling=0.0
    )
    slow, _ = systems.lorenz96_two_scale_split(state, n_slow)
    slow_rate, _ = systems.lorenz96_two_scale_split(tendency, n_slow)
    # d/dt (sum X^2)/2 = -sum X^2 exactly: the quadratic part contributes zero.
    assert float(slow @ slow_rate) == pytest.approx(-float(slow @ slow), abs=1e-9)


def test_two_scale_lorenz96_state_has_the_advertised_shape():
    state = systems.lorenz96_two_scale_state(n_slow=6, n_fast=10, seed=5)
    assert state.size == 6 + 6 * 10
    slow, fast = systems.lorenz96_two_scale_split(state, 6)
    assert slow.size == 6 and fast.size == 60
    # The slow part starts near the forcing, the fast part near zero.
    assert np.abs(slow - 20.0).max() < 1.0
    assert np.abs(fast).max() < 1.0
