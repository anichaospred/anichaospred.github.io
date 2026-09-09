r"""A learned emulator, trainable in closed form: the echo state network.

The question chapter 29 asks is whether a model *fitted* to a chaotic system
inherits its dynamics -- its Lyapunov spectrum, its unstable dimension, its
error-growth law -- or only its short-term forecasts. Answering it needs an
emulator that can be trained inside a browser, which rules out gradient
descent, and one whose tangent map is available analytically, which rules out
treating the emulator as a black box.

A **reservoir computer** satisfies both. A fixed random recurrent network is
driven by the data,

.. math:: \mathbf{r}_{k+1} = \tanh\!\bigl(\mathbf{W}\mathbf{r}_k
          + \mathbf{W}_{\rm in}\,\nu(\mathbf{u}_k)\bigr),

and only a **linear readout** is fitted, by ridge regression:

.. math:: \mathbf{u}_{k+1} = \mathbf{u}_k
          + \bigl[\mathbf{r}_{k+1},\, \mathbf{r}_{k+1}^2,\,
                  \nu(\mathbf{u}_k),\, 1\bigr]\,\mathbf{W}_{\rm out}.

Training is one linear solve -- no optimiser, no learning rate, no stopping
criterion, and the same answer every time. The nonlinearity is real and lives
in the reservoir; the squared term breaks the odd symmetry of :math:`\tanh`,
without which the readout cannot represent a system whose statistics are not
symmetric.

Three things this module is careful about, each of which cost a measurement to
find:

* **Inputs are standardised.** Feeding Lorenz 96 (mean 2.4, spread 3.7)
  straight into :math:`\tanh` saturates the reservoir and costs two orders of
  magnitude in one-step accuracy. :func:`standardiser` exists to make that
  hard to forget.
* **The readout predicts the increment**, not the state. The state is mostly
  persistence, so fitting it flatters the model and wastes the readout's
  capacity on reproducing the identity.
* **Driving and rolling out are different operations, and confusing them is
  silent.** :func:`reservoir_drive` feeds the *data* in, which is how a
  reservoir is synchronised to a known state; :func:`esn_rollout` feeds the
  emulator's *own predictions* back, which is a forecast. Using a rollout to
  "warm up" before scoring a forecast synchronises the reservoir to a
  trajectory that has already left the data, and the result still looks like
  Lorenz 96 -- it merely starts somewhere else, so the measured skill is
  meaningless. That mistake produced two wrong answers while chapter 29 was
  being written and neither of them looked wrong.
* **The spectral radius must be below one.** That is the echo state property,
  and it is a stability condition rather than a tuning preference: at radius
  1.4 the free-running rollout diverges after 949 steps with a variance 273
  times the truth's (chapter 29, section 2).

The emulator's own Lyapunov spectrum is computed by
:func:`esn_lyapunov_spectrum` from the analytic tangent in
:func:`esn_tangent_apply`, which is checked against finite differences in the
tests. That matters because the headline result is a comparison of spectra, and
a tangent that was subtly wrong would produce a wrong spectrum that looked
entirely plausible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "standardiser",
    "standardise",
    "echo_state_network",
    "reservoir_drive",
    "readout_features",
    "fit_readout",
    "esn_step",
    "esn_rollout",
    "esn_tangent_apply",
    "esn_lyapunov_spectrum",
]


def standardiser(data: Array) -> dict:
    """Mean and spread of the training data, as a single scalar pair.

    A *scalar* mean and spread, not per-component: the systems here are
    translation-invariant along the site axis, so per-site normalisation would
    invent structure the dynamics do not have.
    """
    values = np.asarray(data, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std())}


def standardise(u: Array, scaling: dict) -> Array:
    """Apply :func:`standardiser`'s scaling."""
    return (np.asarray(u, dtype=float) - scaling["mean"]) / scaling["sd"]


def echo_state_network(
    n_input: int,
    n_reservoir: int = 1000,
    spectral_radius: float = 0.4,
    input_scaling: float = 0.15,
    sparsity: float = 0.03,
    seed: int | None = 1,
) -> dict:
    r"""A fixed random reservoir, rescaled to an exact spectral radius.

    Returns ``{"W", "W_in", "n_reservoir", "n_input", "spectral_radius"}``.
    ``W`` is sparse in the sense that a fraction ``sparsity`` of its entries
    are non-zero, then scaled so that :math:`\rho(\mathbf{W})` is exactly
    ``spectral_radius`` -- the tests assert that to machine precision, because
    it is the parameter the stability of the whole thing turns on.

    Nothing here is trained. The reservoir is the same random matrix whatever
    data arrives, which is what makes the fit a single linear solve.
    """
    rng = np.random.default_rng(seed)
    n_reservoir = int(n_reservoir)
    weights = rng.normal(0.0, 1.0, (n_reservoir, n_reservoir))
    weights *= rng.random((n_reservoir, n_reservoir)) < float(sparsity)
    largest = np.max(np.abs(np.linalg.eigvals(weights)))
    if largest > 0.0:
        weights *= float(spectral_radius) / largest
    return {
        "W": weights,
        "W_in": rng.uniform(
            -float(input_scaling), float(input_scaling),
            (n_reservoir, int(n_input)),
        ),
        "n_reservoir": n_reservoir,
        "n_input": int(n_input),
        "spectral_radius": float(spectral_radius),
    }


def reservoir_drive(
    network: dict, inputs: Array, scaling: dict, initial: Array | None = None
) -> Array:
    """Drive the reservoir with a sequence of inputs, returning every state.

    The returned array is ``(n_steps, n_reservoir)`` and row ``k`` is the state
    *after* absorbing input ``k`` -- the convention :func:`fit_readout` and
    :func:`esn_step` both assume.
    """
    inputs = np.asarray(inputs, dtype=float)
    n_reservoir = network["n_reservoir"]
    state = (
        np.zeros(n_reservoir) if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    out = np.empty((inputs.shape[0], n_reservoir))
    weights, w_in = network["W"], network["W_in"]
    for index in range(inputs.shape[0]):
        state = np.tanh(
            weights @ state + w_in @ standardise(inputs[index], scaling)
        )
        out[index] = state
    return out


def readout_features(
    reservoir_state: Array, u: Array, scaling: dict
) -> Array:
    r"""The readout's design row: :math:`[\mathbf{r},\, \mathbf{r}^2,\,
    \nu(\mathbf{u}),\, 1]`.

    The squared term is not decoration. :math:`\tanh` is odd, so a readout
    linear in :math:`\mathbf{r}` alone can only represent a map with a symmetry
    the target does not have -- and Lorenz 96 has a mean of 2.4.
    """
    reservoir_state = np.asarray(reservoir_state, dtype=float)
    u = np.asarray(u, dtype=float)
    ones = np.ones(u.shape[:-1] + (1,))
    return np.concatenate(
        [reservoir_state, reservoir_state**2, standardise(u, scaling), ones],
        axis=-1,
    )


def fit_readout(
    reservoir_states: Array,
    inputs: Array,
    targets: Array,
    scaling: dict,
    ridge: float = 1.0e-5,
) -> Array:
    r"""Ridge-regress ``targets`` on the readout features. One linear solve.

    .. math:: \mathbf{W}_{\rm out}
              = \bigl(\mathbf{F}^\top\mathbf{F}
                + \gamma \mathbf{I}\bigr)^{-1}\mathbf{F}^\top\mathbf{Y}

    ``targets`` should be the **increment** :math:`\mathbf{u}_{k+1} -
    \mathbf{u}_k`, not the next state: predicting the state means spending most
    of the readout's capacity reproducing the identity, and it makes the fitted
    error look far better than the model is.

    With ``ridge = 0`` this is ordinary least squares and recovers an exactly
    linear target exactly, which the tests use. The design matrix is
    ill-conditioned by construction -- :math:`10^{10}` and worse -- so the
    regularisation is doing real work and not only preventing overfitting.
    """
    features = readout_features(
        np.asarray(reservoir_states, dtype=float),
        np.asarray(inputs, dtype=float),
        scaling,
    )
    gram = features.T @ features + float(ridge) * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ np.asarray(targets, dtype=float))


def esn_step(
    network: dict,
    readout: Array,
    reservoir_state: Array,
    u: Array,
    scaling: dict,
) -> tuple[Array, Array]:
    """One autonomous step: absorb ``u``, then predict its increment.

    Returns ``(new_reservoir_state, new_u)``. Autonomous because the prediction
    is fed back as the next input, which is what makes the emulator a dynamical
    system in its own right -- and what makes its Lyapunov spectrum a
    meaningful thing to ask about.
    """
    new_state = np.tanh(
        network["W"] @ np.asarray(reservoir_state, dtype=float)
        + network["W_in"] @ standardise(u, scaling)
    )
    increment = readout_features(new_state, u, scaling) @ readout
    return new_state, np.asarray(u, dtype=float) + increment


def esn_rollout(
    network: dict,
    readout: Array,
    reservoir_state: Array,
    u0: Array,
    n_steps: int,
    scaling: dict,
    blow_up: float = 1.0e4,
) -> tuple[Array, Array]:
    """Free-running forecast for ``n_steps``, stopping early if it diverges.

    Returns ``(trajectory, final_reservoir_state)``. The trajectory is
    ``(k+1, n_input)`` where ``k`` is however many steps survived, so a short
    return value **is** the diagnostic: a rollout that stops early has blown
    up, and the caller should check its length rather than assume it.
    """
    state = np.asarray(reservoir_state, dtype=float).copy()
    u = np.asarray(u0, dtype=float).copy()
    out = np.empty((int(n_steps) + 1, u.size))
    out[0] = u
    for index in range(int(n_steps)):
        state, u = esn_step(network, readout, state, u, scaling)
        if not np.isfinite(u).all() or np.abs(u).max() > float(blow_up):
            return out[: index + 1], state
        out[index + 1] = u
    return out, state


def esn_tangent_apply(
    network: dict,
    readout: Array,
    new_reservoir_state: Array,
    tangent: Array,
    scaling: dict,
) -> Array:
    r"""Apply the emulator's Jacobian to a block of tangent vectors.

    ``tangent`` has shape ``(n_reservoir + n_input, k)`` and
    ``new_reservoir_state`` is the reservoir state **after** the step whose
    Jacobian is wanted. Returns the propagated block.

    Differentiating :func:`esn_step`, with
    :math:`\mathbf{s} = 1 - \mathbf{r}'^2` and :math:`\sigma` the standardising
    spread:

    .. math::
        \delta\mathbf{r}' &= \mathbf{s} \odot
            \bigl(\mathbf{W}\,\delta\mathbf{r}
            + \mathbf{W}_{\rm in}\,\delta\mathbf{u}/\sigma\bigr) \\
        \delta\mathbf{u}' &= \delta\mathbf{u}
            + \mathbf{W}_{\rm out}^{(u)\top}\delta\mathbf{u}/\sigma
            + \mathbf{A}^\top \delta\mathbf{r}',
        \qquad
        A_{kj} = W^{(r)}_{{\rm out},kj} + 2 r'_k W^{(r^2)}_{{\rm out},kj}

    Only matrix-vector products appear, so the cost is linear in ``k`` and the
    full :math:`(n_{\rm res}+n)^2` Jacobian is never formed -- which is what
    makes :func:`esn_lyapunov_spectrum` affordable at a thousand reservoir
    nodes.

    The tests check this against central differences of :func:`esn_step`,
    because the chapter's headline is a comparison of Lyapunov spectra and a
    subtly wrong tangent gives a wrong spectrum that looks perfectly sensible.
    """
    n_res = network["n_reservoir"]
    n_in = network["n_input"]
    readout = np.asarray(readout, dtype=float)
    new_reservoir_state = np.asarray(new_reservoir_state, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    d_reservoir, d_u = tangent[:n_res], tangent[n_res:]
    sd = scaling["sd"]

    slope = 1.0 - new_reservoir_state**2
    new_d_reservoir = slope[:, None] * (
        network["W"] @ d_reservoir + (network["W_in"] @ d_u) / sd
    )
    coupling = (
        readout[:n_res] + 2.0 * new_reservoir_state[:, None] * readout[n_res : 2 * n_res]
    )
    new_d_u = (
        d_u
        + (readout[2 * n_res : 2 * n_res + n_in].T @ d_u) / sd
        + coupling.T @ new_d_reservoir
    )
    return np.concatenate([new_d_reservoir, new_d_u], axis=0)


def esn_lyapunov_spectrum(
    network: dict,
    readout: Array,
    reservoir_state: Array,
    u0: Array,
    scaling: dict,
    dt: float = 0.01,
    n_exponents: int = 8,
    n_steps: int = 30000,
    n_transient: int = 2000,
    seed: int | None = 5,
) -> Array:
    r"""Leading Lyapunov exponents of the *emulator*, by Benettin on its own
    tangent map, in units of inverse time.

    ``dt`` is the physical time per emulator step, so the exponents come out
    comparable with :func:`chaoslib.lyapunov.lyapunov_spectrum` applied to the
    system that generated the training data. That comparison is chapter 29.

    Only ``n_exponents`` are computed. The emulator has
    :math:`n_{\rm res} + n` state dimensions and almost all of them are
    strongly contracting -- the reservoir's own decay -- so the full spectrum is
    both expensive and mostly a property of the reservoir rather than of the
    dynamics it learned.

    **One thing this cannot do, and it matters.** For Lorenz 96 the exponents
    must sum to exactly :math:`-N`, the divergence of the flow, and that
    identity is what validates every spectrum computed elsewhere in this book.
    The emulator has no such identity: its state space is a thousand
    dimensions of reservoir with no phase-space volume to conserve. So the
    strongest available check on a Lyapunov calculation is **unavailable for
    the learned model**, which is a limitation of the method and not of this
    implementation.

    Returns ``nan`` values if the rollout diverges before the transient is
    over, rather than a spectrum computed from a blown-up trajectory.
    """
    n_res = network["n_reservoir"]
    n_in = network["n_input"]
    state = np.asarray(reservoir_state, dtype=float).copy()
    u = np.asarray(u0, dtype=float).copy()
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(
        rng.normal(size=(n_res + n_in, int(n_exponents)))
    )[0]
    total = np.zeros(int(n_exponents))
    counted = 0
    for index in range(int(n_steps)):
        state, u = esn_step(network, readout, state, u, scaling)
        if not np.isfinite(u).all() or np.abs(u).max() > 1.0e4:
            break
        basis = esn_tangent_apply(network, readout, state, basis, scaling)
        basis, upper = np.linalg.qr(basis)
        if index >= int(n_transient):
            total += np.log(np.maximum(np.abs(np.diag(upper)), 1.0e-300))
            counted += 1
    if counted == 0:
        return np.full(int(n_exponents), np.nan)
    return total / (counted * float(dt))
