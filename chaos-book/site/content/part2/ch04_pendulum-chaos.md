---
title: "Chapter 4 · Regular motion and why it is predictable"
weight: 204
part: "Part II — From regular motion to chaos"
knob: '$\theta_0$, $\delta_0$'
status: "live"
---

## Overview

Weather models are nonlinear, and nonlinearity usually takes the blame for the
two-week forecast limit. That explanation is not good enough. The pendulum in this
chapter is thoroughly nonlinear — its period depends on amplitude, its phase portrait
is nothing like a circle, and no closed-form solution in elementary functions exists —
and it is perfectly predictable forever. Give it an initial condition wrong by one part
in $10^8$ and the forecast is wrong by one part in $10^8$ a thousand swings later.

Add a second rod and the same integrator, at the same tolerance, loses all skill within
a few seconds.

So the ingredient that matters is not nonlinearity. It is **phase-space dimension**. A
single pendulum has one degree of freedom, so its state $(\theta,\omega)$ lives in a
plane, and conservation of energy confines the motion to a one-dimensional level curve.
The Poincaré–Bendixson theorem then leaves no room for anything but fixed points and
closed orbits: a trajectory confined to a curve cannot stretch and fold, so it cannot
be chaotic no matter how nonlinear the restoring force. The double pendulum has two
degrees of freedom, a four-dimensional phase space, and a three-dimensional energy
surface — the minimum in which stretch-and-fold can operate.

That is the argument this chapter exists to make concrete, and it is the argument the
rest of Part II builds on.

## The model

The exact nonlinear pendulum,

$$\ddot\theta = -\frac{g}{L}\sin\theta,$$

whose period is *not* $2\pi\sqrt{L/g}$ except in the small-angle limit, but

$$T = 4\sqrt{L/g}\;K\!\left(\sin^2(\theta_0/2)\right)$$

with $K$ the complete elliptic integral of the first kind. The notebook overlays the
numerically integrated orbit on the exact phase-space curve and compares the measured
period against this formula, so the reader can confirm the integrator is faithful before
trusting anything it says about the double pendulum.

For two rods the full Euler–Lagrange equations apply, with the state
$(\theta_1,\theta_2,\omega_1,\omega_2)$ and the characteristic $\sin(\theta_1-\theta_2)$
coupling. Energy is conserved but is no longer enough to pin the motion to a curve.

Both systems come from [`chaoslib.systems`]({{< relref "notation.md" >}}) —
`pendulum`, `double_pendulum`, `pendulum_energy`, `double_pendulum_energy` and
`pendulum_period_exact` — and are tested there against energy conservation over long
integrations, the elliptic-integral period at four amplitudes, and the small-angle limit.
Integration is adaptive RK45 at $r_{\mathrm{tol}} = 10^{-10}$, tight enough that the
divergence you see is the physics rather than the solver.

{{< marimo src="/nb/ch04_pendulum-chaos.html" >}}

## Exercises

**Analytic.** Show that a one-degree-of-freedom autonomous system whose energy is
conserved cannot have a positive Lyapunov exponent. Where exactly does the argument use
autonomy? Construct a *driven* pendulum that is chaotic, and identify which step of the
argument it defeats.

**Computational.** Measure the single pendulum's period as a function of $\theta_0$ from
$5^\circ$ to $175^\circ$ and plot the ratio to the small-angle result. At what amplitude
does the small-angle approximation err by 1 %? By 10 %?

**Exploratory.** Set both double-pendulum angles small — say $\theta_1 = 10^\circ$,
$\theta_2 = 5^\circ$ — and run the twin-trajectory experiment. Is the separation growing
exponentially, or linearly? Now raise the energy until it does grow exponentially. Chaos
in this system is not a property of the equations alone but of the equations *and the
energy*: locate that boundary as sharply as you can, and say what it means for a system
whose "energy" varies from day to day.

## Further reading

- Smith (2007), *Chaos: A Very Short Introduction* — the clearest short treatment of why
  dimension rather than nonlinearity is the discriminator.
- Strogatz, *Nonlinear Dynamics and Chaos* — Poincaré–Bendixson, and the phase-plane
  methods used here *[citation needed: section]*.
- Continue to the [Lorenz 63 chapter]({{< relref "ch06_lorenz63.md" >}}), where the same
  twin-trajectory experiment is run on a system built from the atmosphere.
