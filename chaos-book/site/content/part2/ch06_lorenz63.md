---
title: "Chapter 6 · The Lorenz (1963) system: the butterfly"
weight: 206
part: "Part II — From regular motion to chaos"
lectures: "L7 (seeds L4–5, L10)"
knob: '$\sigma$, $\rho$, $\beta$'
status: "live"
---

## Overview

In 1963 Edward Lorenz truncated the equations for thermal convection to three ordinary
differential equations, integrated them on an LGP-30, and found that a run restarted from
a printout rounded to three decimals — $0.506$ instead of $0.506127$ — diverged completely
from the original. That is the origin of the two-week forecast limit, and of the phrase
that came from the title of his 1972 talk.

This chapter is the hinge of the book. [Chapter 4]({{< relref "ch04_pendulum-chaos.md" >}})
established that chaos needs a phase space of at least three dimensions; Lorenz 63 is the
smallest system built from atmospheric physics that has one. Everything in Parts III to V
— Lyapunov exponents, error growth, ensembles, adjoints, data assimilation — is developed
on this system before being trusted on anything larger.

The notebook works through four things in order:

1. **The attractor.** Vary $\sigma$, $\rho$ and $\beta$ and the initial state, and watch
   the long-term behaviour change qualitatively: decay to the origin, a spiral onto one
   of the convective rolls $C^\pm$, the strange attractor, and the periodic window near
   $\rho \approx 100$. A live readout classifies the regime and fits the leading
   finite-time Lyapunov exponent.
2. **Sensitive dependence.** Two trajectories from initial states differing by $\delta_0$,
   with $\ln\|\delta\|$ plotted against time so the exponential stretch, and the
   saturation that ends it, are both visible.
3. **Ensembles.** A cloud of perturbed initial states, with spread and ensemble-mean error
   plotted together — and the crucial observation that the answer depends on *where on the
   attractor you start*.
4. **The real atmosphere.** Where the numbers land: $\lambda \approx 0.9$ MTU$^{-1}$ here
   against $\approx 0.35$ day$^{-1}$ for the atmosphere, and the logarithmic return on
   better observations, $\Delta t = \ln 10/\lambda \approx 6.5$ days per decade of
   analysis-error reduction.

## The model

$$\dot X = \sigma (Y - X), \qquad
  \dot Y = X(\rho - Z) - Y, \qquad
  \dot Z = XY - \beta Z$$

$X$ measures the intensity of the convective overturning, $Y$ the temperature difference
between ascending and descending branches, and $Z$ the departure of the vertical
temperature profile from linear. Symbols follow [the notation page]({{< relref "notation.md" >}}).

The origin is always a fixed point; for $\rho > 1$ a symmetric pair appears at

$$C^\pm = \left(\pm\sqrt{\beta(\rho-1)},\; \pm\sqrt{\beta(\rho-1)},\; \rho-1\right),$$

the steady clockwise and counter-clockwise rolls, and these lose stability in a Hopf
bifurcation at

$$\rho_H = \frac{\sigma(\sigma+\beta+3)}{\sigma-\beta-1} \approx 24.74$$

for the classical $\sigma = 10$, $\beta = 8/3$. Above $\rho_H$ neither roll is stable and
the trajectory wanders between them forever without repeating.

All of this comes from `chaoslib.systems` and `chaoslib.lyapunov`. Two identities are
worth knowing because they make the implementation checkable rather than merely plausible:
the Jacobian's trace is $-(\sigma+1+\beta)$ at *every* state, so the three Lyapunov
exponents must sum to exactly that; and the tangent-linear propagator must satisfy
$\det\mathbf{M} = e^{\tau\,\mathrm{tr}\mathbf{J}}$. Both are asserted in the test suite,
and the first is displayed live in the notebook.

For the record, the values the library reproduces: $\lambda_1 = 0.9056$,
$\sum_i\lambda_i = -13.6667$, and a Kaplan–Yorke dimension of $2.06$ — the attractor is
more than a surface and less than a volume.

{{< marimo src="/nb/ch06_lorenz63.html" >}}

## Exercises

**Analytic.** Derive $\rho_H$ by linearising about $C^+$ and finding where a complex pair
of eigenvalues crosses the imaginary axis. Then explain why the formula requires
$\sigma > \beta + 1$, and what happens physically when it does not hold.

**Computational.** Fix $\rho = 28$ and measure the leading finite-time Lyapunov exponent
at twenty different points on the attractor, using a 0.5 MTU window. Plot the
distribution. The asymptotic value is 0.9056 — how wide is the spread around it, and what
does its width imply for the claim "the atmosphere has a two-week limit"?

**Exploratory.** Reduce $\delta_0$ by a factor of ten and measure how much extra lead time
you gain before the error reaches half of saturation. Repeat twice more. The gain per
decade should be roughly constant, at $\ln 10/\lambda$. Now argue from that number about
how much forecast improvement a tenfold increase in observing-system accuracy could
deliver — and compare your answer to the historical record in
[Chapter 22]({{< relref "../part5/ch22_verification.md" >}}).

## Further reading

- Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric
  Sciences*, **20**(2), 130–141. Still the clearest statement of the result.
- Lorenz, E. N. (1993). *The Essence of Chaos* — the discovery told by the person who made
  it.
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, ch. 2
  *[citation needed: pages]*.
- Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, §6.1.
