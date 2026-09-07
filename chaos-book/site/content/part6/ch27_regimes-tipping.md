---
title: "Chapter 27 · Regimes, bistability, and tipping points"
weight: 627
part: "Part VI — Predictability of the second kind"
knob: 'tilt, noise amplitude'
status: "live"
---

## Overview

[Chapter 25]({{< relref "ch25_climate-prediction.md" >}}) offered a reassurance: when the
trajectory is unpredictable the statistics need not be, so a projection can be confident
about a distribution it cannot resolve in detail. That reassurance assumes the attractor
deforms *smoothly* as the forcing changes. If a slowly changing parameter destroys the
state the system currently occupies, the distribution moves abruptly, and it does so
exactly where a smooth-response argument is least willing to look.

So: **can you see a tipping point coming?** The literature says yes in principle — a state
about to be destroyed relaxes more slowly, and slow relaxation shows up in a record as
rising variance and rising autocorrelation. This chapter verifies that theory against
exact identities, and then does what the theory does not: runs the indicator as a
**detection problem with a calibrated false-alarm rate**.

## The model

The normal form for two competing states, tilted by a parameter and kicked by noise:

$$
\dot x = x - x^3 + \mu + \sigma\,\xi(t),
\qquad
V(x) = -\tfrac12 x^2 + \tfrac14 x^4 - \mu x ,
$$

so $\dot x = -V'(x)$ — a ball in a double well. Being a *gradient* system buys two exact
results that everything else is checked against. The stationary density is exactly
Boltzmann, $p \propto e^{-2V/\sigma^2}$; and the fold sits at $\mu_c = 2/(3\sqrt3)$,
$x_c = -1/\sqrt3$, where the right-hand side **and** its derivative vanish identically
rather than to tolerance.

Measured as the ratio of the two lobes' populations, the Boltzmann form holds to
**4.6 %** across a ratio spanning a factor of fourteen. That single comparison tests the
drift and the integrator's $\sigma\sqrt{\Delta t}$ noise convention together, which is
worth having, since a factor of two in the diffusion coefficient is the standard error
and it is invisible in a trajectory plot.

{{< marimo src="/nb/ch27_regimes-tipping.html" >}}

## Tipping without a tipping point

A system can leave its state with the parameter held perfectly still: an unlucky run of
kicks carries the ball over the barrier. Kramers' waiting time is exponential in
$1/\sigma^2$ with slope exactly $2\Delta V$, and the measurement is unambiguous — fitted
slope **0.2333** against an exact **0.2350**, an error of 0.8 %.

**The prefactor is another matter.** It sits at **0.61** of the formula, so Kramers'
expression overestimates the waiting *time* by about 60 % at these barrier heights. It is
asymptotic in $2\Delta V/\sigma^2$ and a tipping problem lives at moderate values of that
ratio. Use it for scaling, not for a date.

One noise level is deliberately left in the figure as an open square: only 84 % of its
members escaped within the run, so its mean is **censored**, and censoring biases a mean
escape time *low* because the slow escapes are the ones missing. Including that single
point drags the fitted slope to 0.2065 — an error of **12.1 %** rather than 0.8 %. It is
the deepest barrier in the sweep, which is to say the most interesting one, and it is
also what a finite computing budget or a finite observational record will always hand
you. `chaoslib.earlywarning.escape_times` returns `nan` for a member that never escaped
rather than quietly omitting it.

## Critical slowing down is exactly true, and stops being useful before the fold

Near a stable state the dynamics linearise to an Ornstein–Uhlenbeck process, for which
$\operatorname{var} x = \sigma^2/2|\lambda|$ and the lag-1 autocorrelation is
$e^{\lambda\Delta t}$, both exact. And $\lambda \to 0$ at the fold at a known rate:
expanding about $(x_c,\mu_c)$, where $f_{xx} = 2\sqrt3$,

$$
\lambda \simeq -2 \cdot 3^{1/4} \sqrt{\mu_c - \mu},
\qquad
\Delta V \simeq \tfrac43 3^{-1/4} (\mu_c - \mu)^{3/2}.
$$

Up to $\mu = 0.25$ the measured standard deviation matches its exact value to **3.5 %**
and the autocorrelation to **1.6 %**, both rising as advertised.

**Then the indicator stops describing its own system, well short of the fold.** At
$\mu = 0.32$ — against a fold at $0.385$ — 46 % of members have already left the well and
the measured spread is **3.6 times** the theory. Closer in it is worse than useless: by
$\mu = 0.382$ every member has escaped and the measured autocorrelation has **fallen to
0.47** while the theory says 0.96. An analyst watching that number would conclude the
system was becoming *more* stable at the moment it finished tipping.

The reason is not numerical. The predicted fluctuation grows as $(\mu_c-\mu)^{-1/2}$
while the distance from the well to the saddle shrinks as $(\mu_c-\mu)^{1/2}$. They
cross, and once the predicted fluctuation is the size of the basin, the fluctuation *is*
the escape.

## Two timing laws, with opposite signs

**A noiseless system leaves late.** In the fold normal form $\dot u = \sqrt3 u^2 +
\gamma\tau$ the substitution $u = -w'/w$ gives Airy's equation, so departure is the first
zero of $w$:

$$
\mu_{\text{tip}} - \mu_c \simeq |a_1|\,3^{-1/6}\,\gamma^{2/3} = 1.9469\,\gamma^{2/3},
$$

with $|a_1| = 2.3381\ldots$. Measured over a factor of sixteen in sweep rate, the ratio
to that law climbs monotonically to **0.988** as the sweep slows — approaching from below,
as an asymptotic law should.

**A noisy system leaves early**, and by more than the naive estimate. The barrier vanishes
as $(\mu_c-\mu)^{3/2}$, faster than linearly, so escape becomes certain strictly before
the fold. Integrating Kramers' rate along the sweep is elementary under $w = d^{3/2}$:

$$
d^* = \left[\frac{\sigma^2}{\tfrac83 3^{-1/4}}
      \ln \frac{2\cdot3^{1/4}\,\sigma^2}
      {3\pi\,\tfrac83 3^{-1/4}\,\gamma\,\ln 2}\right]^{2/3}.
$$

With the sweep slow enough that the deterministic delay cannot contaminate the answer,
this predicts the measured median tipping tilt to **3.3 %** across a factor of five in
$\sigma$.

**The logarithm is the whole content.** Drop it and you get the $\sigma^{4/3}$ scaling
usually quoted, which is wrong here by a factor rising from **1.81 to 3.18** across that
range — not an offset, a trend, which is why fitting a power law to the measurements
returns $\sigma^{1.69}$ rather than $\sigma^{1.333}$. There is no pure power law: the
logarithm carries the sweep rate, and a slower sweep gives noise more time to find the
barrier.

Which says something about real systems. **The distance-to-fold at which a system commits
is set by its noise and its rate of change, not by the fold alone.** Two systems with
identical bifurcation structure tip at different forcings if their internal variability
differs. A tipping threshold quoted as a property of the system — a temperature, a
freshwater flux — is incomplete without the variability and the rate of approach attached.

## Early warning: good recall, no specificity

The recipe: slide a window along the record, compute variance and lag-1 autocorrelation
in each, and test for an upward trend with Kendall's $\tau$. **How large a $\tau$?** That
is the only part that determines whether the method has skill, and it is where most
presentations stop. $\tau$ computed on overlapping windows of an autocorrelated record has
a null distribution far wider than the independent-sample one, so a threshold from a table
is meaningless. Here it is calibrated against a null run — the same system with the
parameter held fixed — so the false-alarm rate is **5 % by construction**.

Four scenarios, 200 realisations each:

| scenario | tipped | alarm (variance) | alarm (autocorrelation) |
|---|---|---|---|
| null, $\mu$ fixed at 0 | 0 % | 5 % | 5 % |
| ramp through the fold | 100 % | **100 %** | 89 % |
| ramp stopping at $\mu = 0.30$ | **0 %** | **100 %** | 89 % |
| noise-induced, $\mu$ fixed at 0.25 | **80 %** | **13 %** | 18 % |

**What works.** Against a genuine sweep the indicator fires on every realisation, and not
at the last moment: the alarm still fires on more than 90 % of cases when the decision
must be taken **1200 time units** early — a third of the record discarded, the call made
at a tilt of 0.28 when tipping comes at 0.36. Lead time is not the problem.

**What fails.** The scenario that ramps to $\mu = 0.30$ and stops **never tips**, and the
alarm fires on **100 %** of those realisations — indistinguishable from the runs that do.
The indicator is not broken; it is answering a different question. Rising variance is
evidence that the system is *approaching a bifurcation*, true in both cases. It carries no
information about whether the approach will continue, because that is a fact about the
forcing and not about the system's dynamics. **No statistic computed from a record of $x$
can supply it.**

**And the transitions with no precursor at all.** The noise-induced scenario tips 80 % of
the time with the parameter perfectly still, and the alarm fires on 13 % against a 5 %
baseline — essentially no skill. Four realisations in five tip with no warning whatever.
That is not an estimator failing: there is genuinely nothing to detect, because the
potential never changed.

The honest summary is that early-warning indicators have **good lead time and good recall,
and no specificity.** In a real record you do not know which scenario you are in — that is
the entire problem — and the indicator does not tell you. Which is worth being precise
about rather than cynical about: a rising-variance alarm is real information, saying the
system is closer to a fold than it was. What it is not is a forecast.

## Exercises

1. The noise slider changes $\sigma$ without changing the potential. At what noise does
   "which state the system is in" stop being a useful description, and what does that
   correspond to in the detection problem?
2. The Kramers prefactor is 60 % optimistic here. Estimate the barrier height at which it
   would be good to 10 %, and say whether a system with that barrier is one anybody would
   worry about tipping.
3. The two failure conditions above — the fluctuation reaching the basin width, and
   members escaping — coincide. Show that they must, using the near-fold scalings.
4. The "stops short" scenario defeats the indicator. Design a measurement that *would*
   distinguish it from a genuine sweep, and say what extra information your measurement
   requires.

## Further reading

- Scheffer et al. (2009), on early-warning signals for critical transitions
  *[citation needed]*
- Lenton et al. (2008), on tipping elements in the Earth system *[citation needed]*
- Ditlevsen & Johnsen, on noise-induced versus bifurcation-induced tipping
  *[citation needed]*
- Ashwin et al., on rate-induced tipping *[citation needed]*
- Boers (2021), on early-warning signals in the observational record and their statistical
  pitfalls *[citation needed]*
- Kramers (1940), on escape over a potential barrier *[citation needed]*
