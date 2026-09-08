---
title: "Chapter 2 · A short history of numerical weather prediction"
weight: 102
part: "Part I — What predictability means"
knob: 'spurious wind, extrapolation interval, mean depth'
status: "live"
---

## Overview

In 1922 Lewis Fry Richardson published a six-hour weather forecast computed by hand
from the governing equations. His answer was a surface-pressure change of **145 hPa**
*[citation needed]*. The observed change was essentially nothing, and nothing
comparable has ever been observed anywhere on Earth.

The equations were right, and so, as far as anyone has been able to check, was the
arithmetic. **So what went wrong?** The usual one-line answer — his initial conditions
were unbalanced — is true and not quite the point. This chapter builds the smallest
system that contains Richardson's problem and measures what happened, and the answer is
sharper than the slogan: **the tendency he computed was real, and his error was
extrapolating an oscillation.**

## The model

One-dimensional rotating shallow water on a periodic domain:

$$
\partial_t u + u\,\partial_x u - f v = -g\,\partial_x h,
\qquad
\partial_t v + u\,\partial_x v + f u = 0,
\qquad
\partial_t h + \partial_x (h u) = 0 .
$$

This is the least a model can contain and still have Richardson's difficulty: a slow
balanced mode that carries the weather, and a fast inertia-gravity mode that carries
none of it. Derivatives are spectral, so the linear dispersion relation is exact to
round-off rather than accurate to the order of a stencil — the measured frequencies sit
on $\omega^2 = f^2 + gHk^2$ to **0.05 %**.

Two features of that relation drive everything. There is a **floor** — no
inertia-gravity wave is slower than $f$, about 17 hours at mid-latitudes — and $\omega$
grows without bound with $k$, so refining the grid makes the system stiffer. With
$H = 8000$ m, the external mode Richardson's equations carried, $c = \sqrt{gH} = 280$
m/s and the Rossby radius is 2800 km: *comparable to a synoptic weather system*, so the
balanced and unbalanced parts of an analysis are not separated by scale and cannot be
told apart by eye.

In this system the balanced state is **exactly** steady. With $u \equiv 0$ and
$fv = g\,\partial_x h$ every tendency vanishes identically — $|\partial_t u|$ is
$4\times10^{-19}$, which is round-off, and a 30 m/s jet sits there with a six-hour drift
of exactly zero. That is one dimension being kind: in two dimensions, with $f$ varying
and advection able to feed back, no exact balanced manifold exists, which is why
nonlinear normal-mode initialisation and digital filtering had to be invented.

{{< marimo src="/nb/ch02_history-of-nwp.html" >}}

## The tendency was real; extrapolating it was the error

Break the balance the way an observing system breaks it — with a spurious **divergent**
wind. That is the right way to break it, because $\partial_t h = -\partial_x(hu)$ is a
flux divergence, so the pressure tendency responds only to the divergent flow.

One metre per second of spurious divergence forces a pressure tendency of **9.18 hPa per
hour**, exactly linear in the wind (normalised tendencies 0.5000, 1.0000, 2.0000, 5.0000
at 0.5, 1, 2 and 5 m/s). So Richardson's 145 hPa needs a spurious divergent wind of
about **2.6 m/s**, if the tendency is multiplied by six hours as he did. That is a small
error in a 1910 wind analysis, and inside the uncertainty of a modern analysis in a
data-sparse region. **The surprising thing is not that his answer was large but how
little it took.**

And the multiplication is the mistake. At 1 m/s the tendency extrapolated over six hours
gives 55.1 hPa; the model's actual six-hour change is **1.28 hPa**, a factor of 43
smaller. The tendency *oscillates*, at the inertia-gravity period of 2.45 hours, so

$$
\Delta p(T) = \frac{A}{\omega}\sin \omega T
$$

is **bounded by $A/\omega$ for ever** while the extrapolation $AT$ grows without limit.
Measured: the largest excursion anywhere in 24 hours is 3.684 hPa against a predicted
bound of 3.587 hPa — a ratio of **1.027**. The extrapolation error is of order $\omega T$,
about a factor of ten once the forecast is long compared with a gravity-wave period.

That is the useful form of the lesson, because it says what to do: **filter in time, or
start from a state with no fast mode.**

## A small pressure tendency is not a balanced state

There is a trap here that survives into modern practice. Since the height equation is a
flux divergence, it sees *only* the divergent wind. Break the balance in the
**rotational** wind instead — scale $v$ down, leaving $u = 0$ — and the initial pressure
tendency is not small. It is **exactly zero**. The forecast is ruined anyway: removing
the rotational wind entirely gives a six-hour pressure change of **29.3 hPa**, and
removing a tenth of it gives 2.86 hPa, with the diagnostic reading zero throughout.

So "the pressure tendency is small" and "the state is balanced" are different statements,
and the first does not imply the second. Pressure-tendency and divergence-based balance
checks are still used and are blind to exactly this error. What catches it is checking
the wind against the mass field — which is what geostrophic balance is, and what a
variational analysis enforces through $\mathbf{B}$
([chapter 18]({{< relref "../part5/ch18_variational-da.md" >}})).

It also explains why Richardson's failure was legible. He broke the divergent wind, so
his diagnostic screamed. An analysis that breaks the rotational wind produces a forecast
that is quietly wrong, which is worse.

## The timestep, and why filtering was worth a computer

The fast mode is a problem a second time, for reasons of arithmetic rather than physics.
An explicit scheme is stable only while information moves less than about a grid cell per
step, and the relevant speed is the fastest signal in the system — the external gravity
wave at 280 m/s, not the wind at 30.

Measured, the threshold is $C\,\Delta x/c$ with $C$ between **0.843 and 0.911** across a
factor of four in $c$ — an 8 % spread, no fitted constant beyond the scheme's own Courant
number. And the wind barely enters: quadrupling the jet moves the threshold a few per
cent, while quadrupling $c$ quarters it. **The timestep is set by the wave, not by the
weather** — and the wave carries no forecast information at all.

Hence **filtering**: equations from which the fast mode has been removed analytically.
The barotropic vorticity equation $\partial_t \zeta + J(\psi,\zeta) = 0$ has no gravity
waves by construction, because it has no divergence and no free surface. It is what
Charney, Fjørtoft and von Neumann integrated on ENIAC in 1950 *[citation needed]*, and
over 400 steps it conserves energy and enstrophy to better than $10^{-6}$ %. At the
external mode with a 30 m/s jet, a six-hour forecast costs **170 steps** unfiltered
against **16** filtered — a factor of 10.5, which on a machine doing a few hundred
operations a second was the difference between a forecast and none.

Filtering then dominated for about a decade and lost. It had to: the barotropic vorticity
equation cannot represent fronts or convection or anything else living in the divergent
flow. What replaced it was the *primitive* equations again — Richardson's equations —
with the imbalance handled at the **analysis** step instead of by amputating the physics.
That is the line from this chapter to chapters 18–20: **the modern answer to Richardson is
not a better equation set but a better initial condition.**

## Exercises

1. The extrapolation error is $\omega T$. Read the ratio off the sliders at 1 hour and at
   24 hours. Where would the two curves cross, and what would have to be true of the
   atmosphere for a six-hour extrapolation to be safe?
2. Reduce the depth to 500 m, roughly an internal rather than external mode. What happens
   to the Rossby radius, and would Richardson's arithmetic have worked on a model
   carrying only that mode?
3. The rotational imbalance is invisible to the pressure tendency. Propose a diagnostic
   that *would* catch it using only fields Richardson had, and say whether he could have
   computed it by hand.
4. The timestep goes as $1/c$ with $c = \sqrt{gH}$. If you halve $\Delta x$, what happens
   to the total arithmetic for a fixed forecast length — and is that better or worse than
   the factor filtering buys?

## Further reading

- Richardson, L. F. (1922). *Weather Prediction by Numerical Process.* Cambridge
  University Press *[citation needed: pages]*.
- Lynch, P. (2006). *The Emergence of Numerical Weather Prediction: Richardson's Dream.*
  Cambridge University Press — the modern reconstruction, including why the 145 hPa arose
  *[citation needed]*.
- Charney, J. G., Fjørtoft, R. and von Neumann, J. (1950). Numerical integration of the
  barotropic vorticity equation. *Tellus*, **2**, 237–254 *[citation needed: pages]*.
- Courant, R., Friedrichs, K. and Lewy, H. (1928), on the stability of difference
  equations *[citation needed]*.
- Bauer, P., Thorpe, A. and Brunet, G. (2015). The quiet revolution of numerical weather
  prediction. *Nature*, **525**, 47–55 — where the story goes after 1950.
