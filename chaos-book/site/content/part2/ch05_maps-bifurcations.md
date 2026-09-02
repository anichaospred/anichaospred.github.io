---
title: "Chapter 5 · Maps, bifurcations, and the routes to chaos"
weight: 205
part: "Part II — From regular motion to chaos"
knob: '$r$, $x_0$, the parameter window'
status: "live"
---

## Overview

Every forecast model contains parameters nobody can measure exactly: a mixing length,
an autoconversion threshold, an entrainment rate. Suppose one is uncertain by half a
percent. How much does that matter?

Usually, half a percent. But there are parameter values where the answer is
qualitatively different — where an arbitrarily small change turns a steady state into
an oscillation, or an oscillation into chaos. Those values are **bifurcations**, and
this chapter is about how they are organised.

The vehicle is deliberately absurd:

$$x_{n+1} = r\,x_n(1 - x_n), \qquad 0 \le x \le 1 .$$

One variable, one parameter, no time step, no derivatives — just multiplication. It has
no atmosphere in it anywhere. And yet the **numbers** describing its route to chaos are
the same numbers found in convecting fluids, dripping taps and nonlinear circuits. That
portability is the claim [chapter 3]({{< relref "../part1/ch03_model-hierarchy.md" >}})
makes about why a three-variable model teaches something true about a $10^9$-variable
one. This chapter is where it stops being a claim.

## The model

A fixed point satisfies $x^* = 1 - 1/r$, and attracts when
$|f'(x^*)| = |2 - r| < 1$. At $r = 3$ that multiplier reaches $-1$ and the fixed point
gives way to a 2-cycle; the 2-cycle gives way to a 4-cycle at 3.449, and so on. The
notebook makes the stability calculation graphical (the **cobweb**), then sweeps the
whole family at once.

`chaoslib.maps` provides the machinery: `bifurcation_points` and
`map_lyapunov_exponent` are vectorised over the parameter axis, so a 1400-point sweep
costs the same as a single value — which is why almost everything in this chapter runs
live in the browser rather than being precomputed.

{{< marimo src="/nb/ch05_maps-bifurcations.html" >}}

## Four results

**The Lyapunov exponent is the diagnosis, and the diagram alone is not.** For a 1-D map
[chapter 7]({{< relref "../part3/ch07_lyapunov-exponents.md" >}})'s definition collapses to $\lambda(r) = \langle\ln|f'_r(x)|\rangle$ — no
tangent linear model, no re-orthonormalisation, one average. Measured
$\lambda(4) = 0.69315$ against the exact $\ln 2$, since at $r = 4$ the map is conjugate
to $x \mapsto 2x \bmod 1$. Its sign separates periodic from chaotic parameters where the
picture cannot: the **period-3 window** sits well above the accumulation point
$r_\infty = 3.5699457$ and has $\lambda < 0$ across three quarters of its width.

**"Chaotic" is a property of a parameter value, not of a system.** Inside that period-3
window is a complete cascade $3 \to 6 \to 12 \to 24$, and inside *its* chaotic band are
further windows of period 9, 12, 30 and 36. The chaotic and periodic parameter sets are
interleaved at every scale, so no finite sampling of parameter space establishes which
side a given model sits on.

**$\delta = 4.669$ belongs to no particular map.** The notebook locates *superstable*
parameters — where the $2^n$-cycle contains the critical point, so its multiplier is
exactly zero and the parameter solves the smooth equation $f_r^{2^n}(x_c) = x_c$ — and
does it for three families sharing no algebra: logistic $r x(1-x)$, sine $r\sin(\pi x)$,
cubic $r x(1-x^2)$, with critical points $1/2$, $1/2$, $1/\sqrt3$ and first superstable
parameters 2, $1/2$, $3/2$. Three unrelated cascades, one ratio: 4.669191, 4.664075,
4.669038 against $\delta = 4.669201609$.

The self-similarity is quantitative too. The period-3 window's own $3\cdot 2^n$ cascade
has superstable parameters 3.831874, 3.844569, 3.848345, 3.849198, 3.849383, 3.849423,
whose spacing ratios converge to the same $\delta$ from a cascade spanning 0.0175 in $r$
rather than 1.57. That identity is the renormalisation structure which *makes* $\delta$
universal.

**A regime can end with nothing changing.** Below the tangent bifurcation at
$r_c = 1 + 2\sqrt2$ the 3-cycle does not exist, but $f^3$ passes close to the diagonal,
and an orbit entering that channel is nearly 3-periodic for tens of iterations before
bursting out. Mean laminar length diverges as $(r_c - r)^{-0.4965}$ against the
predicted $-1/2$, measured over a 33-fold range of $r_c - r$. Nothing external changes
when a burst begins — and the diverging timescale is the mechanism behind
critical-slowing-down early-warning indicators, taken up in
[chapter 27]({{< relref "../part6/ch27_regimes-tipping.md" >}}).

## Exercises

**Analytic.** Show that the 2-cycle born at $r = 3$ is
$x_\pm = [(r+1) \pm \sqrt{(r-3)(r+1)}]/2r$, and that it loses stability at
$r = 1 + \sqrt6$. Then show that at $r_c = 1 + 2\sqrt2$ the third iterate satisfies
$f^3(x) = x$ *and* $(f^3)'(x) = +1$ simultaneously, and explain why that makes the
3-cycle's birth invisible to a root-finder.

**Computational.** Measure $\delta$ from a fourth unimodal family of your own choosing.
Then explain, from the size of the successive gaps, why double precision limits the
logistic cascade to about nine levels — and why that is a limitation of arithmetic
rather than of the mathematics.

**Exploratory.** Set the parameter window to the period-3 range and read $\lambda$; then
set $r = 3.83$ in Section 1 and count the cycle. Explain why the bifurcation diagram
alone could mislead you here and $\lambda$ could not.

## Further reading

- Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear
  transformations. *Journal of Statistical Physics*, **19**, 25–52.
- May, R. M. (1976). Simple mathematical models with very complicated dynamics.
  *Nature*, **261**, 459–467.
- Pomeau, Y. and Manneville, P. (1980). Intermittent transition to turbulence in
  dissipative dynamical systems. *Communications in Mathematical Physics*, **74**,
  189–197 — the $-1/2$ law.
- Strogatz, S. H. *Nonlinear Dynamics and Chaos*, ch. 10 *[citation needed: edition and
  section numbers]*.
- Smith, L. A. (2007). *Chaos: A Very Short Introduction* *[citation needed: pages]*.
