---
title: "Chapter 7 · Lyapunov exponents and doubling times"
weight: 307
part: "Part III — Quantifying chaos and predictability"
knob: 'ρ, integration length $T$, window $\tau$'
status: "live"
---

## Overview

[Chapter 6]({{< relref "../part2/ch06_lorenz63.md" >}}) showed two nearly identical
Lorenz trajectories pulling apart and fitted a growth rate to the separation. Run that
experiment again from a different point on the attractor and you get a different number
— 0.6 here, 1.1 there. So which is *the* rate? Is there one at all? And if the answer
depends on where you start, what does a forecast centre mean when it says the atmosphere
has a two-week limit?

This chapter turns "chaotic" into a number and is careful about which number. There are
three, they are routinely confused, and they answer different questions: the **spectrum**
$\lambda_1 \ge \lambda_2 \ge \lambda_3$, which describes the attractor and depends on
nothing else; the **finite-time exponent** $\lambda(x,\tau)$, which describes one state at
one lead time; and a **twin-trajectory fit**, which describes one realisation and depends
partly on luck. Chapter 6 measured the third. This chapter computes the first, shows how
widely the second varies, and ends with the case where all of them mislead.

## The model

Oseledets' multiplicative ergodic theorem guarantees that

$$\lambda_i = \lim_{T\to\infty}\frac{1}{T}\ln \sigma_i\!\left(\mathbf{M}(x_0,T)\right)$$

exists and is **independent of $x_0$** for almost every starting point — which is what
makes these properties of the system rather than of an experiment. Computing them
directly fails numerically, because every column of $\mathbf{M}$ grows like
$e^{\lambda_1 T}$ and the subdominant directions are lost within a few Lyapunov times.
The **Benettin algorithm** carries an orthonormal frame along the trajectory,
re-orthonormalises by QR at every step, and accumulates $\ln|R_{ii}|$; nothing overflows
and every direction stays resolved. `chaoslib.lyapunov` implements it.

The check that makes the result trustworthy is exact. For Lorenz 63 the divergence of the
flow is state-independent, so

$$\sum_i \lambda_i = -(\sigma + 1 + \beta) = -13.6\overline{6}$$

holds at **any** integration length, independent of the trajectory. That tests the
implementation rather than its convergence — a sign error or a mis-ordered QR fails it
immediately, while the leading exponent alone would still look plausible. It is asserted
in the test suite and displayed live in the notebook.

### What is computed live, and what is not

The spectrum at the reader's chosen ρ is computed live (about ten seconds). Two figures
are precomputed by `scripts/generate_rho_sweep.py`, because each needs many Benettin runs
and neither has a knob: the $\lambda_1(\rho)$ curve across the whole transition, and the
transient-chaos measurement in Section 5. Computing those live would cost roughly 80
seconds and four minutes respectively in the browser, for a result identical for every
reader.

{{< marimo src="/nb/ch07_lyapunov-exponents.html" >}}

## Exercises

**Analytic.** Show that $\sum_i \lambda_i$ equals the time-averaged divergence
$\langle \nabla\!\cdot\! f\rangle$, and hence that it is exactly $-(\sigma+1+\beta)$ for
Lorenz 63. Then explain why $\lambda_2 \approx 0$ is structural for any continuous-time
flow on a bounded attractor that is not a fixed point.

**Computational.** Verify that the doubling time and the Lyapunov time differ by
$\ln 2$, and that $D_{KY} = 2 + \lambda_1/|\lambda_3|$ reproduces the value
[chapter 8]({{< relref "ch08_attractor-dimension.md" >}}) obtains from a sampled
trajectory with no reference to the dynamics.

**Exploratory.** Set ρ = 22.7 and vary the integration length. The exponent decays from
+0.73 towards zero. Before reading Section 5, work out what could make a positive
exponent shrink as you measure it for longer — then check your answer against the
trajectory itself.

## A caution this chapter is built around

Below the transition, Lorenz 63 has a **chaotic saddle**: an invariant set that is
genuinely chaotic but is not an attractor. A trajectory near it separates exponentially,
looks chaotic in every diagnostic, and then falls off and settles onto a fixed point. At
ρ = 22.7 a 900-MTU run ends with the state completely motionless — a range of
$2\times10^{-13}$ — while $\lambda_1$ measured over $T = 50$ comes out at $+0.73$.

**A positive finite-$T$ exponent is not proof of a chaotic attractor.** It is evidence of
exponential separation over the window you measured, and the gap between those two claims
is the gap between $\lim_{T\to\infty}$ in Oseledets' theorem and the $T$ you could
afford.

## Further reading

- Benettin, G., Galgani, L., Giorgilli, A. and Strelcyn, J.-M. (1980). Lyapunov
  characteristic exponents for smooth dynamical systems. *Meccanica*, **15**, 9–20.
- Oseledets, V. I. (1968). A multiplicative ergodic theorem. *Trudy Moskov. Mat. Obšč.*,
  **19**, 179–210 *[citation needed: pages]*.
- Sparrow, C. (1982). *The Lorenz Equations: Bifurcations, Chaos, and Strange
  Attractors.* Springer — the bifurcation structure behind Section 5.
- Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*, §6.1.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 2 *[citation needed: pages]*.
