---
title: "Chapter 18 · Variational data assimilation"
weight: 518
part: "Part V — The machinery of prediction"
knob: 'window length, $\mathbf{B}$'
status: "live"
---

## Overview

At 00 UTC you must produce the single best estimate of the atmosphere's present state.
You have a six-hour forecast that is wrong in ways you partly understand, and a few
million observations scattered irregularly through the last six hours, each wrong in
ways you understand better. Neither is the truth.

The variational answer is to stop treating this as interpolation and treat it as
**minimisation**. Write one scalar measuring how badly a candidate state disagrees with
everything you know, and find the state that minimises it:

$$
J(x_0) = \tfrac12 (x_0 - x^b)^{\top}\mathbf{B}^{-1}(x_0 - x^b)
  + \tfrac12 \sum_k \big(y_k - \mathbf{H}\,\mathcal{M}_{0\to k}(x_0)\big)^{\top}
    \mathbf{R}^{-1}\big(\cdots\big).
$$

Both terms are squared distances measured in units of their own uncertainty, which is
what makes them commensurable. **3D-Var is the special case
$\mathcal{M} = \mathrm{identity}$** — every observation treated as though taken at the
analysis time. **4D-Var keeps $\mathcal{M}$**, so a candidate initial state is judged by
the whole trajectory it produces. That is the only difference in the formula, and this
chapter is mostly about how much follows from it.

## The model

Lorenz 63, observed every 0.1 TU with $\sigma_o = 2$, against a background covariance
$\mathbf{B} = \Sigma_{\mathrm{clim}}/64$ — a background error near 1.0 per component.

That last choice is not cosmetic and was not the first one tried. With
$\mathbf{B} = \Sigma_{\mathrm{clim}}/4$, a background as bad as a climatological guess,
the cost function is multimodal enough that a third of cases end up *worse* than the
background, Gauss–Newton makes essentially no progress, and every measurement is
dominated by minimiser failure rather than by the physics it is meant to isolate. An
operational background is a short forecast, not a climatology.

{{< marimo src="/nb/ch18_variational-da.html" >}}

## The cost function is not a bowl

$J_o$ would be an exact quadratic in 3D-Var, since $\mathbf{H}$ is linear. With
$\mathcal{M}$ in the way it is not, and the folding gets worse with the window. Counting
strict local minima on one two-dimensional slice:

| window (TU) | observations | local minima of $J_o$ | of $J$ |
|---|---|---|---|
| 0.5 | 6 | 2 | 1 |
| 1.0 | 11 | 4 | 3 |
| 2.0 | 21 | 38 | 17 |

The background term suppresses multimodality — it adds a convex bowl, which is why the
$J$ column is smaller — but it cannot suppress all of it. On that same slice the analysis
moves from a background 4.01 from the truth to **0.72** at a 0.5 TU window, and to
**5.56** at 2.0 TU: with three and a half times as many observations, *further from the
truth than the background it started from*.

## Five results

**The gradient test's diagnostic is the depth of the trough, not the upturn.** The
standard check $\Phi(\alpha) = [J(x_0+\alpha h) - J(x_0)]/(\alpha h^{\top}\nabla J)$
must tend to 1. Plotted as $|\Phi-1|$ it makes a V, and it is tempting to read the
left-hand rise as the signature of correctness. It is not: that branch comes from
cancellation error in evaluating $J$ and knows nothing about the gradient, so a *wrong*
gradient rises there too. Over the middle decades the exact gradient falls by a factor
of $2\times10^{3}$ to $5\times10^{-8}$; one whose every component is 1 % too large is
flat to within 1 %, and one missing its background term is flat at $0.13$.

**And the test has a blind spot at the obvious place to run it.** At $x_0 = x^b$ the
background term of the gradient, $\mathbf{B}^{-1}(x_0-x^b)$, is exactly zero — so a
gradient missing that term *entirely* produces a curve **bitwise identical** to the
correct one. Test at a displaced point.

**The model acts as a constraint, not an interpolator.** Observe only $x$. With a
diagonal $\mathbf{B}$, 3D-Var's increment lives entirely in $x$ and returns $y$ and $z$
*bitwise unchanged* — it has no mechanism to reach them. 4D-Var cuts the error in the
unobserved $y$ by 37 %, using observations that never touched it, because the trajectory
launched from $x_0$ depends on all three components. This is why operational 4D-Var can
assimilate radiances and bending angles, which are not model variables at all. Honestly:
$z$ improves by only 4 % — observability through the dynamics is real but not uniform.

**The Hessian is the analysis-error covariance, and for linear dynamics it *is* the
Kalman filter's.** 4D-Var never forms a covariance, which is the standard argument
against it. But $\mathbf{A}^{-1} = \nabla^2 J = \mathbf{B}^{-1} + \sum_k
\mathbf{M}_k^{\top}\mathbf{H}^{\top}\mathbf{R}^{-1}\mathbf{H}\mathbf{M}_k$ is the
uncertainty, it depends on *when and where* observations are taken but never on their
values, and it is flow-dependent even though $\mathbf{B}$ is not. On a linear system,
one outer iteration reproduces the Kalman analysis mean to $1.6\times10^{-15}$ and
$\mathbf{M}\mathbf{A}\mathbf{M}^{\top}$ reproduces its analysis covariance to
$4.4\times10^{-16}$.

**Timing is information, and it costs something at the analysis time.** Hold the
observation *count* fixed and move only the timing: six observations of the full state,
all at the analysis time or spread across the window. With a loose background the spread
configuration has **43 % more** analysis-error variance — it is the worse analysis. By a
lead of 0.25 TU it has **2.6× less** forecast-error variance, and it stays ahead at every
lead thereafter. A centre that tuned its observing network on analysis-time scores would
tune away the thing that matters.

## The knob: window length

Lengthening the window admits more observations. It also asks the minimiser to work on a
cost function whose curvature comes from a linearisation with a shelf life. The result is
a broad minimum between **0.8 and 1.2 TU** — 0.7 to 1.1 e-folding times, a *fraction of
the error-doubling time*, which is the same statement
[chapter 15]({{< relref "ch15_tangent-linear-adjoint.md" >}}) made about tangent-linear
validity by a completely independent route.

Both ends fail for opposite reasons. At 0.1 TU there are two observations and the
analysis is worse than the background in 46 % of cases. At 2.0 TU there are 21
observations — more information than any other row — and the error is 3.4 times the
minimum, because the minimiser cannot find the state that fits them. **This is why
operational windows are 6 to 12 hours**, and it is a statement about linearisation, not
about computer time.

## A negative result, kept

The incremental form is exactly Gauss–Newton — the inner normal equations' right-hand
side *is* minus the outer gradient. From a good background (error 1.33) it converges in
six outer iterations to the L-BFGS answer to $1\times10^{-3}$, with a visible quadratic
tail. From a background twice as bad it **stalls completely**, at $J = 451$ where L-BFGS
reaches 13.7.

The failure is Gauss–Newton's, not 4D-Var's: an undamped full step from outside the
quadratic basin overshoots, and a line search would extend the range considerably. But it
establishes *why* the operational configuration is what it is. Incremental 4D-Var is used
where the background is a short forecast, and it is that assumption — not the arithmetic
— that makes the method work. Note also what the incremental form does *not* buy at
$n=3$: counted in adjoint sweeps it is no cheaper than L-BFGS. Its advantage is
structural and appears only at operational size.

## Exercises

1. Push the section 1 slider to $c = 64$ and watch the minimum leave the background. At
   what $c$ does the analysis stop being background-dominated?
2. The analysis error at 0.1 TU is worse than at 0.3 TU, though both are well inside the
   linear regime. What is failing at 0.1 TU, and why is it not a linearisation problem?
3. Section 4's alignment weakens as $\mathbf{B}$ tightens. Predict what happens as
   $\sigma_o \to 0$ at fixed $\mathbf{B}$, and name the term of the Hessian you are
   reasoning about.
4. With a loose $\mathbf{B}$ the spread configuration is worse at lead 0 and better at
   lead 0.25 — the same $\mathbf{A}$ seen through
   $\mathbf{M}\mathbf{A}\mathbf{M}^{\top}$. What must be true of $\mathbf{A}$'s
   eigenvectors for a larger trace to propagate into a smaller one?
5. Derive the 3D-Var claim algebraically: show that for diagonal $\mathbf{B}$ and
   $\mathbf{H} = (1\;0\;0)$ the analysis increment has zero second and third components.

## Where this goes next

[Chapter 19]({{< relref "ch19_ensemble-da.md" >}}) replaces the fixed $\mathbf{B}$ with
an ensemble estimate, buying flow-dependence at the analysis time rather than only inside
the window — at the price of sampling error, localisation and inflation.
[Chapter 20]({{< relref "ch20_da-in-practice.md" >}}) cycles all three schemes and
measures what analysis error does to the forecast. The singular vectors that section 4
compares against are
[chapter 16]({{< relref "ch16_adjoint-sensitivity.md" >}})'s.

## Further reading

- Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, ch. 5
  *[citation needed: section numbers]*
- Courtier, Thépaut & Hollingsworth (1994), on the incremental formulation
  *[citation needed]*
- Talagrand & Courtier (1987), on the adjoint in variational assimilation
  *[citation needed]*
- Fisher & Andersson, on the Hessian as analysis-error covariance in operational practice
  *[citation needed]*
