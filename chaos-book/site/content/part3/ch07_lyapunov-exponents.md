---
title: "Chapter 7 · Lyapunov exponents and doubling times"
weight: 307
part: "Part III — Quantifying chaos and predictability"
lectures: "L4–5, L7"
knob: '$\rho$ across the transition'
status: "planned"
---
## Overview

'Chaotic' becomes a number here. The Lyapunov spectrum measures the average rate at which nearby trajectories separate along each direction in phase space, and its leading member $\lambda_1$ fixes the error-doubling time $\ln 2/\lambda_1$. The chapter is careful about a distinction that is routinely blurred: the asymptotic spectrum is a property of the **attractor**, while the finite-time exponent is a property of a **state and a lead time** — and it is the second that decides whether today's forecast is unusually good.

## The planned notebook

The Lorenz 63 spectrum by the Benettin algorithm, with the exact identity $\sum_i \lambda_i = -(\sigma+1+\beta)$ displayed live as a check the reader can watch hold. `chaoslib.lyapunov.lyapunov_spectrum`, `finite_time_exponents`.

## Further reading

- Palmer & Hagedorn (2006), ch. 2; Kalnay (2003), §6.1
