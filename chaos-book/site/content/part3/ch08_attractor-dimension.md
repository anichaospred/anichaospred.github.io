---
title: "Chapter 8 · Attractors, fractal dimension, and entropy"
weight: 308
part: "Part III — Quantifying chaos and predictability"
lectures: "L7"
knob: 'the scaling window'
status: "planned"
---
## Overview

The Lorenz attractor is more than a surface and less than a volume: its dimension is about 2.06. This chapter introduces the correlation and Kaplan–Yorke dimensions and the Kolmogorov–Sinai entropy, which by Pesin's identity equals the sum of the positive Lyapunov exponents and measures the rate at which the system destroys information about its own initial state.

## The planned notebook

$\ln C(r)$ against $\ln r$ with the fit window exposed as a slider — so the reader discovers for themselves how easily a fractal dimension is mis-measured by fitting outside the scaling range, and can compare the trajectory-based estimate against the dynamics-based one. `chaoslib.dimension`, `chaoslib.lyapunov.kaplan_yorke_dimension`.

## Further reading

- Palmer & Hagedorn (2006), ch. 2 *[citation needed]*
