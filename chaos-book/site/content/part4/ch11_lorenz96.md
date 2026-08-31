---
title: "Chapter 11 · Lorenz 96: a many-variable atmosphere analogue"
weight: 411
part: "Part IV — Many scales, many degrees of freedom"
lectures: "L4–5, L7"
knob: '$F$, $N$'
status: "planned"
---
## Overview

Forty variables on a cyclic chain, with quadratic advection, linear damping and constant forcing. Lorenz 96 is the standard testbed for everything in Part V because it is the smallest system that behaves like a **field**: errors have a spatial scale, they propagate, and the Lyapunov spectrum has structure rather than a single positive exponent. At $N=40$, $F=8$ there are exactly thirteen positive exponents and a Kaplan–Yorke dimension near 27.

## The planned notebook

A Hovmöller diagram beside the live Lyapunov spectrum, with $F$ and $N$ as sliders so the reader can find the onset of chaos and watch the number of positive exponents grow. `chaoslib.systems.lorenz96`, `chaoslib.lyapunov`.

## Further reading

- Lorenz (1996) *[citation needed]*; Kalnay (2003), §5.5 *[citation needed]*
