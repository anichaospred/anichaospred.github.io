---
title: "Chapter 20 · Data assimilation in practice"
weight: 520
part: "Part V — The machinery of prediction"
knob: 'observation density and error'
status: "planned"
---
## Overview

Cycling data assimilation makes the analysis error the floor on forecast error, which finally quantifies the value of better observations — and the answer is sobering. Because error grows exponentially, reducing the analysis error by a factor of ten buys $\Delta t = \ln 10 / \lambda \approx 6.5$ extra days for the atmosphere, and the *next* factor of ten buys the same 6.5 days again. Observing-system improvements return logarithmically.

## The planned notebook

Builds on the existing `da_l63_tutorial` material — 3D-Var, 4D-Var and the EnKF cycling on the same Lorenz 63 system, so the three are compared on equal terms rather than in isolation.

## Further reading

- Kalnay (2003), ch. 5; Palmer & Hagedorn (2006), ch. 8 *[citation needed]*
