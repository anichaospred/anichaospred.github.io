---
title: "Chapter 18 · Variational data assimilation"
weight: 518
part: "Part V — The machinery of prediction"
knob: 'window length, $\mathbf{B}$'
status: "planned"
---
## Overview

3D-Var and 4D-Var as the minimisation of one cost function balancing a background against observations. The interesting limitation is structural: 3D-Var holds the background error covariance $\mathbf{B}$ fixed for all time, so it cannot know that today's background error lies along the flow's growing directions. Everything after it in this Part is an attempt to fix that.

## The planned notebook

4D-Var on Lorenz 63 with the cost-function descent shown live, and the adjoint supplying the gradient. `chaoslib.assimilate.four_dvar_analysis` — note it uses a quasi-Newton minimiser, because fixed-step descent on this cost function diverges whenever the observations are accurate.

## Further reading

- Kalnay (2003), ch. 5; Palmer & Hagedorn (2006), ch. 8 *[citation needed]*
