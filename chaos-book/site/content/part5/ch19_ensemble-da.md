---
title: "Chapter 19 · Ensemble data assimilation"
weight: 519
part: "Part V — The machinery of prediction"
knob: 'ensemble size, localisation radius'
status: "planned"
---
## Overview

The ensemble Kalman filter estimates the background error covariance from the ensemble itself, so it is flow-dependent for free — the whole appeal. The price is sampling error: an ensemble of twenty cannot estimate covariances between distant points, and those entries are noise. Localisation and inflation are the standard remedies, and the chapter shows the filter divergence that follows when they are omitted.

## The planned notebook

EnKF on Lorenz 96 with ensemble size, localisation radius and inflation as sliders, and the filter-divergence cliff findable by the reader. `chaoslib.assimilate.enkf_update`, `gaspari_cohn`.

## Further reading

- Kalnay (2003), ch. 5; Houtekamer & Zhang (2016) *[citation needed]*
