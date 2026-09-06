---
title: "Chapter 23 · Boundary-forced predictability and the S2S window"
weight: 623
part: "Part VI — Predictability of the second kind: from S2S to climate"
knob: 'forcing period and amplitude'
status: "live"
---

## Overview

Weather forecasts are useful for about a week. Seasonal forecasts of the coming winter
are useful too — issued months ahead, when no trace of today's weather can possibly
survive. Both cannot be initial-value problems. **What is the seasonal forecast actually
using?**

[Chapter 1]({{< relref "ch01_what-is-predictability.md" >}}) measured the answer's shape:
information carried by today's state decays to nothing while information carried by the
*forcing* does not decay at all, and the two cross. That chapter held the forcing fixed.
This one lets it move.

## The model

Lorenz 63 with a slowly oscillating Rayleigh number,
$\rho(t) = 28 + 6\sin(2\pi t/40)$, with a period an order of magnitude longer than the
trajectory's own predictability time. That separation is what makes $\rho$ a *boundary
condition* rather than part of the fast dynamics — the cheapest caricature of ENSO, of
the seasonal cycle, or of any other slow driver.

The whole argument is one decomposition: measure each forecast distribution against
**two** climatologies — the one pooled over all forcing phases, and the one conditioned
on the phase at verification time.

{{< marimo src="/nb/ch23_boundary-forced-s2s.html" >}}

## Four results

**Forecast information decays to a floor, not to zero.** Against the pooled climatology
it falls from 2.93 nats and settles at **0.146**; the forcing phase alone is worth
**0.120**. At long lead the forecast has stopped being a forecast in the ordinary sense —
it has become a statement about the forcing.

**And the two contributions add up.** The floor plus what remains of the initial
condition, $0.120 + 0.027 = 0.147$, against a measured plateau of 0.146 — agreement to
1 %. Relative entropy is not additive in general, so this is a *check* rather than an
identity; that it holds this closely says the two sources are, here, close to
independent. The forcing overtakes the initial state at **lead 17**.

**Windows of opportunity are real.** Resolved by the phase a forecast was *launched* at,
the lead at which the initial-state signal is spent runs from 9 to 17 TU — a factor of
1.9. With fifteen launches per bin that could be sampling noise, so it is checked:
splitting the launches into two independent halves and asking whether they agree about
which phases are the predictable ones gives a correlation of **+0.96**.

**Amplitude is what the forcing is worth; period mostly is not.** Phase information rises
monotonically with amplitude (0.0015 → 0.252 nats). Across the slow range, an eightfold
change in period moves it by about 10 %. Once the forcing is slow enough for the system
to equilibrate to whatever $\rho$ currently is, making it slower adds essentially nothing.

## The predictability desert, and why it is not empty

Between the two regimes lies the well-known awkward gap: too far ahead for the initial
state to help much, not far enough for the boundary signal to be the whole story. Here it
runs from about lead 10 to lead 17.

The useful point is that **the desert has a floor under it**. Skill there is small but not
zero, and what is left is boundary-forced — which is exactly why subseasonal forecasting
is worth attempting rather than a category error. The gap is thin, not empty.

## A result that came out backwards

At a forcing period of 2.5 TU the phase information *rises* to 0.167 nats — above every
slow value. That is not the forcing becoming more informative. **It is the decomposition
breaking down.**

With a period shorter than the trajectory's own predictability time, "the phase" is no
longer a slowly-varying boundary condition the state has forgotten about; it is strongly
correlated with where the state is *right now*. Conditioning on it is then partly
conditioning on the initial condition, and the split into first-kind and second-kind
information stops meaning what it says.

The clean separation this chapter relies on is **a property of the timescale gap, not of
the mathematics**, and it is worth knowing that it fails rather than assuming it holds.
Real slow drivers — ENSO at three to seven years against synoptic weather at days — sit
very far from that failure, which is precisely why the framing is useful for them.

## Care taken

The zero-amplitude control matters: with no forcing the phase label is meaningless, so
the phase-conditioned climatology must equal the pooled one. Measured, boundary
information collapses to **0.0015 nats** — the estimator's floor — against 0.120 at the
working amplitude, a factor of 80. Without that control the whole decomposition could
have been measuring an artefact of the binning.

The windows-of-opportunity claim is *not* simply a restatement of which phases are most
informative: across the eight bins the two correlate at +0.66, which with eight points is
suggestive rather than established. They are different questions — one asks what the
boundary condition is worth, the other how long the initial condition survives.

## Exercises

1. The two curves in section 2 start at nearly the same value and separate. At what lead
   do they first differ by more than 10 %, and what does that lead correspond to?
2. The floor-plus-residual check agrees to about one per cent. Construct a case where it
   would fail badly — what would have to be true of the two information sources?
3. The phases with the most informative forcing are not exactly the phases with the
   longest-lasting forecasts. Which of the two would you rather know about, as a user
   deciding whether to act on a forecast?
4. Section 4 shows the decomposition failing at short forcing period. Estimate, from
   chapter 1's numbers, the shortest period for which you would trust it.

## Further reading

- Lorenz (1975), on predictability of the second kind *[citation needed]*
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on seasonal
  prediction *[citation needed: chapter]*
- Mariotti et al., on the subseasonal gap and windows of opportunity *[citation needed]*
- Shukla (1998), on predictability in the midst of chaos *[citation needed]*
