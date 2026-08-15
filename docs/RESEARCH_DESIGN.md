# FALSIFIER-X AIR research design

## Research motivation and novelty boundary

This project investigates a closed-loop workflow for dynamic aviation disruption
networks: prediction → adequacy evidence → localisation → constrained mechanism
hypotheses → active intervention → falsification → repair proposal → held-out
validation → recovery comparison. Its proposed contribution is the integration
and aviation-specific adaptation of these components, not the invention of GNN
delay prediction, causal inference, digital twins, or sequential falsification.

The graph/spatio-temporal prediction motivation includes Zhu et al. (IEEE T-ITS,
2024, doi:10.1109/TITS.2024.3443261). Sequential hypothesis testing is motivated
by Huang et al., *Automated Hypothesis Validation with Agentic Sequential
Falsifications* (ICML 2025). Neither is copied by this implementation.

## Current architecture

`schema.py` defines typed, mostly immutable contracts. `graph.py` constructs an
observable multi-relational graph with flight, airport, scheduled rotation,
airport-association, and shared-resource semantics. A flight's optional
observable `resource_id` creates a flight-to-resource `USES_RESOURCE` edge and a
time-ordered `RESOURCE_DEPENDENCY` edge between flights sharing it. These edges
express operational opportunity for propagation, not proof that the hidden twin
mechanism is active.

`predictor.py` provides the replaceable `DelayPredictor` boundary. The current
Ridge baseline deliberately excludes graph inputs and uses training-residual
prediction intervals. `adequacy.py` combines standardized residual, interval
miscoverage, persistence, residual concentration, residual agreement, and an OOD
gate. It produces an *adequacy evidence score*, not a causal test or p-value.
Thresholds must be calibrated on a validation split and frozen before evaluation.

`mechanisms.py` contains a semantic registry, not a manually scored experiment
table. Candidates exist only when their relation type is present in the affected
observable subgraph. `experiments.py` derives candidate intervention effects from
observed residual structure and graph topology. It selects the untried
intervention with maximum cost-normalised predicted outcome disagreement, runs
only that intervention, then updates likelihood-shaped evidence. This is an
explicit non-Bayesian expected-information approximation.

`twin.py` is a synthetic environment. Its resource mechanism uses the same
observable shared-resource assignment as the graph, while keeping whether that
mechanism is active private. The learner can call only `observe` and
`counterfactual`. Paired seeds hold exogenous noise fixed for an intervention.
The evaluator may vary scenario seeds and use `held_out_scenarios`; no learner
path reads hidden mechanism state.

`recovery.py` chooses an action from predicted delay, recovered structure, and
the median measured paired effect of that mechanism's matching experiments. It
does not assume a fixed benefit and does not accept a twin.
`evaluate_selected_action` is intentionally a separate, post-decision
environment evaluation.

## What remains future work

- Baseline model comparisons and a calibrated graph/spatio-temporal model.
- A calibrated graph/spatio-temporal model (for example an ST-GNN) behind the
  existing predictor protocol.
- Validation-set calibration of adequacy and evidence thresholds.
- Richer resource and connection edges only where reliable operational data exists.
- Formal held-out topology/mechanism-strength studies and active/random/exhaustive
  experiment-efficiency baselines.
- Operationally realistic recovery constraints and outcome metrics.

## Real-data boundary

The data package ingests configurable BTS Reporting Carrier On-Time Performance
files and NOAA/NCEI Global Hourly ISD station observations. BTS schedule, actual
times, delay outcomes, cancellations, diversions, tail numbers and airport IDs
are **observed**. Airport/station matching, timezone-aware operational timestamps,
weather features, rotation edges and airport-temporal propagation edges are
**derived** from observed data and documented configuration. They are not causal
claims.

The existing hidden-mechanism twin, controlled interventions, and counterfactual
ground truth remain **synthetic**. Real BTS/NOAA records support future prediction
and graph evaluation; they do not prove a causal disruption mechanism. Weather
alignment is backward-only to scheduled departure, and splits are chronological;
delay outcomes and actual-arrival information are not prediction-time features.

## Evaluation plan

Use disjoint seeds, topology families, disruption locations, and mechanism
strength ranges. Measure interval coverage, adequacy false positives/detection
delay, structural precision/recall, intervention count/cost, and recovery regret
against the twin. Compare active selection with random and exhaustive baselines.
Success on the discovery scenario alone is insufficient: a proposed repair must
improve prediction and recovery on held-out synthetic scenarios before any claim
about the testbed is made.
