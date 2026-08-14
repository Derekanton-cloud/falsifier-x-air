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
observable multi-relational graph with flight, airport, scheduled rotation, and
airport-association semantics. A scheduled edge is an available operational
dependency, not proof that it is causally active.

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

`twin.py` is a synthetic environment. Hidden active mechanisms are private; the
learner can call only `observe` and `counterfactual`. Paired seeds hold exogenous
noise fixed for an intervention. The evaluator may vary scenario seeds and use
`held_out_scenarios`; no learner path reads hidden mechanism state.

`recovery.py` chooses an action from predicted delay and recovered structure
without accepting a twin. `evaluate_selected_action` is intentionally a separate,
post-decision environment evaluation.

## What remains future work

- Real flight, weather, airport-capacity, resource, and passenger data ingestion.
- A calibrated graph/spatio-temporal model (for example an ST-GNN) behind the
  existing predictor protocol.
- Validation-set calibration of adequacy and evidence thresholds.
- Richer resource and connection edges only where reliable operational data exists.
- Formal held-out topology/mechanism-strength studies and active/random/exhaustive
  experiment-efficiency baselines.
- Operationally realistic recovery constraints and outcome metrics.

## Evaluation plan

Use disjoint seeds, topology families, disruption locations, and mechanism
strength ranges. Measure interval coverage, adequacy false positives/detection
delay, structural precision/recall, intervention count/cost, and recovery regret
against the twin. Compare active selection with random and exhaustive baselines.
Success on the discovery scenario alone is insufficient: a proposed repair must
improve prediction and recovery on held-out synthetic scenarios before any claim
about the testbed is made.
