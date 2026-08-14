# FALSIFIER-X AIR — research foundation v0.1

FALSIFIER-X AIR is a synthetic research testbed for a specific question: when a
dynamic aviation disruption model persistently fails, can it localise the
unexplained pattern, test constrained missing-dependency hypotheses through
counterfactual interventions, and use surviving structure to improve a recovery
choice?

It is **not** a claim of a new GNN, a causal-discovery guarantee, or a real-world
aviation control system.

## What is implemented

`python -m falsifier_x_air.demo` demonstrates this closed loop:

1. Build an observable flight/airport/rotation graph.
2. Produce tabular baseline predictions and residual intervals.
3. Assess OOD, persistence, interval violations, residual agreement and graph concentration.
4. Localise high-residual flights and generate only graph-semantic candidates.
5. Select and run one informative counterfactual at a time in a synthetic twin.
6. Update approximate evidence, reject inconsistent candidates, and propose a repair.
7. Choose recovery from model estimates; only then evaluate that choice in the twin.

The twin keeps hidden active mechanisms private. The learner only receives
observations and counterfactual outcomes. Experiments are paired by scenario seed
so an intervention is compared with the same synthetic exogenous noise.

## Deliberate limitations

The included Ridge predictor is a non-graph baseline; `DelayPredictor` is the
replacement boundary for a future spatio-temporal GNN. The twin is synthetic and
its mechanism effects are not evidence about real aviation. Adequacy thresholds
and likelihood-shaped evidence updates are configurable heuristics requiring
validation calibration; they are not causal p-values or Bayesian posteriors.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m falsifier_x_air.demo
```

See [the research design](docs/RESEARCH_DESIGN.md) for boundaries, evaluation
plan, and future phases.
