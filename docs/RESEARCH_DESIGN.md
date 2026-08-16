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

## Phase 7 model criticism and falsification protocol

Phase 7 keeps the real-data and controlled-twin tracks separate. BTS+NOAA,
the chronological splits, Phase-5 ST-GNN and Phase-6 CQR are fixed inputs to
the real-data track. The controlled twin is the only source of known mechanism
truth, and therefore the only setting in which mechanism recovery is measured.
No claim of real-world causal recovery follows from twin results.

The learner (`adequacy.py`, `falsifier.py`, `experiments.py`,
`mechanisms.py`, and `recovery.py`) consumes only observable flights, graph
relations, outcomes, predictions, uncertainty, candidate semantics, and paired
intervention outcomes. It does not import the twin or query its private oracle.
`evaluation.py` is the dedicated benchmark layer permitted to call the private
benchmark oracle for hidden mechanisms, affected nodes/edges, and scoring.
An architectural-isolation test enforces that boundary.

Adequacy has five explicit diagnostic channels: prediction/interval violation,
persistence, graph concentration, residual structure, and OOD feature distance.
Frozen validation thresholds yield `ADEQUATE`, `OOD`,
`STRUCTURALLY_SUSPICIOUS`, or `INCONCLUSIVE`. These are evidence categories,
not causal p-values. A large residual alone cannot trigger candidate generation.

For a structural finding, the loop is sequential: localise high-residual flight
neighbourhoods; generate only candidates justified by observable relation types;
select one untried intervention; observe the paired counterfactual; update
transparent likelihood-shaped (non-Bayesian) evidence; reject/retain; and only
recover a single survivor after a confirming experiment clears the frozen support
threshold. Inconclusive is retained as a scientific outcome. The evidence model
assumes a single dominant mechanism; `MULTIPLE` is intentionally reported as a
stress test rather than tuned to pass.

Candidate registry entries specify their required observable relation,
intervention/cost, expected effect signature, and falsification condition.
`DECOY` supplies a correlated observable resource relation while the true twin
mechanism is rotation, so paired interventions must discriminate rather than
correlation alone selecting a cause.

Scenario families are `CORRECT`, `NOISE`, `OOD`, `AIRCRAFT_ROTATION`,
`RESOURCE_DEPENDENCY`, `AIRPORT_CAPACITY`, `MULTIPLE`, `DECOY`,
`STRENGTH_SWEEP`, `NONSTATIONARY`, and `UNSEEN_TOPOLOGY`. Exogenous noise is
paired by seed for each counterfactual. Discovery/calibration seeds are
100--109; final benchmark discovery seeds are 1000--1009 and never overlap.
Each final policy is then evaluated only on its paired held-out seed
`100000 + discovery_seed`, never on the scenario used to generate evidence.

### Phase-7 discovery-partition diagnosis (seeds 100--109 only)

The original final benchmark is not used for calibration. A separate
`discovery_diagnosis.py` trace records, for each allowed discovery seed, hidden
label (evaluation only), observable delay/residual vector, adequacy channels,
localisation, candidate set, every paired intervention effect, candidate evidence
and decision. The diagnostic rejects any seed outside 100--109.

The diagnosis found an implementation defect in the original rotation/resource
effect signature. Their true interventions produced measurable paired effects,
but the learner predicted effect from residual concentration. For discovery seed
100 this predicted 4.54 against a rotation effect of 17.20, and 3.22 against a
resource effect of 18.21; the evidence update therefore rejected the correct
candidate. The minimal correction predicts a represented chain intervention from
the sum of observable predecessor delays on its observable relation edges times
the registry coefficient. It uses neither hidden state nor mechanism truth.
At seed 100 this gives 17.20 for rotation and 12.14 for resource; the latter is
retained (log evidence -0.95), while incompatible candidates are rejected.

This correction does **not** alter the single-dominant-mechanism evidence model,
adequacy thresholds, or active selector. On the discovery partition, both
rotation and resource are structurally suspicious/recovered in 3/10 scenarios;
the first failure in the other 7/10 is adequacy (`INCONCLUSIVE`) before
localisation/candidate generation. Thus the remaining limitation is sensitivity
of the current uncertainty/graph-concentration gate under small noisy networks,
not intervention non-identifiability in detected cases. Decoy cases have the
same observable opportunity but only recover the true rotation candidate; no
resource decoy recovery is introduced. Correct/no-hidden and increased-noise
controls have zero recovery; OOD is routed to `OOD` when its observable score is
supplied. Active selection remains unchanged because the small candidate set and
one mechanism-specific intervention per candidate provide little opportunity for
a discrimination-efficiency advantage over baselines.

### Adequacy sensitivity refinement (discovery seeds 100--109 only)

The remaining `INCONCLUSIVE` cases were not caused by prediction/uncertainty:
all chain cases had interval-violation and persistence evidence. They failed
because scalar residual concentration was below 0.25; notably, that statistic
was often larger in correctly specified/noise worlds, so lowering its threshold
would be indefensible.

The minimal added channel is **directional residual lift**: for observable
`AIRCRAFT_ROTATION` and `RESOURCE_DEPENDENCY` edges, it is the mean positive
standardized residual of scheduled successors minus that of predecessors. It
uses only the observable graph, residuals, and interval scale. In discovery,
chain worlds ranged from 4.43 to 6.58, while correctly specified/noise/OOD
controls had maximum positive lift 0.69. The threshold 3.0 was selected between
these separated discovery ranges and is now frozen; it was not selected from
the final partition. Structural suspicion still requires prediction/uncertainty
violation and persistence, plus either this directional signal or the original
concentration evidence. Thus a large residual alone remains insufficient.

Before this refinement, rotation and resource were detected/recovered 3/10 each
on discovery. After it, both are detected/recovered 10/10; correct and noise
remain `ADEQUATE` with 0/10 recovery, OOD remains `OOD` 10/10, and the decoy
family is detected/recovered as the true rotation mechanism 10/10 with no
resource-decoy recovery. This validates sensitivity within the small controlled
twin, not a general guarantee. The exact diagnostic trace and summary are
written to `data/processed/phase7_discovery_diagnosis.{json,md}`.

Evaluation reports detection sensitivity/specificity and false structural
discoveries; node/edge precision, recall, F1 and top-k where a ranking is
available; mechanism precision/recall/F1, false recovery, experiments and cost;
and recovery-policy delay change on held-out seeds. `RANDOM`, `MAX_EFFECT`,
`ACTIVE`, and `EXHAUSTIVE` are transparent selector baselines. `set_metrics`
has an independently hand-checked test (prediction `{A,B,C}`, truth `{B,C,D}`
gives precision=recall=F1=2/3), rather than only duplicate metric code.

Mechanism recovery, model repair, and operational recovery are distinct. The
current implementation validates only a learned operational intervention policy
on held-out twin scenarios; it does not refit or alter the locked predictor.
`python -m falsifier_x_air.evaluation` writes `phase7_results.json` and a concise
`phase7_report.md` beneath `data/processed/`, including Python/Numpy versions,
seeds, selector, counts/costs, configurations, and metrics.

## What remains future work

- Baseline model comparisons and a calibrated graph/spatio-temporal model.
- A calibrated graph/spatio-temporal model (for example an ST-GNN) behind the
  existing predictor protocol.
- Validation-set calibration of adequacy and evidence thresholds.
- Richer resource and connection edges only where reliable operational data exists.
- Calibrating the frozen adequacy/evidence thresholds against a larger discovery
  partition and expanding the benchmark's active-vs-baseline aggregation.
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
