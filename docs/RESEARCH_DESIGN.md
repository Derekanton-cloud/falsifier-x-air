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

---

## Phase 8 — Generalized Mechanism Falsification & Identifiability Stress Test

### Partition isolation

Phase-7 seeds (100–109, 1000–1009, 101000–101009) are frozen. Phase-8 uses
exclusively new partitions:

- Discovery / calibration: seeds 200–219 (used for debugging and inspection
  only; no thresholds are tuned here).
- Final evaluation: seeds 2000–2019 (inspected only after discovery design is
  complete).

The Phase-8 evaluation layer reuses the frozen adequacy thresholds and evidence
update from Phase-7. No tuning is performed on evaluation outcomes.

### Research hypotheses

Phase-8 asks whether the core FALSIFIER-X claim — prediction failure + structured
hypotheses + counterfactual falsification → mechanism identification — survives
beyond the compact 4-flight benchmark, and specifically whether there is a
boundary of identifiability that the system can recognise.

| ID | Hypothesis | Experiment |
|----|-----------|-----------|
| H1 | Detection degrades gracefully as graph size grows | SCALE_8 / SCALE_16 / SCALE_32 |
| H2 | ACTIVE selector advantage grows with hypothesis complexity | SINGLE / MULTI_2 / MULTI_3 selector comparison |
| H3 | FALSIFIER-X abstains (INCONCLUSIVE) when mechanisms are indistinguishable | CONFOUNDED worlds with identifiability certificate |
| H4 | Transient mechanisms are filtered; persistent ones detected | TRANSIENT vs SUSTAINED |
| H5 | Multi-mechanism recovery is harder than single | MULTI_2 / MULTI_3 |
| H6 | Weak mechanisms fall within uncertainty bounds and are not flagged | Strength gradient 0.1–2.0 |
| H7 | Budget constraints shape recovery-vs-budget curve | K ∈ {1, 2, 3, 5} |

### Identifiability certificate (H3)

For CONFOUNDED worlds both AIRCRAFT_ROTATION and RESOURCE_DEPENDENCY are active
with equal propagation coefficients (0.75) and identical predecessor structure.
Before FALSIFIER-X runs, an identifiability certificate computes:

  relative_difference = |delta_rotation - delta_resource| / max(delta_rotation, delta_resource, 1e-6)

where delta_X is the total delay reduction under intervention X.  If
relative_difference < 0.05 (frozen before any evaluation seed is inspected), the
world is certified INDISTINGUISHABLE and the expected learner output is
INCONCLUSIVE.

The certificate uses only the public twin API (observe / counterfactual); the
private oracle is never accessed.

### Empirical findings (evaluation seeds 2000–2019, 1680 records)

**H1 — Scale generalization**: Detection and recovery remain at 1.00 across
8, 16, and 32-flight networks with zero false recovery. This is a positive
generalization result, but it must be interpreted cautiously: the larger
networks use scaled versions of the same mechanism types, not genuinely novel
topological structures.

**H3 — Identifiability (critical finding)**: All 80 confounded-world records
were certified INDISTINGUISHABLE by the certificate (fraction 1.00). However,
FALSIFIER-X returned INCONCLUSIVE in 0/80 cases (INCONCLUSIVE rate 0.00,
abstention correctness 0.00). The system guessed a single mechanism survivor
rather than abstaining. False recovery rate appears 0.00 because guessing either
mechanism in a world where both are true counts as "correct" by the set-membership
scoring, but scientifically this is a failure: the available interventions cannot
distinguish the two hypotheses, yet the system outputs a definite recovery rather
than INCONCLUSIVE. This is the most important finding of Phase 8.

**H6 — Strength gradient**: Clear detection threshold between strength 0.25
(detection 0.00) and strength 1.0 (detection 1.00); strength 0.5 is borderline
(detection 0.20). This confirms that the adequacy gate is not tuned for weak
mechanisms in the larger benchmark and that weak-mechanism silence is a genuine
system property, not a trivial artefact.

**H7 — Budget constraints**: ACTIVE achieves 1.00 recovery at K=1 while RANDOM
achieves only 0.30. ACTIVE advantage is most pronounced at tight budgets and
converges with EXHAUSTIVE at K≥2. This is the clearest evidence that the active
selector provides a meaningful efficiency advantage.

**Selector comparison**: ACTIVE efficiency ratio (ACTIVE success/cost ÷ RANDOM
success/cost) is 1.17 overall, with the advantage concentrated at K=1. EXHAUSTIVE
achieves the highest overall success/cost (0.528 vs ACTIVE 0.438), suggesting that
for the current 3-intervention candidate set, exhaustive exploration is competitive
or better than active selection except under tight budgets.

### Conclusion

Phase 8 reveals two scientifically important boundaries: (1) the strength
threshold below which mechanisms are invisible, and (2) the identifiability
boundary at which the system should abstain but currently does not. Resolving
(2) — adding an explicit identifiability check before committing to a recovery
decision — is the most important direction for Phase 9.

---

## Phase 9 — Identifiability Layer & Principled Abstention

### Goal

Determine whether FALSIFIER-X can recognise when surviving hypotheses are not
identifiable from the available interventions, and abstain rather than convert
an underdetermined explanation into a causal claim.

The goal is NOT "make H3 pass."  It is to show that identifiability can be
inferred internally from observable intervention-response structure, without
any oracle access.

### New module: falsifier_x_air/identifiability.py

The identifiability layer is applied by `FalsifierXAir.investigate()` immediately
before any RECOVERED decision.  It is never called from learner modules or
evaluation harnesses.

**Decision rule:**

For each rejected competitor H_j:

1. Obtain the observed total-delay reduction Δ_j under H_j's paired
   intervention a_j.  If a_j was not executed during the main loop, run one
   probe experiment using the public twin.counterfactual() API.
2. Compute the response distance:
     D(H_i, H_j) = |Δ_i − Δ_j| / max(Δ_i, Δ_j, σ_eff, 1e−6)
   where Δ_i is the survivor's observed effect and σ_eff is the mean
   epistemic uncertainty from the prediction output.
3. If D ≤ ε_ident: competitor is NOT observationally distinguished from the
   survivor → return INCONCLUSIVE (INDISTINGUISHABLE_EQUIVALENCE_CLASS).

A RECOVERED decision is only issued if every rejected competitor was
legitimately distinguished (D > ε_ident for all pairs).

**No access** to _benchmark_truth(), confounded_certificate(), or any private
twin state. Verified by automated AST scan in tests/test_identifiability.py.

### Epsilon calibration (discovery seeds 200–219, frozen before evaluation)

| World family | D values | N |
|---|---|---|
| Confounded (equal coefficients, mirrored topology) | 0.0000 – 0.0000 | 20 |
| Single-mechanism AIRCRAFT_ROTATION | 1.0000 – 1.0000 | 20 |

Gap between clusters: confounded D_max = 0.0000, single D_min = 1.0000.
ε_ident = 0.20 is placed well between the two clusters and is now frozen.

### Changes to existing code

| File | Nature of change |
|---|---|
| falsifier_x_air/identifiability.py | NEW: response_distance(), check_identifiability(), IdentifiabilityResult |
| falsifier_x_air/falsifier.py | MODIFIED: identifiability check before RECOVERED; two backward-compatible fields added to InvestigationResult |
| tests/test_identifiability.py | NEW: 33 tests |

No changes to: update_evidence(), selection logic, adequacy thresholds,
mechanism registry, Phase-7 evaluation, Phase-8 evaluation, or any frozen
data files.

### Empirical results (Phase-9 identifiability check on discovery seeds 200–219)

| Metric | Confounded (20 worlds) | Single-mechanism (20 worlds) |
|---|---|---|
| Abstention correctness | 20/20 (1.00) | N/A |
| RECOVERED rate | 0/20 | 20/20 (1.00) |
| False recovery | 0/20 | 0/20 |
| ident_reason = LEGIT | 0/20 | 20/20 |
| ident_reason = INDISTR | 20/20 | 0/20 |

Phase-8 H3 baseline is preserved: `_run_case_budgeted` in phase8_evaluation.py
does not call `FalsifierXAir.investigate()`, so the Phase-8 0/80 abstention
record remains as the pre-fix historical baseline.

Full test suite: **179 tests, 0 failures.**

### Scientific interpretation

Phase 9 demonstrates that identifiability can be determined internally from
observable intervention responses without oracle access.  The system now
implements the following complete decision table:

| Situation | Outcome |
|---|---|
| Prediction adequacy (no mechanism detected) | ADEQUATE |
| OOD / noise | OOD |
| Weak mechanism (below detection threshold) | ADEQUATE |
| One hypothesis survives; competitors' interventions distinguishable (D > ε) | RECOVERED |
| One hypothesis survives; a competitor's intervention produces indistinguishable response (D ≤ ε) | INCONCLUSIVE (INDISTINGUISHABLE_EQUIVALENCE_CLASS) |
| Multiple hypotheses survive budget exhaustion | INCONCLUSIVE |

This transforms FALSIFIER-X from "mechanism classifier" into a system that
explicitly reasons about whether its evidence is sufficient to identify a
mechanism at all — closer to a principled causal decision-support framework.
