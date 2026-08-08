# FALSIFIER-X AIR — Research Design v1.0

## Research question

Can an aviation AI system detect when its current causal structure is insufficient
to explain disruption propagation, actively distinguish constrained candidate
mechanisms through interventions, recover the missing structure under controlled
ground truth, and improve downstream recovery?

## Structural adequacy

Evidence families:
- predictive residual
- uncertainty violation
- persistence
- graph-structured residual concentration
- cross-model consistency

Gates:
1. calibrated uncertainty
2. OOD/distribution shift
3. persistence
4. structured propagation
5. cross-model consistency

The prototype uses a transparent weighted score. The production research version
must calibrate weights and thresholds on validation data and freeze them before
test evaluation.

## Candidate mechanism constraints

Candidates are typed graph edits:
(source_type, relation_type, target_type, context, expected_effect)

They must satisfy local-subgraph, temporal-order, aviation-ontology, type,
and intervention-feasibility constraints.

## Active experiments

Choose an intervention maximizing predicted disagreement between candidates
while penalizing intervention cost. The prototype uses a transparent rule;
the research version should compare it with random and passive selection and
investigate information-gain/Bayesian alternatives.

## Digital Twin

The Twin contains mechanisms that can be hidden from the AI. It must support
reproducible seeds, interventions, counterfactual replay, held-out mechanisms,
held-out network configurations, and ground-truth export.

## Evaluation

Measure:
- prediction error
- inadequacy false-positive rate and detection delay
- mechanism structural precision/recall and SHD
- intervention count/cost
- recovery delay/cancellation/recovery-time metrics

Mandatory baselines:
non-graph ML, GNN/ST-GNN, causal baseline where feasible, random/passive
experiment selection, and FALSIFIER-X ablations.

## Anti-circularity

Use independent seeds, held-out mechanisms, held-out topologies, held-out
parameter ranges, and a separate final test set. Real-world aviation data is
for secondary investigation, not automatic proof of new causal mechanisms.
