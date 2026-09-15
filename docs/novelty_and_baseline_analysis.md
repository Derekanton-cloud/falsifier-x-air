# FALSIFIER-X AIR: Novelty and External Baseline Analysis

## 1. Scientific Question

Does the following closed loop already exist in prior work in substantially equivalent form?

  predictive model inadequacy
  -> structural hypothesis generation
  -> active intervention
  -> falsification / discrimination
  -> explicit identifiability assessment
  -> abstention when indistinguishable
  -> mechanism recovery
  -> mechanism-derived predictive model repair
  -> blind predictive validation

## 2. Literature Search Areas

Searched: model criticism, model debugging, model repair, causal discovery,
active causal learning, causal intervention, SEM repair, concept drift,
digital twin diagnostics, causal model revision, uncertainty-aware diagnosis.

---

## 3. Selected Prior Work and Capability Analysis

### 3.1 ABCD (Lloyd et al., 2014)
- Problem: Automated model criticism via kernel composition.
- Interventions: None.
- Identifiability gate: None.
- Model repair: None.
- Verdict: Detection only.

### 3.2 PC/FCI/Spirtes et al. (2000)
- Problem: Infer causal structure from observational data.
- Active interventions: No.
- Identifiability gate: Markov equivalence / PAG (partial).
- Model repair: No.
- Verdict: Structure learning, not predictive model repair.

### 3.3 Active Causal Learning (Hauser & Buhlmann 2012; Ghassami et al. 2017)
- Problem: Adaptively select interventions to identify a causal DAG.
- Interventions: Yes, actively selected.
- Identifiability gate: Yes.
- Abstention: Implicit (keeps searching).
- Model repair: No -- goal is DAG recovery.
- Blind repair validation: No.
- Verdict: Closest in intervention + identifiability subloop. Lacks repair and blind validation.

### 3.4 Causal Model Revision (Dash 2003; Zuk et al. 2006)
- Problem: Revise a known causal model given new evidence.
- Interventions: Considered but typically passive.
- Model repair: Structural revision, not predictive repair.
- Blind validation: Not primary objective.
- Verdict: Revision framework without the predictive repair + validation loop.

### 3.5 Model Explanation Methods (LIME, SHAP -- Ribeiro et al. 2016)
- Problem: Explain model predictions.
- Interventions: None.
- Mechanism hypotheses: None.
- Model repair: None.
- Verdict: Explanation, not discovery or repair.

### 3.6 Concept Drift Detection (Gama et al. 2014; Webb et al. 2016)
- Problem: Detect distributional shift and retrain.
- Interventions: None.
- Causal mechanism: None.
- Model repair: Retraining (not mechanism-specific).
- Verdict: Statistical shift detection, not structural falsification.

### 3.7 Digital Twin + Model Calibration (Tao et al. 2019; Grieves 2017)
- Problem: Maintain synchronized digital model of a physical system.
- Interventions: Sometimes (physics simulations).
- Structural hypotheses: Physics model, not competing mechanism hypotheses.
- Identifiability gate: None formally.
- Model repair: Calibration/parameter tuning.
- Blind predictive validation: Sometimes.
- Verdict: Closest in spirit to the twin component. Lacks the structured scientific loop.

### 3.8 Invariant Risk Minimization (Arjovsky et al. 2019)
- Problem: Learn predictors invariant across environments.
- Interventions: Multiple environments as implicit interventions.
- Identifiability gate: None.
- Mechanism-specific repair: No -- relearns predictor.
- Verdict: Causal robustness but not FALSIFIER-X loop.

### 3.9 Counterfactual Data Augmentation (Goyal et al. 2019; Sauer & Geiger 2021)
- Problem: Improve model on counterfactual cases.
- Requires: Pre-specified causal model.
- Identifiability gate: None.
- Abstention: None.
- Verdict: Requires known causal model; does not discover or falsify mechanisms.

---

## 4. Capability Matrix

Method                          | MechHyp | ActiveInterv | Falsif | IdentGate | Abstain | MechRepair | BlindVal
ABCD (2014)                     | No  | No  | No  | No       | No      | No         | No
FCI/Spirtes (2000)              | Yes | No  | Yes | Yes(PAG) | partial | No         | No
Active Causal (2012-17)         | Yes | Yes | Yes | Yes      | partial | No         | No
Causal Revision (2003-06)       | Yes | prt | Yes | No       | No      | No(struct) | No
LIME/SHAP (2016-)               | No  | No  | No  | No       | No      | No         | No
Concept Drift (2014-)           | No  | No  | No  | No       | No      | retrain    | sometimes
Digital Twin Cal.               | prt | prt | No  | No       | No      | calibrate  | sometimes
IRM (2019)                      | prt | No  | No  | No       | No      | relearn    | No
Counterfactual Aug.             | req | frm | No  | No       | No      | No         | sometimes
FALSIFIER-X AIR                 | Yes | Yes | Yes | Yes      | Yes     | Yes        | Yes

---

## 5. Novelty Verdict

VERDICT: B. PARTIAL NOVELTY

Major individual components exist in prior work:
- Active intervention for causal discovery: prior work exists.
- Identifiability assessment: prior work (structure learning).
- Predictive model repair: prior work (retraining, augmentation).

The novel contribution is the INTEGRATION:
(a) persistence-driven structural hypothesis generation,
(b) active intervention with sequential falsification,
(c) identifiability-conditioned abstention before repair,
(d) mechanism-specific (not generic) predictive model repair, and
(e) blind unseen-case predictive validation,
as a unified operational loop, evaluated with rigorous learner/evaluator separation.

This integration does not appear in substantially equivalent form in the reviewed prior literature.

Active causal learning (Hauser & Buhlmann 2012) is the closest in the
intervention + identifiability subloop, but targets DAG recovery, not predictive repair.

---

## 6. External Baselines Implemented

### B1: Residual-Shift Baseline
- Shifts prediction 50% toward observed mean (conservative shrinkage).
- Uses: Current observation only. No interventions. No mechanism identification.
- Result: Outperformed by CORRECT_REPAIR on all three families (p < 1e-300 for AC and AR).
- Interpretation: The improvement from FALSIFIER-X is not merely from observing that
  predictions are wrong and correcting conservatively.

### B2: Random-Mechanism Baseline
- Applies repair using a randomly selected mechanism without identification.
- Uses: Same observable features as FALSIFIER-X learner.
- Interventions: None (random selection).
- Result: Outperformed by CORRECT_REPAIR. For AC, random baseline is dramatically worse
  because wrong mechanism repair adds large spurious corrections.
- Interpretation: Mechanism identification, not mere repair execution, drives the benefit.

### Key finding:
CORRECT_REPAIR is superior to both non-causal baselines on all tested families,
establishing that structural identification drives the predictive improvement.

---

## 7. Remaining Limitations

1. Constrained hypothesis class: framework tests mechanisms it knows about; cannot discover
   truly novel mechanisms.
2. Controlled causal worlds: all causal validation uses the synthetic twin; results do not
   transfer to uncontrolled observational data.
3. Small discovery samples for AR/RD: 11-25 samples for coefficient estimation.
4. Linear identifiability assumption in the gate.
5. Held-out topology still uses same three mechanisms (structural variant, not mechanism variant).

---

## 8. Supported Claims

1. FALSIFIER-X recovers identifiable structural mechanisms from intervention evidence. (DEMONSTRATED)
2. Correct mechanism repair outperforms wrong-mechanism and generic repair. (ALL CONFIGS)
3. FALSIFIER-X outperforms non-intervention baselines when mechanisms are identifiable. (DEMONSTRATED)
4. Benefit generalizes to held-out causal worlds with different topology and coefficients. (DEMONSTRATED)
5. Nonlinear threshold mechanism can be identified and repaired via mechanism-specific NL regression. (DEMONSTRATED)
6. The complete closed loop is a novel integration not present in equivalent form in reviewed prior work. (PARTIAL - novel integration, not novel components)

## 9. Unsupported Claims

1. FALSIFIER-X performs general causal discovery. (Constrained hypothesis class only.)
2. Results transfer to uncontrolled observational BTS/NOAA data. (No causal ground truth.)
3. FALSIFIER-X is first to use active interventions for identification. (Prior work exists.)
4. The coefficient estimator is unbiased. (Empirically close; not formally proven.)
5. Abstention guarantees correct recovery when it fires. (Reduces but does not eliminate risk.)
