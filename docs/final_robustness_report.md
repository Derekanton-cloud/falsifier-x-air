# FALSIFIER-X AIR: Final Robustness and External Validation

## 1. Discovery Counts (Base Configuration)
- AIRCRAFT_ROTATION: 100 seeds | 25 identified | 25 correct | 0 wrong | 75 inconclusive | 25 for estimation
- RESOURCE_DEPENDENCY: 100 seeds | 11 identified | 11 correct | 0 wrong | 89 inconclusive | 11 for estimation
- AIRPORT_CAPACITY: 100 seeds | 100 identified | 100 correct | 0 wrong | 0 inconclusive | 100 for estimation

## 2. Parameter Estimation
- AIRCRAFT_ROTATION: beta_hat=0.8441, OLS SE=0.0053, n=25
- RESOURCE_DEPENDENCY: beta_hat=0.6048, OLS SE=0.0135, n=11
- AIRPORT_CAPACITY: beta_hat=20.0254, OLS SE=0.0411, n=100
- GENERIC: beta_hat=10.0780, OLS SE=0.1915, n=136
- NONLINEAR: beta_hat=0.0000, OLS SE=0.0000, n=0

## 3. Base Blind Evaluation (Development Config)
### AIRCRAFT_ROTATION
  ID: 86/300 correct (28.7%) | 0 wrong (0.0%) | 214 inconclusive (71.3%)
  E2E: orig=7.1887 -> e2e=5.4532 | D=1.7355 [1.4230,2.0480] p=1.3678446694333023e-27 | repaired=86 abstained=214
  Cond (N=86): orig=7.434 correct=1.380 wrong=11.479 generic=6.990 oracle=1.376 oracle_gap=0.003
  D correct-wrong: -10.0990 [-10.3263,-9.8717] p=< 1e-300
  D correct-generic: -5.6101 [-5.7584,-5.4617] p=< 1e-300
  D correct-residual: -5.8080 [-5.9476,-5.6683] p=< 1e-300
  D correct-random: -6.7548 [-7.7926,-5.7170] p=2.8251927098267706e-37
### RESOURCE_DEPENDENCY
  ID: 38/300 correct (12.7%) | 0 wrong (0.0%) | 262 inconclusive (87.3%)
  E2E: orig=3.9199 -> e2e=3.5703 | D=0.3496 [0.2445,0.4547] p=7.046519265803295e-11 | repaired=38 abstained=262
  Cond (N=38): orig=4.223 correct=1.462 wrong=8.011 generic=5.272 oracle=1.459 oracle_gap=0.003
  D correct-wrong: -6.5489 [-6.7334,-6.3643] p=< 1e-300
  D correct-generic: -3.8096 [-3.9988,-3.6203] p=< 1e-300
  D correct-residual: -2.3101 [-2.4152,-2.2050] p=< 1e-300
  D correct-random: -2.1060 [-2.8316,-1.3804] p=1.2794181360824013e-08
### AIRPORT_CAPACITY
  ID: 300/300 correct (100.0%) | 0 wrong (0.0%) | 0 inconclusive (0.0%)
  E2E: orig=8.9961 -> e2e=1.1746 | D=7.8216 [7.7612,7.8819] p=< 1e-300 | repaired=300 abstained=0
  Cond (N=300): orig=8.996 correct=1.175 wrong=9.364 generic=2.976 oracle=1.174 oracle_gap=0.000
  D correct-wrong: -8.1895 [-8.2563,-8.1228] p=< 1e-300
  D correct-generic: -1.8011 [-1.8413,-1.7608] p=< 1e-300
  D correct-residual: -3.3241 [-3.3643,-3.2839] p=< 1e-300
  D correct-random: -118.3961 [-128.0451,-108.7470] p=8.414198270535249e-128

## 4. Held-Out Causal World Evaluation (W1)
### AIRCRAFT_ROTATION
  ID: 100/100 correct (100.0%) | 0 wrong (0.0%) | 0 inconclusive (0.0%)
  E2E: orig=15.8381 -> e2e=2.5554 | D=13.2828 [12.8797,13.6858] p=< 1e-300 | repaired=100 abstained=0
  Cond (N=100): orig=15.838 correct=2.555 wrong=14.095 generic=11.014 oracle=1.580 oracle_gap=0.975
  D correct-wrong: -11.5395 [-11.8221,-11.2570] p=< 1e-300
  D correct-generic: -8.4586 [-8.7999,-8.1174] p=< 1e-300
  D correct-residual: -12.4913 [-12.8942,-12.0884] p=< 1e-300
  D correct-random: -9.4891 [-10.8263,-8.1519] p=5.611028218323974e-44
### RESOURCE_DEPENDENCY
  ID: 100/100 correct (100.0%) | 0 wrong (0.0%) | 0 inconclusive (0.0%)
  E2E: orig=9.6190 -> e2e=1.8501 | D=7.7689 [7.4613,8.0765] p=< 1e-300 | repaired=100 abstained=0
  Cond (N=100): orig=9.619 correct=1.850 wrong=11.079 generic=6.629 oracle=1.580 oracle_gap=0.270
  D correct-wrong: -9.2287 [-9.4368,-9.0207] p=< 1e-300
  D correct-generic: -4.7785 [-4.9164,-4.6407] p=< 1e-300
  D correct-residual: -5.7457 [-5.9945,-5.4970] p=< 1e-300
  D correct-random: -4.6525 [-5.3615,-3.9436] p=7.286383363815653e-38
### AIRPORT_CAPACITY
  ID: 100/100 correct (100.0%) | 0 wrong (0.0%) | 0 inconclusive (0.0%)
  E2E: orig=12.1702 -> e2e=2.0109 | D=10.1593 [10.0012,10.3175] p=< 1e-300 | repaired=100 abstained=0
  Cond (N=100): orig=12.170 correct=2.011 wrong=12.675 generic=5.991 oracle=1.580 oracle_gap=0.431
  D correct-wrong: -10.6637 [-10.8114,-10.5159] p=< 1e-300
  D correct-generic: -3.9801 [-4.1043,-3.8559] p=< 1e-300
  D correct-residual: -4.0753 [-4.1557,-3.9948] p=< 1e-300
  D correct-random: -235.1386 [-267.9193,-202.3580] p=6.760134076606174e-45

## 5. Coverage-Reliability Tradeoff (W3)
### K=1
  AIRCRAFT_ROTATION: coverage=0.32 acc_repaired=1.00 false_rec=0.00 e2e_improvement=1.9165
  RESOURCE_DEPENDENCY: coverage=0.00 acc_repaired=0.00 false_rec=0.00 e2e_improvement=0.0000
  AIRPORT_CAPACITY: coverage=0.00 acc_repaired=0.00 false_rec=0.00 e2e_improvement=0.0000
### K=2
  AIRCRAFT_ROTATION: coverage=0.32 acc_repaired=1.00 false_rec=0.00 e2e_improvement=1.9165
  RESOURCE_DEPENDENCY: coverage=0.00 acc_repaired=0.00 false_rec=0.00 e2e_improvement=0.0000
  AIRPORT_CAPACITY: coverage=1.00 acc_repaired=1.00 false_rec=0.00 e2e_improvement=7.8017
### K=3
  AIRCRAFT_ROTATION: coverage=0.32 acc_repaired=1.00 false_rec=0.00 e2e_improvement=1.9165
  RESOURCE_DEPENDENCY: coverage=0.17 acc_repaired=1.00 false_rec=0.00 e2e_improvement=0.4820
  AIRPORT_CAPACITY: coverage=1.00 acc_repaired=1.00 false_rec=0.00 e2e_improvement=7.8017

## 6. Nonlinear Repair Experiment (W4)
  Pre-declared threshold tau=5.0, true NL coefficient=3.0
  Discovery records: 50
  NL beta_hat=2.9953 (OLS SE=0.0032) | Linear beta_hat=2.4281
  MAE: orig=25.8207 linear=3.6131 NL=1.2045 generic=28.1140 oracle=1.2035
  D NL-linear:  -2.4086 [-2.4819,-2.3354] p=< 1e-300
  D NL-generic: -26.9096 [-27.2323,-26.5868] p=< 1e-300
  D NL-oracle:  0.0009 (oracle gap = cost of estimating NL coefficient)

## 7. Ablation Study (W3/W5)
### AIRCRAFT_ROTATION
  PASSIVE: mean_MAE=7.1602 coverage=0.00
  RANDOM: mean_MAE=5.2745 coverage=0.32
  ACTIVE: mean_MAE=5.2745 coverage=0.32
### RESOURCE_DEPENDENCY
  PASSIVE: mean_MAE=3.9761 coverage=0.00
  RANDOM: mean_MAE=3.4984 coverage=0.17
  ACTIVE: mean_MAE=3.4984 coverage=0.17
### AIRPORT_CAPACITY
  PASSIVE: mean_MAE=9.0521 coverage=0.00
  RANDOM: mean_MAE=1.1874 coverage=1.00
  ACTIVE: mean_MAE=1.1874 coverage=1.00

## 8. Weakness Assessment
W1 Synthetic dependence: Tested via held-out topology (12-flight, 4-leg chains).
W2 Narrow mechanisms: Three mechanisms evaluated across base + held-out variants.
W3 Low AR/RD coverage: Coverage-reliability tradeoff with K=1,2,3 reported.
W4 Linear repair: Threshold nonlinear mechanism with tau=5.0 tested.
W5 External baselines: Residual-shift and random-mechanism baselines compared.
W6 Novelty: See docs/novelty_and_baseline_analysis.md.
