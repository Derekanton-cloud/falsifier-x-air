import type { ResearchData } from "./types";

/** Offline fallback. Values are transcribed only from data/processed/final_robustness_results.json. */
export const presentationData: ResearchData = {
  source: "presentation-fallback",
  generatedFrom: "data/processed/final_robustness_results.json",
  estimation: {
    AIRCRAFT_ROTATION: { beta_hat: 0.844129346981847, ols_se: 0.005253065204691838, n: 25 },
    RESOURCE_DEPENDENCY: { beta_hat: 0.6047813912339881, ols_se: 0.01349962805355375, n: 11 },
    AIRPORT_CAPACITY: { beta_hat: 20.025417283706588, ols_se: 0.041086438890480144, n: 100 },
  },
  blind: {
    AIRCRAFT_ROTATION: { original: 7.1887, repaired: 5.4532, improvement: 1.7355, ci: [1.423, 2.048], pValue: "1.37 × 10⁻²⁷", repairedCases: 86, abstainedCases: 214, total: 300 },
    RESOURCE_DEPENDENCY: { original: 3.9199, repaired: 3.5703, improvement: 0.3496, ci: [0.2445, 0.4547], pValue: "7.05 × 10⁻¹¹", repairedCases: 38, abstainedCases: 262, total: 300 },
    AIRPORT_CAPACITY: { original: 8.9961, repaired: 1.1746, improvement: 7.8216, ci: [7.7612, 7.8819], pValue: "< 1 × 10⁻³⁰⁰", repairedCases: 300, abstainedCases: 0, total: 300 },
  },
  coverage: { AIRCRAFT_ROTATION: 0.32, RESOURCE_DEPENDENCY: 0.17, AIRPORT_CAPACITY: 1 },
  ablation: {
    AIRCRAFT_ROTATION: { passive: 7.1602, random: 5.2745, active: 5.2745, coverage: 0.32 },
    RESOURCE_DEPENDENCY: { passive: 3.9761, random: 3.4984, active: 3.4984, coverage: 0.17 },
    AIRPORT_CAPACITY: { passive: 9.0521, random: 1.1874, active: 1.1874, coverage: 1 },
  },
  nonlinear: { tau: 5, trueCoefficient: 3, linearEstimate: 2.4281, nonlinearEstimate: 2.9953, original: 25.8207, linear: 3.6131, nonlinear: 1.2045, generic: 28.114, oracle: 1.2035 },
  heldOut: {
    AIRCRAFT_ROTATION: { original: 15.8381, repaired: 2.5554, improvement: 13.2828 },
    RESOURCE_DEPENDENCY: { original: 9.619, repaired: 1.8501, improvement: 7.7689 },
    AIRPORT_CAPACITY: { original: 12.1702, repaired: 2.0109, improvement: 10.1593 },
  },
};
