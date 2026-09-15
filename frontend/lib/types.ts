export type MechanismId = "AIRCRAFT_ROTATION" | "AIRPORT_CAPACITY" | "RESOURCE_DEPENDENCY";
export type InvestigationKind = "identifiable" | "indistinguishable";

export interface MechanismMetric { beta_hat: number; ols_se: number; n: number }
export interface BlindMetric {
  original: number; repaired: number; improvement: number; ci: [number, number]; pValue: string;
  repairedCases: number; abstainedCases: number; total: number;
}
export interface ResearchData {
  source: "validated-artifact" | "presentation-fallback";
  generatedFrom: string;
  estimation: Record<MechanismId, MechanismMetric>;
  blind: Record<MechanismId, BlindMetric>;
  coverage: Record<MechanismId, number>;
  ablation: Record<MechanismId, { passive: number; random: number; active: number; coverage: number }>;
  nonlinear: { tau: number; trueCoefficient: number; linearEstimate: number; nonlinearEstimate: number; original: number; linear: number; nonlinear: number; generic: number; oracle: number };
  heldOut: Record<MechanismId, { original: number; repaired: number; improvement: number }>;
}

export interface LiveGraphNode {
  id: string;
  type: "flight" | "airport" | "resource" | string;
  aircraft?: string;
  delay?: number;
  prediction?: number;
  residual?: number;
  time?: number;
}
export interface LiveGraphEdge { source: string; target: string; relation: string }
export interface LiveExperiment { id: string; intervention: string; description: string; effect: number; baseline: number; counterfactual: number }
export interface LiveCandidate { id: MechanismId; status: string; observations: number; log_evidence: number }
export interface LiveTrace {
  source: "public-digital-twin";
  case: InvestigationKind;
  scenario: string;
  adequacy: { state: string; reason: string; evidence: number; standardized_residual: number; residual_z_threshold: number; interval_miscoverage: number; persistence: number; graph_concentration: number; ood_score: number; channels: Record<string, boolean> };
  graph: { nodes: LiveGraphNode[]; edges: LiveGraphEdge[] };
  affected_flights: string[];
  candidates: LiveCandidate[];
  experiments: LiveExperiment[];
  recovered_mechanism: MechanismId | null;
  outcome: string;
  identifiability: { epsilon: number; reason: string | null; distances: { survivor: string; competitor: string; distance: number }[] };
}
