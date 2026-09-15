import { readFile } from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";
import type { MechanismId, ResearchData } from "../../../lib/types";

export const dynamic = "force-dynamic";

const mechanisms: MechanismId[] = ["AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY"];

/** Read-only adapter: the Python-generated artifact remains the canonical source. */
export async function GET() {
  try {
    const artifactPath = path.resolve(process.cwd(), "..", "data", "processed", "final_robustness_results.json");
    const raw = JSON.parse(await readFile(artifactPath, "utf-8")) as Record<string, any>;
    const blind = Object.fromEntries(mechanisms.map((id) => {
      const item = raw.base_blind[id]; const e2e = item.e2e; const imp = e2e.improvement;
      return [id, { original: e2e.orig_mae, repaired: e2e.e2e_mae, improvement: imp.mean,
        ci: [imp.ci95_low, imp.ci95_high], pValue: String(imp.p_two_sided), repairedCases: e2e.n_repaired,
        abstainedCases: e2e.n_abstained, total: item.identification.n_total }];
    })) as ResearchData["blind"];
    const data: ResearchData = {
      source: "validated-artifact", generatedFrom: "data/processed/final_robustness_results.json",
      estimation: Object.fromEntries(mechanisms.map((id) => [id, { beta_hat: raw.estimation[id].beta_hat, ols_se: raw.estimation[id].ols_se, n: raw.estimation[id].n }])) as ResearchData["estimation"],
      blind,
      coverage: Object.fromEntries(mechanisms.map((id) => [id, raw.coverage_reliability["K=3"][id].coverage])) as ResearchData["coverage"],
      ablation: Object.fromEntries(mechanisms.map((id) => [id, { passive: raw.ablations[id].PASSIVE.mean_mae, random: raw.ablations[id].RANDOM.mean_mae, active: raw.ablations[id].ACTIVE.mean_mae, coverage: raw.ablations[id].ACTIVE.coverage }])) as ResearchData["ablation"],
      nonlinear: { tau: raw.nonlinear.tau, trueCoefficient: raw.nonlinear.true_nl_coeff, linearEstimate: raw.nonlinear.linear_beta_hat, nonlinearEstimate: raw.nonlinear.nl_beta_hat, original: raw.nonlinear.orig_mae, linear: raw.nonlinear.linear_mae, nonlinear: raw.nonlinear.nl_mae, generic: raw.nonlinear.generic_mae, oracle: raw.nonlinear.oracle_mae },
      heldOut: Object.fromEntries(mechanisms.map((id) => { const e = raw.heldout_blind[id].e2e; return [id, { original: e.orig_mae, repaired: e.e2e_mae, improvement: e.improvement.mean }]; })) as ResearchData["heldOut"],
    };
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: "Validated research artifact unavailable", detail: String(error) }, { status: 503 });
  }
}
