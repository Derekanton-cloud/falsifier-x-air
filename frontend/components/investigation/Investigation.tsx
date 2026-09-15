"use client";

import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ArrowDown, Check, ChevronRight, Circle, CircleStop, LoaderCircle, Play, RotateCcw, ScanSearch, ShieldAlert } from "lucide-react";
import dynamic from "next/dynamic";
import { useState } from "react";
import type { InvestigationKind, LiveCandidate, LiveExperiment, LiveTrace, ResearchData } from "../../lib/types";

const AviationGraph = dynamic(() => import("../graph/AviationGraph"), { ssr: false, loading: () => <div className="graph-loading">INITIALIZING OBSERVABLE GRAPH...</div> });
const steps = ["MODEL FAILURE", "INADEQUACY", "LOCALIZATION", "HYPOTHESES", "INTERVENTION", "COUNTERFACTUAL", "FALSIFICATION", "IDENTIFIABILITY", "RECOVERY", "REPAIR", "BLIND VALIDATION", "RESULT"];
const mechanismDescription: Record<string, string> = {
  AIRCRAFT_ROTATION: "Scheduled aircraft-leg propagation",
  AIRPORT_CAPACITY: "Airport-associated congestion",
  RESOURCE_DEPENDENCY: "Shared-resource disruption chain",
};
const mechanismSignature: Record<string, string> = {
  AIRCRAFT_ROTATION: "Downstream residual lift on represented rotation edges",
  AIRPORT_CAPACITY: "Broad airport-associated capacity response",
  RESOURCE_DEPENDENCY: "Temporal lift on a represented shared-resource sequence",
};
const actionToMechanism: Record<string, string> = {
  disable_aircraft_rotation: "AIRCRAFT_ROTATION",
  increase_capacity: "AIRPORT_CAPACITY",
  relieve_resource_dependency: "RESOURCE_DEPENDENCY",
};
type AssessmentState = "pending" | "running" | "resolving" | "resolved" | "unavailable";

function Stage({ index, label, current }: { index: number; label: string; current: number }) {
  return <div className={index < current ? "stage done" : index === current ? "stage active" : "stage"}><span>{String(index + 1).padStart(2, "0")}</span><b>{label}</b></div>;
}

export function Investigation({ data, presentation }: { data: ResearchData; presentation: boolean }) {
  const [step, setStep] = useState(0);
  const [kind, setKind] = useState<InvestigationKind>("identifiable");
  const [assessment, setAssessment] = useState<AssessmentState>("pending");
  const [resolvedChannels, setResolvedChannels] = useState(0);
  const [trace, setTrace] = useState<LiveTrace | null>(null);
  const identified = trace?.outcome === "RECOVERED";

  const selectCase = (next: InvestigationKind) => {
    setKind(next); setStep(0); setTrace(null); setAssessment("pending"); setResolvedChannels(0);
  };
  const assess = async () => {
    if (assessment !== "pending") return;
    setAssessment("running");
    try {
      const response = await fetch("/api/investigation", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ case: kind }) });
      if (!response.ok) throw new Error("live bridge unavailable");
      const result = await response.json() as LiveTrace;
      setTrace(result); setAssessment("resolving");
      for (let index = 1; index <= 6; index += 1) window.setTimeout(() => setResolvedChannels(index), index * 320);
      window.setTimeout(() => setAssessment("resolved"), 6 * 320 + 180);
    } catch {
      setAssessment("unavailable");
    }
  };
  const advance = () => {
    if (step <= 1) { if (assessment === "pending") void assess(); else if (assessment === "resolved") setStep(2); return; }
    if (step === 7 && !identified) { setStep(11); return; }
    setStep((current) => Math.min(current + 1, 11));
  };
  return <div className="investigation" id="live-investigation">
    <div className="commandbar"><div><span className="eyebrow">CONTROLLED TWIN / {assessment === "resolved" ? "PUBLIC PYTHON BRIDGE CONFIRMED" : assessment === "unavailable" ? "BRIDGE UNAVAILABLE" : "AWAITING INADEQUACY ASSESSMENT"}</span><h1>Live investigation</h1></div><div className="controls"><button onClick={() => selectCase("identifiable")}><Play size={15}/>LOAD INVESTIGATION</button><button className={kind === "identifiable" ? "selected" : ""} onClick={() => selectCase("identifiable")}>IDENTIFIABLE CASE</button><button className={kind === "indistinguishable" ? "selected" : ""} onClick={() => selectCase("indistinguishable")}>INDISTINGUISHABLE CASE</button><button onClick={() => selectCase(kind)}><RotateCcw size={14}/>RESET</button></div></div>
    {!presentation && <div className="pipeline">{steps.map((label, index) => <Stage key={label} index={index} label={label} current={step} />)}</div>}
    <AnimatePresence mode="wait"><motion.section key={`${kind}-${step}-${assessment}`} initial={{ opacity: 0, y: 12, filter: "blur(4px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} exit={{ opacity: 0, y: -8 }} transition={{ duration: .32 }} className="investigation-stage">
      {step <= 1 && <Failure assessment={assessment} resolvedChannels={resolvedChannels} trace={trace} onAction={advance} />}
      {step >= 2 && step <= 3 && trace && <Localization trace={trace} focus={step === 3} onAdvance={advance} />}
      {step >= 4 && step <= 6 && trace && <Falsification trace={trace} step={step} onAdvance={advance} />}
      {step === 7 && trace && <Identifiability trace={trace} onAdvance={advance} />}
      {step >= 8 && step <= 10 && trace && identified && <Repair data={data} step={step} mechanism={trace.recovered_mechanism!} onAdvance={advance} />}
      {step === 11 && <Result data={data} trace={trace} onReset={() => selectCase(kind)} />}
    </motion.section></AnimatePresence>
  </div>;
}

const channels = [
  ["Residual magnitude", "residual_magnitude"], ["Temporal persistence", "persistence"], ["Uncertainty violation", "prediction_uncertainty_violation"],
  ["Graph concentration", "graph_concentration"], ["OOD gate", "ood"], ["Structural suspicion", "structural"],
] as const;
function Failure({ assessment, resolvedChannels, trace, onAction }: { assessment: AssessmentState; resolvedChannels: number; trace: LiveTrace | null; onAction: () => void }) {
  const complete = assessment === "resolved" && trace;
  const valueFor = (key: string) => key === "structural" ? trace?.adequacy.state === "STRUCTURALLY_SUSPICIOUS" : key === "residual_magnitude" ? (trace ? trace.adequacy.standardized_residual >= trace.adequacy.residual_z_threshold : false) : Boolean(trace?.adequacy.channels[key]);
  return <div className="failure-grid"><div className="instrument panel"><div className="panel-head"><span className="eyebrow">STEP {complete ? "02 / INADEQUACY" : "01 / OBSERVATION"}</span><AlertTriangle size={18}/></div><h2>{complete ? "Persistent model inadequacy" : "Observe a prediction failure."}</h2><p className="lede">A held observation violates the model&apos;s frozen uncertainty envelope. This is diagnostic evidence—not a causal conclusion.</p><div className="residual-display"><div><label>MODEL PREDICTION</label><strong>ŷ<sub>t</sub></strong><small>baseline prediction</small></div><ArrowDown/><div className="observed"><label>OBSERVED DELAY</label><strong>y<sub>t</sub></strong><small>held observation</small></div><div className="residual"><label>RESIDUAL</label><strong>ε<sub>t</sub></strong><small>outside prediction interval</small></div></div>{complete && <div className="assessment-summary"><span>ASSESSMENT RESULT</span><b>{trace.adequacy.state.replaceAll("_", " ")}</b><small>{trace.adequacy.reason}</small></div>}<button className="primary" disabled={assessment === "running" || assessment === "resolving" || assessment === "unavailable"} onClick={onAction}>{assessment === "pending" ? "ASSESS INADEQUACY" : assessment === "running" || assessment === "resolving" ? "ASSESSING DIAGNOSTICS..." : assessment === "resolved" ? "LOCALIZE OBSERVABLE STRUCTURE" : "LIVE BRIDGE UNAVAILABLE"}<ChevronRight size={16}/></button></div><div className="diagnostic panel"><span className="eyebrow">FROZEN DIAGNOSTIC CHANNELS</span>{channels.map(([label, key], index) => { const pending = index >= resolvedChannels || !trace; const positive = valueFor(key); return <div className="check-row" key={label}><span>{label}</span>{pending ? <b className="pending"><Circle size={13}/>PENDING</b> : <b className={positive ? "good" : "neutral"}>{positive ? <Check size={14}/> : <Circle size={13}/>} {positive ? "EVIDENCE" : "CLEAR"}</b>}</div>; })}<div className={complete ? "initiation" : "initiation awaiting"}>{assessment === "running" || assessment === "resolving" ? <LoaderCircle className="spin" size={18}/> : <ScanSearch size={18}/>}<span>{complete ? "STRUCTURAL INVESTIGATION INITIATED" : assessment === "unavailable" ? "LOCAL PYTHON BRIDGE REQUIRED" : "AWAITING DIAGNOSTIC ASSESSMENT"}</span></div></div></div>;
}

function Localization({ trace, focus, onAdvance }: { trace: LiveTrace; focus: boolean; onAdvance: () => void }) {
  const relations = trace.graph.edges.filter((edge) => edge.relation === "AIRCRAFT_ROTATION" || edge.relation === "RESOURCE_DEPENDENCY");
  return <div className="localization" id="causal-graph"><div className="section-title"><span className="eyebrow">STEP 03 / OBSERVABLE STRUCTURE</span><h2>Localize the residual in its operational topology.</h2><p>Rendered directly from the public graph constructed for <code>{trace.scenario}</code>. Edges encode observable operational relations, not hidden causal truth.</p></div><div className="graph-panel panel"><AviationGraph nodes={trace.graph.nodes} edges={trace.graph.edges} affected={trace.affected_flights}/><div className="graph-side"><span className="eyebrow">LOCALIZATION RESULT</span><strong>{trace.affected_flights.join(" · ")}</strong><p>High-residual flights plus one-hop observable flight neighbors.</p><dl><div><dt>FLIGHT NODES</dt><dd>{trace.graph.nodes.filter((node) => node.type === "flight").length}</dd></div><div><dt>STRUCTURAL RELATIONS</dt><dd>{relations.length}</dd></div><div><dt>TEMPORAL CONSTRAINT</dt><dd>source_time &lt; target_time</dd></div></dl><button className="primary" onClick={onAdvance}>{focus ? "OPEN HYPOTHESIS SPACE" : "FOCUS AFFECTED SUBGRAPH"}<ChevronRight size={16}/></button></div></div></div>;
}

function candidateStatus(candidate: LiveCandidate, visibleExperiments: number, trace: LiveTrace): string {
  const action = Object.entries(actionToMechanism).find(([, id]) => id === candidate.id)?.[0];
  const index = trace.experiments.filter((experiment) => !experiment.id.includes("-ident-")).findIndex((experiment) => experiment.intervention === action);
  if (visibleExperiments === 0) return "CANDIDATE";
  return index >= 0 && index < visibleExperiments ? "UNDER REVIEW" : "UNTESTED";
}
function Falsification({ trace, step, onAdvance }: { trace: LiveTrace; step: number; onAdvance: () => void }) {
  const mainExperiments = trace.experiments.filter((experiment) => !experiment.id.includes("-ident-"));
  const visibleExperiments = Math.min(Math.max(step - 4, 0), mainExperiments.length);
  const total = mainExperiments.length;
  return <div id="falsification-lab"><div className="section-title"><span className="eyebrow">STEP {String(step + 1).padStart(2, "0")} / PUBLIC COUNTERFACTUAL TRACE</span><h2>Falsification lab</h2><p>Only mechanisms justified by the observable local graph are displayed. Effects are paired total-delay reductions returned by the public twin API.</p></div><div className="budget"><span>MAIN TRACE EVENTS REVEALED</span><b>{visibleExperiments} / {total}</b><div><i style={{ width: `${total ? visibleExperiments / total * 100 : 0}%` }}/></div></div><div className="hypotheses">{trace.candidates.map((candidate, index) => <article className={`hypothesis ${candidateStatus(candidate, visibleExperiments, trace).toLowerCase().replaceAll(" ", "-")}`} key={candidate.id}><div className="hypothesis-top"><span>H{index + 1}</span><b>{candidate.id}</b><em>{candidateStatus(candidate, visibleExperiments, trace)}</em></div><p>{mechanismDescription[candidate.id]}</p><dl><div><dt>OBSERVABLE SIGNATURE</dt><dd>{mechanismSignature[candidate.id]}</dd></div><div><dt>LEARNER EVIDENCE</dt><dd>{visibleExperiments ? `log evidence ${candidate.log_evidence.toFixed(2)}` : "No intervention result revealed"}</dd></div><div><dt>NEXT DECISION</dt><dd>{visibleExperiments === total ? "Send surviving candidates to identifiability gate" : "Intervention trace pending"}</dd></div></dl></article>)}</div>{visibleExperiments > 0 && <div className="experiment-table"><div className="experiment-header"><span>HYPOTHESIS</span><span>INTERVENTION</span><span>BASELINE TOTAL</span><span>COUNTERFACTUAL TOTAL</span><span>OBSERVED EFFECT</span></div>{mainExperiments.slice(0, visibleExperiments).map((experiment) => <ExperimentRow key={experiment.id} experiment={experiment}/>)}</div>}<button className="primary next" onClick={onAdvance}>{visibleExperiments < total ? "REVEAL NEXT COUNTERFACTUAL RESULT" : "OPEN IDENTIFIABILITY GATE"}<ChevronRight size={16}/></button></div>;
}
function ExperimentRow({ experiment }: { experiment: LiveExperiment }) { return <div className="experiment-row"><span>{actionToMechanism[experiment.intervention]}</span><span>{experiment.description}</span><span>{experiment.baseline.toFixed(2)}</span><span>{experiment.counterfactual.toFixed(2)}</span><b>{experiment.effect.toFixed(2)} min</b></div>; }

function Identifiability({ trace, onAdvance }: { trace: LiveTrace; onAdvance: () => void }) {
  const identifiable = trace.outcome === "RECOVERED";
  const distancePairs = trace.identifiability.distances;
  const probes = trace.experiments.filter((item) => item.id.includes("-ident-"));
  return <div className={identifiable ? "gate identified" : "gate confounded"} id="identifiability"><div className="section-title"><span className="eyebrow">STEP 08 / PRE-RECOVERY GATE</span><h2>Identifiability is a decision boundary.</h2><p>One candidate surviving a score update is not enough. The system compares observable intervention signatures before authorizing any repair.</p></div>{!identifiable && <div className="equivalence"><div><span>H1 / AIRCRAFT_ROTATION</span><strong>{trace.experiments.find((item) => item.intervention === "disable_aircraft_rotation")?.effect.toFixed(2)} min</strong></div><b>EXPERIMENTALLY INDISTINGUISHABLE</b><div><span>H3 / RESOURCE_DEPENDENCY</span><strong>{trace.experiments.find((item) => item.intervention === "relieve_resource_dependency")?.effect.toFixed(2)} min</strong></div></div>}<div className="gate-flow"><div><label>COMPARED RESPONSES</label><strong>{identifiable ? trace.recovered_mechanism : "H1 and H3"}</strong></div><ArrowDown/><div className="formula"><label>OBSERVABLE RESPONSE DISTANCE</label><strong>D = |Δ<sub>i</sub> - Δ<sub>j</sub>| / max(Δ<sub>i</sub>, Δ<sub>j</sub>, σ<sub>eff</sub>, 10<sup>-6</sup>)</strong><small>ε<sub>ident</sub> = {trace.identifiability.epsilon.toFixed(2)} (frozen after Phase-9 calibration)</small></div><ArrowDown/><div className="decision"><label>DECISION</label><strong>{identifiable ? "LEGITIMATELY IDENTIFIED" : "INDISTINGUISHABLE EQUIVALENCE CLASS"}</strong><span>{distancePairs.map((pair) => `${pair.survivor} / ${pair.competitor}: D=${pair.distance.toFixed(2)}`).join(" · ")}</span></div></div><div className="probe-results"><span>IDENTIFIABILITY PROBES / PUBLIC COUNTERFACTUALS</span>{probes.map((probe) => <div key={probe.id}><b>{actionToMechanism[probe.intervention]}</b><span>{probe.description}</span><strong>{probe.effect.toFixed(2)} min</strong></div>)}</div><div className="gate-outcome">{identifiable ? <><Check/> REPAIR AUTHORIZED</> : <><CircleStop/> REPAIR BLOCKED · ABSTAIN</>}</div><button className="primary next" onClick={onAdvance}>{identifiable ? "FREEZE DISCOVERY PARAMETER" : "RECORD SCIENTIFIC ABSTENTION"}<ChevronRight size={16}/></button></div>;
}

function Repair({ data, step, mechanism, onAdvance }: { data: ResearchData; step: number; mechanism: keyof ResearchData["estimation"]; onAdvance: () => void }) {
  const estimate = data.estimation[mechanism];
  return <div className="repair-stage" id="model-repair"><div className="section-title"><span className="eyebrow">STEPS 09-11 / DISCOVERY -&gt; BLIND</span><h2>{step === 8 ? "Mechanism parameter recovery" : step === 9 ? "Mechanism-specific repair" : "Blind validation chamber"}</h2></div>{step === 8 && <div className="estimator panel"><div><label>DISCOVERY-ONLY ESTIMATE</label><strong>β̂ = {estimate.beta_hat.toFixed(4)}</strong><small>OLS SE = {estimate.ols_se.toFixed(4)} · n = {estimate.n}</small></div><div className="confidence"><i/><span>95% estimator interval visualization</span></div><aside><ShieldAlert size={18}/><b>BLIND DATA LOCKED</b><p>Oracle truth unavailable to learner.</p></aside></div>}{step === 9 && <div className="repair-equation panel"><div><span>ORIGINAL MODEL</span><strong>ŷ<sub>original</sub></strong></div><ChevronRight/><div><span>RECOVERED STRUCTURE</span><strong>{mechanism}</strong><small>β̂ = {estimate.beta_hat.toFixed(4)}</small></div><ChevronRight/><div><span>REPAIRED MODEL</span><strong>ŷ<sub>repaired</sub> = ŷ<sub>original</sub> + β̂ × φ<sub>mechanism</sub>(observable graph)</strong><small>Literal implementation: prediction + coefficient × observable feature in <code>repair.py</code>.</small></div></div>}{step === 10 && <Validation data={data} mechanism={mechanism}/>}<button className="primary next" onClick={onAdvance}>{step === 10 ? "REVEAL RESULT" : "CONTINUE"}<ChevronRight size={16}/></button></div>;
}
function Validation({ data, mechanism }: { data: ResearchData; mechanism: keyof ResearchData["blind"] }) { const result = data.blind[mechanism]; return <div className="validation panel" id="blind-validation"><div className="lock-flow"><span>DISCOVERY<br/><b>LOCKED</b></span><ChevronRight/><span>PARAMETERS<br/><b>FROZEN</b></span><ChevronRight/><span>BLIND<br/><b>TEST</b></span></div><div className="validation-numbers"><div><label>ORIGINAL MAE</label><strong>{result.original.toFixed(4)}</strong></div><div><label>REPAIRED MAE</label><strong>{result.repaired.toFixed(4)}</strong></div><div><label>MAE REDUCTION</label><strong>{result.improvement.toFixed(4)}</strong></div><div><label>95% CI</label><strong>[{result.ci[0]}, {result.ci[1]}]</strong></div><div><label>p-VALUE</label><strong>{result.pValue}</strong></div></div><p>Cached validated experiment result · Base blind evaluation · {mechanism} · {result.total} held-out cases. MAE reduction = original MAE − repaired MAE.</p></div>; }

function Result({ data, trace, onReset }: { data: ResearchData; trace: LiveTrace | null; onReset: () => void }) {
  const identified = trace?.outcome === "RECOVERED"; const result = data.blind.AIRCRAFT_ROTATION;
  return <div className="hero-result">{identified ? <><span className="eyebrow">CACHED VALIDATED EXPERIMENT RESULT</span><h2>Blind model repair</h2><div className="hero-metrics"><div><label>ORIGINAL</label><strong>{result.original.toFixed(4)}</strong></div><ArrowDown/><div className="repaired"><label>REPAIRED</label><strong>{result.repaired.toFixed(4)}</strong></div><div><label>MAE REDUCTION</label><strong>{result.improvement.toFixed(4)} <small>MAE</small></strong></div></div><p>Mechanism identified {result.repairedCases}/{result.total} · Abstained {result.abstainedCases}/{result.total} · False recoveries 0</p></> : <><span className="eyebrow">FINAL DECISION / SCIENTIFIC SAFEGUARD</span><h2>ABSTAIN</h2><p>No mechanism-specific repair was authorized. The competing mechanisms produced experimentally indistinguishable public intervention signatures.</p><blockquote>Uncertainty is not failure.<br/>Forced identification would be the failure.</blockquote></>}<div className="closing">MODEL FAILURE <ArrowDown/> STRUCTURAL EXPLANATION <ArrowDown/> EXPERIMENTAL FALSIFICATION <ArrowDown/> IDENTIFIABILITY <ArrowDown/> REPAIR <ArrowDown/> BLIND VALIDATION</div>{identified && <blockquote>“WHEN THE MODEL FAILS, FALSIFIER-X DOESN&apos;T JUST ASK HOW WRONG IT WAS. IT ASKS WHY.”</blockquote>}<button className="primary" onClick={onReset}><RotateCcw size={15}/>RESTART INVESTIGATION</button></div>;
}
