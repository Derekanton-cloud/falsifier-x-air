"use client";

import { Activity, BookOpen, Boxes, FlaskConical, Maximize2, Network, Orbit, ShieldCheck, Wrench } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

export type ConsoleMode = "investigation" | "evidence";
const nav = [
  { id: "investigation", label: "LIVE INVESTIGATION", icon: Activity },
  { id: "evidence", label: "RESEARCH EVIDENCE", icon: BookOpen },
];

export function ConsoleShell({ mode, setMode, presentation, setPresentation, children }: {
  mode: ConsoleMode; setMode: (mode: ConsoleMode) => void; presentation: boolean; setPresentation: (value: boolean) => void; children: ReactNode;
}) {
  // The SSR and first client render deliberately share this deterministic value.
  const [sessionTime, setSessionTime] = useState("--:--");
  useEffect(() => {
    const formatTime = () => setSessionTime(new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(new Date()));
    formatTime();
    const interval = window.setInterval(formatTime, 30_000);
    return () => window.clearInterval(interval);
  }, []);

  return <main className={presentation ? "console presentation" : "console"}>
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Orbit size={17} /></div><div><strong>FALSIFIER-X <em>AIR</em></strong><span>RESEARCH CONSOLE · FOUNDATION v0.1</span></div></div>
      <div className="top-status">{["DIGITAL TWIN", "FALSIFIER", "IDENTIFIABILITY", "REPAIR ENGINE"].map((item) => <span key={item}><i />{item} ONLINE</span>)}</div>
      <div className="session">EXP-FR-900100 <b>SEED 900100</b><time>{sessionTime}</time></div>
    </header>
    <aside className="sidebar">
      <div className="nav-label">WORKSPACE</div>
      {nav.map(({ id, label, icon: Icon }) => <button key={id} className={mode === id ? "nav-item active" : "nav-item"} onClick={() => setMode(id as ConsoleMode)}><Icon size={16}/><span>{label}</span></button>)}
      <div className="nav-sep" />
      {[[Network, "CAUSAL GRAPH"], [FlaskConical, "FALSIFICATION LAB"], [Boxes, "IDENTIFIABILITY"], [Wrench, "MODEL REPAIR"], [ShieldCheck, "BLIND VALIDATION"]].map(([Icon, label]) => {
        const itemLabel = label as string; const ItemIcon = Icon as typeof Network;
        return <button key={itemLabel} className="nav-item muted" onClick={() => document.getElementById(itemLabel.toLowerCase().replaceAll(" ", "-"))?.scrollIntoView({ behavior: "smooth" })}><ItemIcon size={16}/><span>{itemLabel}</span></button>;
      })}
      <button className="presentation-button" onClick={() => setPresentation(!presentation)}><Maximize2 size={15}/>{presentation ? "EXIT PRESENTATION" : "PRESENTATION MODE"}</button>
    </aside>
    <section className="workspace">{children}</section>
    <footer className="statusbar"><span><i className="ok" />DATA INTEGRITY</span><span><i className="ok" />TEMPORAL CONSTRAINT ENFORCED</span><span><i className="ok" />ORACLE ISOLATED</span><span><i className="ok" />DISCOVERY / BLIND SEPARATION</span><span className="source">PRESENTATION / VALIDATED ARTIFACTS</span></footer>
  </main>;
}
