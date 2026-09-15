"use client";

import { useEffect, useState } from "react";
import { ConsoleShell, type ConsoleMode } from "../components/shell/ConsoleShell";
import { Investigation } from "../components/investigation/Investigation";
import { Evidence } from "../components/evidence/Evidence";
import { presentationData } from "../lib/presentation-data";
import type { ResearchData } from "../lib/types";

export default function Page() {
  const [mode, setMode] = useState<ConsoleMode>("investigation"); const [presentation, setPresentation] = useState(false); const [data, setData] = useState<ResearchData>(presentationData); const [offline, setOffline] = useState(false);
  useEffect(() => { fetch("/api/research").then(r => r.ok ? r.json() : Promise.reject()).then((d: ResearchData) => setData(d)).catch(() => setOffline(true)); }, []);
  useEffect(() => { const query = new URLSearchParams(window.location.search); if (query.get("presentation") === "1") setPresentation(true); if (query.get("mode") === "evidence") setMode("evidence"); }, []);
  useEffect(() => { const handler = (e: KeyboardEvent) => { if (e.key.toLowerCase() === "p") setPresentation(v => !v); if (e.key === "Escape") setPresentation(false); if (e.key.toLowerCase() === "e") setMode("evidence"); if (e.key.toLowerCase() === "l") setMode("investigation"); }; window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler); }, []);
  return <ConsoleShell mode={mode} setMode={setMode} presentation={presentation} setPresentation={setPresentation}>{offline && <div className="offline"><b>RESEARCH BACKEND OFFLINE</b><span>Demo mode is using the deterministic validated-artifact fallback.</span></div>}{mode === "investigation" ? <Investigation data={data} presentation={presentation}/> : <Evidence data={data}/>}</ConsoleShell>;
}
