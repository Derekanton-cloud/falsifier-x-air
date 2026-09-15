# FALSIFIER-X AIR Research Console

This is an isolated Next.js / TypeScript presentation layer for the existing Python research testbed. It does not alter the learner, evaluator, experiment seeds, digital twin, or scientific algorithms.

## Architecture

- `app/api/research/route.ts` is a read-only adapter over `../data/processed/final_robustness_results.json`, the canonical Python-generated final-robustness artifact.
- `lib/presentation-data.ts` is a deterministic offline fallback containing values transcribed from that same artifact. It is only used if the local artifact cannot be read.
- The Live Investigation control calls `frontend/bridge/live_investigation.py`, a JSON-only wrapper around the existing public `AviationDigitalTwin` / `FalsifierXAir` demonstration path. It falls back to a clearly labeled deterministic presentation trace if Python cannot be launched. Final 300-case blind metrics are labeled cached validated experiment results.
- `components/graph/AviationGraph.tsx` renders that actual demo topology with React Three Fiber. It is presentation-scale rather than the full production graph, so it stays smooth on a laptop/projector.

The client never receives benchmark truth, hidden twin mechanisms, or evaluator-only oracle fields. The API has no write endpoints and does not execute expensive robustness evaluations.

## Prerequisites

Node.js 20 LTS or newer. The Python artifact already exists in the repository; no Python server is required for the standard console.

For the live Python bridge, set `FALSIFIER_PYTHON` to your Python executable if `python` is not on PATH. Example PowerShell: `$env:FALSIFIER_PYTHON = 'C:\\Users\\derek\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'`. Git Bash: `export FALSIFIER_PYTHON='/c/Users/derek/AppData/Local/Programs/Python/Python312/python.exe'`.

## Windows PowerShell

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). For a production check:

```powershell
npm run typecheck
npm run lint
npm run build
npm run start
```

## Git Bash

```bash
cd frontend
npm install
npm run dev
```

For a production check:

```bash
npm run typecheck
npm run lint
npm run build
npm run start
```

## Presentation controls

- **Presentation Mode** in the left rail, or `P`, expands the investigation view.
- `L` selects Live Investigation; `E` selects Research Evidence; `Esc` exits presentation mode.
- **Load Investigation** prepares the initial observation; **Assess Inadequacy** is the deliberate action that runs the public Python investigation trace.
- **Identifiable Case** follows recovery and cached blind validation.
- **Indistinguishable Case** stops at the identifiability gate and visibly records abstention.
- **Reset** restarts the deterministic trace.

Directly open presentation mode: [http://localhost:3000/?presentation=1](http://localhost:3000/?presentation=1). Add `&mode=evidence` for the evidence observatory.

## Data provenance

The research evidence interface uses `data/processed/final_robustness_results.json`; Phase 10C and accompanying reports remain in their original Python artifact locations for detailed inspection. The UI intentionally states the limitations: constrained mechanism class, synthetic causal validation, abstention/coverage limitations, limited nonlinear breadth, and configuration-dependent active intervention efficiency.
