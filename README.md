# FALSIFIER-X AIR — research foundation v0.1

FALSIFIER-X AIR is a synthetic research testbed for a specific question: when a
dynamic aviation disruption model persistently fails, can it localise the
unexplained pattern, test constrained missing-dependency hypotheses through
counterfactual interventions, and use surviving structure to improve a recovery
choice?

It is **not** a claim of a new GNN, a causal-discovery guarantee, or a real-world
aviation control system.

## What is implemented

`python -m falsifier_x_air.demo` demonstrates this closed loop:

1. Build an observable flight/airport/rotation/shared-resource graph.
2. Produce tabular baseline predictions and residual intervals.
3. Assess OOD, persistence, interval violations, residual agreement and graph concentration.
4. Localise high-residual flights and generate only graph-semantic candidates.
5. Select and run one informative counterfactual at a time in a synthetic twin.
6. Update approximate evidence, reject inconsistent candidates, and propose a repair.
7. Choose recovery from measured counterfactual evidence; only then evaluate that choice in the twin.

The twin keeps hidden active mechanisms private. Shared-resource assignments are
observable operational inputs: the graph represents flights using a resource and
the corresponding flight-to-flight dependency. The hidden twin mechanism may or
may not activate that observable dependency. Experiments are paired by scenario
seed so an intervention is compared with the same synthetic exogenous noise.

## Deliberate limitations

The included Ridge predictor is a non-graph baseline; `DelayPredictor` is the
replacement boundary for a future spatio-temporal GNN. The twin is synthetic and
its mechanism effects are not evidence about real aviation. Adequacy thresholds
and likelihood-shaped evidence updates are configurable heuristics requiring
validation calibration; they are not causal p-values or Bayesian posteriors.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m falsifier_x_air.demo
```

See [the research design](docs/RESEARCH_DESIGN.md) for boundaries, evaluation
plan, and future phases.

## Real-data foundation (BTS + NOAA)

The `falsifier_x_air.data` package prepares observed aviation records and dynamic
graph inputs; it does not implement an ST-GNN or claim causal discovery from
observational data. BTS Reporting Carrier On-Time Performance files provide the
flight records. NOAA/NCEI Global Hourly ISD files provide station weather.

Configure a bounded period, filters, paths, graph windows, and chronological
split cutoffs in [configs/data.toml](configs/data.toml). Provide authoritative
airport metadata at `data/metadata/airports.csv` (`airport_code`, latitude,
longitude, IANA `timezone`) and NOAA station metadata at
`data/metadata/noaa_stations.csv` (`station_id`, latitude, longitude; optional
`availability_score`). These metadata files are inputs, not Python constants.

```powershell
python -m falsifier_x_air.data.pipeline download-bts --config configs/data.toml
python -m falsifier_x_air.data.pipeline validate-bts --config configs/data.toml
python -m falsifier_x_air.data.pipeline download-noaa --config configs/data.toml
python -m falsifier_x_air.data.pipeline prepare --config configs/data.toml
python -m falsifier_x_air.data.pipeline build-graph --config configs/data.toml
python -m falsifier_x_air.data.pipeline validate --config configs/data.toml
```

Raw downloads live under `data/raw/`; interim and generated tables under
`data/interim/` and `data/processed/` are ignored by Git. The pipeline creates
canonical timezone-aware flight records, backward-only origin-weather features,
chronological train/validation/test labels, observed rotation and airport-temporal
edges, and a configuration-hashed manifest. Actual delay and delay-cause columns
are outcomes/post-outcome fields and must not enter prediction-time inputs.

If BTS changes its automated archive endpoint, obtain the original Reporting
Carrier archive through the official TranStats Download page and place its `.zip`
or `.csv` in `data/raw/bts/`. Run `validate-bts` before preparation; it rejects
empty files, non-ZIP error pages, archives without exactly one CSV, and inputs
missing the required BTS schema.
