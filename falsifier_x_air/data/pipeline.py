"""Stage orchestration, manifests, quality reporting, and command entry points."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .config import DataPipelineConfig, load_config
from .flights import clean_bts_files, load_airports, validate_bts_archive
from .graph_data import build_dynamic_edges, chronological_split, graph_report
from .weather import align_weather, match_airports_to_stations, parse_noaa_csv


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.exists() else pd.read_csv(path.with_suffix(".csv"))
    for column in frame.columns:
        if column.endswith("_utc") or column in {"source_time", "target_time"}:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def _operating_airports(flights: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Restrict weather metadata to airports represented by the prepared flights."""
    codes = set(flights.origin_airport.dropna()) | set(flights.destination_airport.dropna())
    selected = airports[airports.airport_code.isin(codes)].copy()
    missing = sorted(codes - set(selected.airport_code))
    if missing:
        raise ValueError(f"Prepared flights contain airports absent from airport metadata: {missing}")
    return selected.reset_index(drop=True)


def _require_complete_station_matches(matches: pd.DataFrame, airports: pd.DataFrame) -> None:
    missing = sorted(set(airports.airport_code) - set(matches.airport_code))
    if missing:
        raise ValueError(f"No configured NOAA station within the configured radius for prepared airports: {missing}")


def download_bts(config: DataPipelineConfig) -> list[Path]:
    """Download configured monthly BTS pre-zips only; caller controls date range."""
    config.ensure_directories()
    outputs = []
    for month in pd.period_range(config.start_date, config.end_date, freq="M"):
        filename = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{month.year}_{month.month}.zip"
        output = config.raw_bts / filename
        if not output.exists():
            urlretrieve(f"{config.bts_prezip_url}/{filename}", output)
        outputs.append(output)
    return outputs


def download_noaa(config: DataPipelineConfig, matches: pd.DataFrame) -> list[Path]:
    """Download official Global Hourly files safely, validating before publish."""
    config.ensure_directories()
    targets = [(str(station), year, config.raw_noaa / f"{station}_{year}.csv")
               for station in sorted(matches.station_id.astype("string").unique())
               for year in range(config.start_date.year, config.end_date.year + 1)]

    def validate_observation_file(path: Path, station: str) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"NOAA observation download is missing or empty: {path}")
        header = pd.read_csv(path, nrows=0)
        if not {"STATION", "DATE"} <= set(header.columns):
            raise ValueError(f"{path.name} is not a NOAA Global Hourly CSV (requires STATION and DATE)")
        observed: set[str] = set()
        for chunk in pd.read_csv(path, usecols=["STATION", "DATE"], dtype={"STATION": "string"}, chunksize=100_000):
            observed.update(chunk["STATION"].dropna().astype(str))
        if observed != {station}:
            raise ValueError(f"{path.name} contains station IDs {sorted(observed)}, expected {station}")

    def download_one(station: str, year: int, output: Path) -> Path:
        if output.exists():
            try:
                validate_observation_file(output, station)
                return output
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
                output.unlink()
        temporary = output.with_suffix(".csv.part")
        temporary.unlink(missing_ok=True)
        try:
            urlretrieve(f"{config.noaa_isd_base_url}/{year}/{station}.csv", temporary)
            validate_observation_file(temporary, station)
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return output

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(lambda target: download_one(*target), targets))
    provenance = {
        "source": config.noaa_isd_base_url,
        "study_period": [config.start_date.isoformat(), config.end_date.isoformat()],
        "station_count": len(set(matches.station_id.astype("string"))),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in outputs
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (config.metadata / "noaa_observations_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return outputs


def prepare(config: DataPipelineConfig) -> dict[str, object]:
    airports = load_airports(config.airport_metadata)
    paths = sorted([*config.raw_bts.glob("*.csv"), *config.raw_bts.glob("*.zip")])
    if not paths:
        raise FileNotFoundError(f"No BTS archive found in {config.raw_bts}. Download through official TranStats UI and place the verified .zip/.csv there.")
    archive_metadata = [validate_bts_archive(path) for path in paths]
    flights, quality = clean_bts_files(paths, airports, config.max_flights, config.airports, config.carriers)
    flight_output = _write_table(flights, config.processed / "flights" / "flights.parquet")
    if not config.station_metadata.exists():
        return {"flights": str(flight_output), "quality": quality.__dict__, "bts_inputs": archive_metadata,
                "weather_status": "not prepared: NOAA station metadata has not yet been supplied"}
    stations = pd.read_csv(config.station_metadata, dtype={"station_id": "string"})
    operating_airports = _operating_airports(flights, airports)
    matches, station_report = match_airports_to_stations(operating_airports, stations, config.station_radius_km)
    _require_complete_station_matches(matches, operating_airports)
    _write_table(matches, config.metadata / "airport_station_matches.parquet")
    weather_paths = sorted(
        path for year in range(config.start_date.year, config.end_date.year + 1)
        for path in config.raw_noaa.glob(f"*_{year}.csv")
    )
    weather = parse_noaa_csv(weather_paths) if weather_paths else pd.DataFrame()
    weather_output = _write_table(weather, config.processed / "weather" / "weather.parquet")
    joined = align_weather(flights, weather, matches, config.matching_tolerance_minutes) if not flights.empty and not weather.empty else flights.copy()
    joined_output = _write_table(joined, config.processed / "joined" / "flights_weather.parquet")
    return {"flights": str(flight_output), "weather": str(weather_output), "joined": str(joined_output), "quality": quality.__dict__, "station_matches": station_report.__dict__, "bts_inputs": archive_metadata}


def build_graph(config: DataPipelineConfig) -> dict[str, object]:
    flights = _read_table(config.processed / "flights" / "flights.parquet")
    edges = build_dynamic_edges(flights, config.minimum_rotation_minutes, config.maximum_rotation_minutes, config.airport_propagation_minutes)
    _write_table(edges, config.processed / "graphs" / "edges.parquet")
    airports = pd.DataFrame({"node_id": pd.concat([flights.origin_airport, flights.destination_airport]).dropna().unique(), "node_type": "airport"})
    flight_nodes = pd.DataFrame({"node_id": flights.flight_id, "node_type": "flight", "timestamp_utc": flights.scheduled_departure_utc})
    _write_table(pd.concat([airports, flight_nodes], ignore_index=True), config.processed / "graphs" / "nodes.parquet")
    split = chronological_split(flights, config.train_end.isoformat(), config.validation_end.isoformat())
    _write_table(pd.DataFrame({"flight_id": flights.flight_id, "scheduled_departure_utc": flights.scheduled_departure_utc, "target_arrival_delay_minutes": flights.arrival_delay_minutes, "split": split}), config.processed / "graphs" / "baseline_ready.parquet")
    report = graph_report(flights, edges)
    return {"graph": report.__dict__, "split_counts": split.value_counts(dropna=False).to_dict()}


def validate(config: DataPipelineConfig) -> dict[str, object]:
    """Write a reproducible quality report over prepared artifacts."""
    flights = _read_table(config.processed / "flights" / "flights.parquet")
    edges = _read_table(config.processed / "graphs" / "edges.parquet")
    joined_path = config.processed / "joined" / "flights_weather.parquet"
    joined = _read_table(joined_path) if joined_path.exists() or joined_path.with_suffix(".csv").exists() else flights
    missing_timestamps = {column: int(flights[column].isna().sum()) for column in ("scheduled_departure_utc", "scheduled_arrival_utc", "actual_departure_utc", "actual_arrival_utc") if column in flights}
    weather_columns = [column for column in joined if column.startswith("origin_weather_")]
    weather_matched = int(joined["origin_weather_observation_utc"].notna().sum()) if "origin_weather_observation_utc" in joined else 0
    report = {
        "flight_rows": len(flights), "unique_flights": int(flights.flight_id.nunique()),
        "date_range": [str(flights.flight_date.min()), str(flights.flight_date.max())],
        "earliest_timestamp_utc": str(flights.scheduled_departure_utc.min()), "latest_timestamp_utc": str(flights.scheduled_departure_utc.max()),
        "airports": sorted(set(flights.origin_airport.dropna()) | set(flights.destination_airport.dropna())),
        "carriers": sorted(flights.carrier.dropna().unique().tolist()),
        "missing_fraction": {column: float(value) for column, value in flights.isna().mean().items()},
        "missing_timestamp_counts": missing_timestamps, "duplicate_flights": int(flights.duplicated("flight_id").sum()),
        "missing_airport_metadata": int(flights[["origin_timezone", "destination_timezone"]].isna().any(axis=1).sum()), "cancelled": int(flights.cancelled.sum()), "diverted": int(flights.diverted.sum()),
        "weather_matched": weather_matched, "weather_unmatched": len(joined) - weather_matched, "weather_match_percentage": 100 * weather_matched / len(joined) if len(joined) else 0.0,
        "missing_weather_feature_counts": {column: int(joined[column].isna().sum()) for column in weather_columns},
        "arrival_delay_summary": flights.arrival_delay_minutes.describe().to_dict(), "graph": graph_report(flights, edges).__dict__,
    }
    output = config.metadata / "data_quality_report.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def write_manifest(config: DataPipelineConfig, details: dict[str, object]) -> Path:
    digest = hashlib.sha256((config.root / "configs" / "data.toml").read_bytes()).hexdigest()
    manifest = {"schema_version": config.schema_version, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "configuration_sha256": digest,
                "sources": {"bts": config.bts_prezip_url, "noaa": config.noaa_isd_base_url}, **details}
    output = config.metadata / "dataset_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="FALSIFIER-X AIR data pipeline")
    parser.add_argument("stage", choices=("download-bts", "validate-bts", "download-noaa", "prepare", "build-graph", "validate"))
    parser.add_argument("--config", default="configs/data.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "download-bts":
        result: dict[str, object] = {"downloads": [str(path) for path in download_bts(config)]}
    elif args.stage == "validate-bts":
        inputs = sorted([*config.raw_bts.glob("*.csv"), *config.raw_bts.glob("*.zip")])
        if not inputs:
            raise FileNotFoundError(f"No BTS .zip/.csv files found in {config.raw_bts}. Place an official TranStats archive there first.")
        result = {"inputs": [validate_bts_archive(path) for path in inputs]}
    elif args.stage == "download-noaa":
        match_path = config.metadata / "airport_station_matches.parquet"
        if match_path.exists() or match_path.with_suffix(".csv").exists():
            matches = _read_table(match_path)
        else:
            flights_path = config.processed / "flights" / "flights.parquet"
            if not flights_path.exists() and not flights_path.with_suffix(".csv").exists():
                raise FileNotFoundError("Prepared flights are required to derive the actual airport set for NOAA downloads; run prepare after supplying station metadata.")
            flights = _read_table(flights_path)
            airports = _operating_airports(flights, load_airports(config.airport_metadata))
            stations = pd.read_csv(config.station_metadata, dtype={"station_id": "string"})
            matches, _ = match_airports_to_stations(airports, stations, config.station_radius_km)
            _require_complete_station_matches(matches, airports)
            _write_table(matches, match_path)
        result = {"downloads": [str(path) for path in download_noaa(config, matches)]}
    elif args.stage == "prepare":
        result = prepare(config)
    elif args.stage == "build-graph":
        result = build_graph(config)
    else:
        result = validate(config)
    manifest = write_manifest(config, result)
    print(f"Completed {args.stage}; manifest: {manifest}")


if __name__ == "__main__":
    main()
