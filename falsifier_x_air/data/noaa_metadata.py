"""Build validated NOAA/NCEI ISD station metadata for prepared BTS airports."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_STATION_COLUMNS = ("station_id", "latitude", "longitude")


def _distance_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0088
    lat1, lon1 = np.radians([lat1, lon1])
    return 2 * radius * np.arcsin(np.sqrt(
        np.sin((np.radians(lat2) - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(np.radians(lat2)) * np.sin((np.radians(lon2) - lon1) / 2) ** 2
    ))


def validate_noaa_stations(stations: pd.DataFrame) -> None:
    """Validate the exact metadata contract consumed by weather matching."""
    if tuple(stations.columns) != REQUIRED_STATION_COLUMNS:
        raise ValueError(f"noaa_stations.csv schema must be exactly {list(REQUIRED_STATION_COLUMNS)}")
    if stations.isna().any().any() or stations.station_id.duplicated().any():
        raise ValueError("NOAA station metadata has missing values or duplicate station IDs")
    if not stations.latitude.between(-90, 90).all() or not stations.longitude.between(-180, 180).all():
        raise ValueError("NOAA station metadata contains invalid coordinates")


def _read_isd_history(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    required = {"USAF", "WBAN", "LAT", "LON", "BEGIN", "END"}
    history = pd.read_csv(path, dtype={"USAF": "string", "WBAN": "string"})
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"NOAA ISD history source missing fields: {sorted(missing)}")
    history["BEGIN"] = pd.to_datetime(history["BEGIN"].astype("string"), format="%Y%m%d", errors="coerce")
    history["END"] = pd.to_datetime(history["END"].astype("string"), format="%Y%m%d", errors="coerce")
    history["LAT"] = pd.to_numeric(history["LAT"], errors="coerce")
    history["LON"] = pd.to_numeric(history["LON"], errors="coerce")
    history["station_id"] = history["USAF"].str.zfill(6) + history["WBAN"].str.zfill(5)
    valid = history[
        (history["BEGIN"] <= start_date)
        & (history["END"] >= end_date)
        & history["LAT"].between(-90, 90)
        & history["LON"].between(-180, 180)
        & ~((history["LAT"] == 0) & (history["LON"] == 0))
    ].copy()
    if valid.empty:
        raise ValueError("NOAA ISD history source contains no stations valid for the configured period")
    duplicates = valid[valid.station_id.duplicated(keep=False)]
    if not duplicates.empty:
        conflicting = duplicates.groupby("station_id")[["LAT", "LON"]].nunique().max(axis=1)
        if (conflicting > 1).any():
            raise ValueError("NOAA ISD history has conflicting coordinates for an active station ID")
        valid = valid.drop_duplicates("station_id", keep="first")
    return valid


def build_noaa_station_metadata(
    flights_path: Path,
    airports_path: Path,
    isd_history_path: Path,
    output_path: Path,
    matches_path: Path,
    provenance_path: Path,
    start_date: str,
    end_date: str,
    radius_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the nearest official ISD station for every airport in prepared flights."""
    flights = pd.read_parquet(flights_path) if flights_path.suffix == ".parquet" else pd.read_csv(flights_path, low_memory=False)
    observed = sorted(set(flights["origin_airport"].dropna()) | set(flights["destination_airport"].dropna()))
    airports = pd.read_csv(airports_path)
    airports = airports[airports.airport_code.isin(observed)].copy()
    if set(airports.airport_code) != set(observed):
        raise ValueError("Prepared flights contain airports absent from airports.csv")
    if airports[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Airport metadata contains missing coordinates")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    history = _read_isd_history(isd_history_path, start, end)

    rows = []
    for airport in airports.sort_values("airport_code").itertuples(index=False):
        candidates = history.copy()
        candidates["distance_km"] = _distance_km(float(airport.latitude), float(airport.longitude), candidates["LAT"], candidates["LON"])
        candidates = candidates[candidates.distance_km <= radius_km].sort_values(["distance_km", "station_id"], kind="stable")
        if candidates.empty:
            continue
        station = candidates.iloc[0]
        rows.append({"airport_code": airport.airport_code, "station_id": station.station_id, "distance_km": station.distance_km,
                     "matching_method": "nearest_valid_noaa_isd_station_within_radius", "source_station_name": station.get("STATION NAME", pd.NA),
                     "source_icao": station.get("ICAO", pd.NA), "source_begin": station.BEGIN.date().isoformat(), "source_end": station.END.date().isoformat()})
    matches = pd.DataFrame(rows, columns=["airport_code", "station_id", "distance_km", "matching_method", "source_station_name", "source_icao", "source_begin", "source_end"])
    unmatched = sorted(set(observed) - set(matches.airport_code))
    if unmatched:
        raise ValueError(f"No valid NOAA ISD station within {radius_km} km for prepared airports: {unmatched}")

    selected = history.merge(matches[["station_id"]].drop_duplicates(), on="station_id", how="inner")
    stations = selected.rename(columns={"LAT": "latitude", "LON": "longitude"})[[*REQUIRED_STATION_COLUMNS]].sort_values("station_id").reset_index(drop=True)
    validate_noaa_stations(stations)
    if not set(stations.station_id).issubset(set(history.station_id)):
        raise ValueError("Selected station does not exist in the authoritative NOAA ISD history source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(output_path, index=False)
    matches.to_csv(matches_path, index=False)
    provenance = {
        "study_period": [start.date().isoformat(), end.date().isoformat()], "airport_count_prepared": len(observed),
        "airport_count_matched": len(matches), "airport_count_unmatched": len(unmatched), "unmatched_airports": unmatched,
        "station_count": len(stations), "station_source": "NOAA/NCEI Integrated Surface Database (ISD) station history",
        "station_source_url": "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
        "station_source_file": str(isd_history_path), "station_source_sha256": hashlib.sha256(isd_history_path.read_bytes()).hexdigest(),
        "selection_rules": [
            "Use only station-history records with BEGIN <= configured start date and END >= configured end date.",
            "Require non-null latitude/longitude in valid geographic bounds; reject the 0,0 placeholder.",
            f"For each airport represented in prepared flights, select the nearest valid station within {radius_km} km using haversine distance (Earth radius 6371.0088 km).",
            "Break equal-distance ties by station_id ascending.",
        ],
        "availability_score": "omitted: no NOAA observation availability was downloaded or computed",
        "matches_file": str(matches_path), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return stations, matches
