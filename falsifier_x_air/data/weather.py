"""NOAA ISD metadata matching, parsing, and leakage-safe temporal alignment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationMatchReport:
    airports: int
    matched: int


def _distance_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0088
    lat1, lon1 = np.radians([lat1, lon1])
    return 2 * radius * np.arcsin(np.sqrt(np.sin((np.radians(lat2) - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(np.radians(lat2)) * np.sin((np.radians(lon2) - lon1) / 2) ** 2))


def match_airports_to_stations(airports: pd.DataFrame, stations: pd.DataFrame, radius_km: float) -> tuple[pd.DataFrame, StationMatchReport]:
    """Select nearest in-radius station deterministically; availability may break distance ties."""
    stations = stations.copy()
    required = {"station_id", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"Station metadata missing columns: {sorted(missing)}")
    if stations[list(required)].isna().any().any() or stations["station_id"].duplicated().any():
        raise ValueError("Station metadata has missing required values or duplicate station_id entries")
    # NOAA ISD identifiers are strings; preserving that type prevents values such
    # as 70273026451 from becoming a non-existent 70273026451.0 URL.
    stations["station_id"] = stations["station_id"].astype("string")
    rows = []
    for airport in airports.itertuples(index=False):
        candidates = stations.copy()
        candidates["distance_km"] = _distance_km(float(airport.latitude), float(airport.longitude), candidates.latitude, candidates.longitude)
        candidates = candidates[candidates.distance_km <= radius_km]
        if candidates.empty:
            continue
        sort_columns = ["distance_km"]
        ascending = [True]
        if "availability_score" in candidates:
            sort_columns = ["availability_score", "distance_km"]
            ascending = [False, True]
        selected = candidates.sort_values(sort_columns, ascending=ascending, kind="stable").iloc[0]
        rows.append({"airport_code": airport.airport_code, "station_id": selected.station_id, "distance_km": selected.distance_km,
                     "matching_method": "availability_then_distance" if "availability_score" in candidates else "nearest_within_radius"})
    matches = pd.DataFrame(rows, columns=["airport_code", "station_id", "distance_km", "matching_method"])
    return matches, StationMatchReport(len(airports), len(matches))


def parse_noaa_csv(paths: list[Path]) -> pd.DataFrame:
    """Parse NOAA Global Hourly CSV codes; unavailable/sentinel observations remain NaN."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        if not {"STATION", "DATE"} <= set(frame.columns):
            raise ValueError(f"{path.name} is not a NOAA Global Hourly CSV (requires STATION and DATE)")
        result = pd.DataFrame({"station_id": frame.STATION.astype("string"), "timestamp_utc": pd.to_datetime(frame.DATE, utc=True, errors="coerce")})
        for source, target, missing in (("TMP", "temperature_c", 9999), ("DEW", "dew_point_c", 9999), ("SLP", "pressure_hpa", 99999), ("VIS", "visibility_m", 999999)):
            values = frame.get(source, pd.Series(index=frame.index, dtype="string")).astype("string").str.split(",").str[0]
            numeric = pd.to_numeric(values, errors="coerce").replace(missing, np.nan)
            result[target] = numeric / 10 if source in {"TMP", "DEW", "SLP"} else numeric
        wind = frame.get("WND", pd.Series(index=frame.index, dtype="string")).astype("string").str.split(",")
        result["wind_direction_degrees"] = pd.to_numeric(wind.str[0], errors="coerce").replace(999, np.nan)
        result["wind_speed_mps"] = pd.to_numeric(wind.str[3], errors="coerce").replace(9999, np.nan) / 10
        result["wind_gust_mps"] = pd.to_numeric(frame.get("GUST", pd.Series(index=frame.index, dtype="string")).astype("string").str.split(",").str[0], errors="coerce").replace(9999, np.nan) / 10
        frames.append(result)
    return pd.concat(frames, ignore_index=True).dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc") if frames else pd.DataFrame()


def align_weather(flights: pd.DataFrame, weather: pd.DataFrame, matches: pd.DataFrame, tolerance_minutes: int) -> pd.DataFrame:
    """Backward-only as-of joins: no weather after scheduled operation time enters features."""
    origin_matches = matches.rename(columns={"airport_code": "origin_airport", "station_id": "origin_station_id", "distance_km": "origin_station_distance_km", "matching_method": "origin_station_matching_method"})
    flight = flights.merge(origin_matches, on="origin_airport", how="left")
    def join(frame: pd.DataFrame, station_column: str, timestamp_column: str, prefix: str) -> pd.DataFrame:
        left = frame.sort_values(timestamp_column)
        right = weather.rename(columns={"station_id": station_column}).sort_values("timestamp_utc")
        right["timestamp_utc"] = right["timestamp_utc"].astype(left[timestamp_column].dtype)
        left[station_column] = left[station_column].astype(object)
        right[station_column] = right[station_column].astype(object)
        result = pd.merge_asof(left, right, left_on=timestamp_column, right_on="timestamp_utc", by=station_column,
                               direction="backward", tolerance=pd.Timedelta(minutes=tolerance_minutes))
        result["weather_observation_utc"] = result["timestamp_utc"]
        result["weather_age_minutes"] = (result[timestamp_column] - result["timestamp_utc"]).dt.total_seconds() / 60
        renamed = {column: f"{prefix}_{column}" for column in weather.columns if column not in {"station_id", "timestamp_utc"}}
        renamed.update({"weather_observation_utc": f"{prefix}_observation_utc", "weather_age_minutes": f"{prefix}_age_minutes"})
        return result.rename(columns=renamed).drop(columns="timestamp_utc")
    origin = join(flight, "origin_station_id", "scheduled_departure_utc", "origin_weather")
    return origin
