"""Feature engineering for aviation delay prediction; strictly enforces temporal causal boundaries."""

from datetime import timedelta
import pandas as pd
import numpy as np


def generate_features(joined_data: pd.DataFrame) -> pd.DataFrame:
    """Transform joined flight/weather data into prediction-time features.
    
    This layer strictly excludes post-outcome information and future leakage.
    """
    df = joined_data.copy()
    df["scheduled_departure_utc"] = pd.to_datetime(df["scheduled_departure_utc"], utc=True)
    df["actual_arrival_utc"] = pd.to_datetime(df["actual_arrival_utc"], utc=True)

    # 1. Temporal Features
    df["dep_hour"] = df["scheduled_departure_utc"].dt.hour
    df["dep_day_of_week"] = df["scheduled_departure_utc"].dt.dayofweek

    # 2. Historical Operational Features (Strictly Causal)
    # We must sort by scheduled departure to ensure we only look backward.
    df = df.sort_values("scheduled_departure_utc")

    # Aircraft-level (rotation) historical delay
    # Previous flight for the same tail_number
    # Important: df is sorted by scheduled_departure_utc
    df["prev_aircraft_delay"] = df.groupby("tail_number")["arrival_delay_minutes"].shift(1)
    # Validate that the previous flight's arrival was actually before current departure
    prev_arrival = df.groupby("tail_number")["actual_arrival_utc"].shift(1)
    mask = (prev_arrival >= df["scheduled_departure_utc"]) | prev_arrival.isna()
    df.loc[mask, "prev_aircraft_delay"] = np.nan

    # Airport-level historical delay (previous 3 hours)
    # We use a rolling mean on arrivals, then map it back to departures at the same airport.
    arrivals = df.dropna(subset=["actual_arrival_utc"]).sort_values("actual_arrival_utc").copy()

    congestion_results = []
    for airport, group in arrivals.groupby("destination_airport"):
        # We need the index to be actual_arrival_utc for rolling window to work
        # But we must handle duplicate timestamps.
        indexed = group.set_index("actual_arrival_utc")
        # Use a left-closed rolling window to exclude arrivals at the exact departure time (strict causality)
        group_congestion = indexed["arrival_delay_minutes"].rolling("3h", closed="left").mean()
        # Reset index but keep the timestamps
        res = group_congestion.reset_index()
        res.columns = ["match_time", "airport_congestion"]
        res["origin_airport"] = airport
        congestion_results.append(res)

    congestion_map = pd.concat(congestion_results).sort_values("match_time")

    df = pd.merge_asof(
        df.sort_values("scheduled_departure_utc"),
        congestion_map,
        left_on="scheduled_departure_utc",
        right_on="match_time",
        by="origin_airport",
        direction="backward"
    )

    # 3. Clean and Select
    # Use only allowed features
    feature_columns = [
        "carrier", "origin_airport", "destination_airport",
        "scheduled_duration_minutes", "distance",
        "dep_hour", "dep_day_of_week",
        "origin_weather_temperature_c", "origin_weather_dew_point_c",
        "origin_weather_pressure_hpa", "origin_weather_visibility_m",
        "origin_weather_wind_direction_degrees", "origin_weather_wind_speed_mps",
        "prev_aircraft_delay", "airport_congestion"
    ]
    
    # We keep flight_id for joining back to splits
    return df[["flight_id"] + feature_columns]


def prepare_training_data(features: pd.DataFrame, baseline_ready: pd.DataFrame):
    """Join features with split labels and targets, dropping invalid rows."""
    full = features.merge(baseline_ready, on="flight_id")
    
    # Drop cancelled/diverted (target is NaN)
    valid = full.dropna(subset=["target_arrival_delay_minutes"])
    
    return valid
