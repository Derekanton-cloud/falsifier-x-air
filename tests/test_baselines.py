"""Tests for baseline features and prediction contract."""

import pandas as pd
import numpy as np
import pytest
from falsifier_x_air.data.features import generate_features, prepare_training_data


@pytest.fixture
def sample_joined_data():
    """Create a small synthetic dataset for testing leakage and feature generation.

    Airport congestion mechanics (closed='left' rolling):
    - The rolling window at timestamp T includes arrivals in the half-open interval
      [T - 3h, T). Arrivals AT T are excluded.
    - The congestion map stores one value per arrival timestamp. A merge_asof with
      direction='backward' picks the most recent entry <= the departure timestamp.
    - Therefore, for a departure at time D to see non-NaN congestion at airport X,
      there must be at least TWO arrivals at X: an earlier arrival A1 whose rolling
      window (closed='left') includes A1's preceding arrival, producing a non-NaN
      value. Then merge_asof picks that value for the departure at D.

    Dataset layout:
      X0  departs Z at 08:00, arrives B at 09:00 (delay 5)   <- "seed" arrival
      F1  departs A at 10:00, arrives B at 12:00 (delay 10)  <- rolling at 12:00 includes X0 -> mean=7.5
      F2  departs B at 14:00, arrives A at 16:00 (delay 20)  <- picks congestion from 12:00 entry (7.5)

      X1  departs Z at 09:00, arrives C at 10:00 (delay 25)  <- seed for C
      F3  departs A at 11:00, arrives C at 13:00 (delay 30)  <- rolling at 13:00 includes X1 -> mean=27.5
      F4  departs C at 15:00, arrives A at 17:00 (delay 40)  <- picks congestion from 13:00 (27.5)
    """
    data = {
        "flight_id": ["X0", "X1", "F1", "F2", "F3", "F4"],
        "tail_number": ["T0", "T9", "T1", "T1", "T2", "T2"],
        "origin_airport": ["Z", "Z", "A", "B", "A", "C"],
        "destination_airport": ["B", "C", "B", "A", "C", "A"],
        "scheduled_departure_utc": [
            "2024-01-01 08:00:00+00:00",  # X0
            "2024-01-01 09:00:00+00:00",  # X1
            "2024-01-01 10:00:00+00:00",  # F1
            "2024-01-01 14:00:00+00:00",  # F2
            "2024-01-01 11:00:00+00:00",  # F3
            "2024-01-01 15:00:00+00:00",  # F4
        ],
        "actual_arrival_utc": [
            "2024-01-01 09:00:00+00:00",  # X0 arrives B at 09:00 (delay 5)
            "2024-01-01 10:00:00+00:00",  # X1 arrives C at 10:00 (delay 25)
            "2024-01-01 12:00:00+00:00",  # F1 arrives B at 12:00 (delay 10)
            "2024-01-01 16:00:00+00:00",  # F2 arrives A at 16:00
            "2024-01-01 13:00:00+00:00",  # F3 arrives C at 13:00 (delay 30)
            "2024-01-01 17:00:00+00:00",  # F4 arrives A at 17:00
        ],
        "arrival_delay_minutes": [5.0, 25.0, 10.0, 20.0, 30.0, 40.0],
        "scheduled_duration_minutes": [60, 60, 110, 100, 120, 110],
        "distance": [500, 500, 1000, 1000, 1200, 1200],
        "carrier": ["AA", "AA", "AA", "AA", "DL", "DL"],
        "origin_weather_temperature_c": [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
        "origin_weather_dew_point_c": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        "origin_weather_pressure_hpa": [1015, 1014, 1013, 1012, 1011, 1010],
        "origin_weather_visibility_m": [12000, 11000, 10000, 9000, 8000, 7000],
        "origin_weather_wind_direction_degrees": [170, 175, 180, 190, 200, 210],
        "origin_weather_wind_speed_mps": [4, 5, 5, 6, 7, 8],
    }
    return pd.DataFrame(data)


def test_generate_features_causality(sample_joined_data):
    features = generate_features(sample_joined_data)

    # F1: tail T1, first flight (X0 is T0). prev_aircraft_delay should be NaN.
    # F2: tail T1, second flight. Departs 14:00. F1 arrived B at 12:00. prev_aircraft_delay = 10.0.
    f1 = features[features.flight_id == "F1"].iloc[0]
    f2 = features[features.flight_id == "F2"].iloc[0]

    assert np.isnan(f1.prev_aircraft_delay)
    assert f2.prev_aircraft_delay == 10.0

    # F4: tail T2, second flight. Departs 15:00. F3 arrived C at 13:00. prev_aircraft_delay = 30.0.
    f4 = features[features.flight_id == "F4"].iloc[0]
    assert f4.prev_aircraft_delay == 30.0


def test_airport_congestion_causality(sample_joined_data):
    features = generate_features(sample_joined_data)

    # F2 departs from B at 14:00.
    # Arrivals at B: X0 at 09:00 (delay=5), F1 at 12:00 (delay=10).
    # Rolling at F1's arrival (12:00, closed='left') covers [09:00, 12:00) -> includes X0 (09:00).
    # mean = 5.0. merge_asof picks this value for F2's departure at 14:00.
    f2 = features[features.flight_id == "F2"].iloc[0]
    assert f2.airport_congestion == pytest.approx(5.0), (
        f"Expected congestion=5.0 at B (X0 delay=5 in window), got {f2.airport_congestion}"
    )

    # F4 departs from C at 15:00.
    # Arrivals at C: X1 at 10:00 (delay=25), F3 at 13:00 (delay=30).
    # Rolling at F3's arrival (13:00, closed='left') covers [10:00, 13:00) -> includes X1 (10:00).
    # mean = 25.0. merge_asof picks this for F4's departure at 15:00.
    f4 = features[features.flight_id == "F4"].iloc[0]
    assert f4.airport_congestion == pytest.approx(25.0), (
        f"Expected congestion=25.0 at C (X1 delay=25 in window), got {f4.airport_congestion}"
    )


def test_forbidden_features_exclusion(sample_joined_data):
    features = generate_features(sample_joined_data)
    
    forbidden = ["actual_arrival_utc", "arrival_delay_minutes", "actual_departure_utc", "departure_delay_minutes"]
    for col in forbidden:
        assert col not in features.columns


def test_prepare_training_data_drops_nans():
    features = pd.DataFrame({"flight_id": ["F1", "F2"], "feat1": [1, 2]})
    baseline_ready = pd.DataFrame({
        "flight_id": ["F1", "F2"],
        "target_arrival_delay_minutes": [10.0, np.nan],
        "split": ["train", "test"]
    })
    
    valid = prepare_training_data(features, baseline_ready)
    assert len(valid) == 1
    assert valid.iloc[0].flight_id == "F1"
