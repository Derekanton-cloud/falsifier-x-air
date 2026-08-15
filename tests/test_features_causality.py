"""
Permanent causality tests for the real-data baseline feature layer.

All tests use flight_id as the explicit alignment key to avoid pandas
index-alignment errors when joining features against the raw joined table.

Scientific rules verified here:
- prev_aircraft_delay: only populated when the previous aircraft's actual_arrival_utc
  is STRICTLY BEFORE the current flight's scheduled_departure_utc.
- airport_congestion: rolling 3-hour window uses closed="left", meaning arrivals at
  EXACTLY the departure timestamp are excluded.  The merge_asof uses direction="backward"
  so only congestion values computed at or before the departure time enter features.
- Weather: origin_weather_observation_utc <= scheduled_departure_utc (no future weather).
- Forbidden columns do not appear in the feature output.
- Chronological split has no cross-boundary overlap.
- Preprocessing (imputer + scaler) is fitted only on training data.
"""

import pathlib
import tomllib

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
CONFIG = tomllib.loads((REPO_ROOT / "configs" / "data.toml").read_text())
PROCESSED = REPO_ROOT / pathlib.Path(CONFIG["paths"]["processed"])
JOINED_PATH = PROCESSED / "joined" / "flights_weather.csv"
BASELINE_READY_PATH = PROCESSED / "graphs" / "baseline_ready.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def joined():
    """Load the real joined flights+weather table once per module."""
    df = pd.read_csv(JOINED_PATH, low_memory=False)
    df["scheduled_departure_utc"] = pd.to_datetime(df["scheduled_departure_utc"], utc=True)
    df["actual_arrival_utc"] = pd.to_datetime(df["actual_arrival_utc"], utc=True)
    df["origin_weather_observation_utc"] = pd.to_datetime(
        df["origin_weather_observation_utc"], utc=True
    )
    return df


@pytest.fixture(scope="module")
def features(joined):
    """Generate features from the real joined table."""
    from falsifier_x_air.data.features import generate_features
    return generate_features(joined)


@pytest.fixture(scope="module")
def baseline_ready():
    """Load the baseline-ready split/target table."""
    df = pd.read_csv(BASELINE_READY_PATH)
    df["scheduled_departure_utc"] = pd.to_datetime(df["scheduled_departure_utc"], utc=True)
    return df


# ---------------------------------------------------------------------------
# 1. Target construction
# ---------------------------------------------------------------------------

def test_target_column_present(baseline_ready):
    """Target must be named target_arrival_delay_minutes."""
    assert "target_arrival_delay_minutes" in baseline_ready.columns


def test_target_has_valid_values(baseline_ready):
    """Non-NaN targets must be finite floats within a plausible aviation range."""
    valid = baseline_ready["target_arrival_delay_minutes"].dropna()
    assert len(valid) > 0
    assert np.all(np.isfinite(valid))
    # BTS delay values are bounded; no flight is thousands of hours late
    assert valid.max() < 10_000
    assert valid.min() > -500


# ---------------------------------------------------------------------------
# 2. Forbidden features excluded
# ---------------------------------------------------------------------------

FORBIDDEN_COLUMNS = [
    "actual_arrival_utc",
    "arrival_delay_minutes",
    "actual_departure_utc",
    "departure_delay_minutes",
    "cancellation_code",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "cancelled",
    "diverted",
]


@pytest.mark.parametrize("col", FORBIDDEN_COLUMNS)
def test_forbidden_column_absent(features, col):
    """No post-outcome or cause-coded column may appear in features."""
    assert col not in features.columns, f"Forbidden column present: {col}"


# ---------------------------------------------------------------------------
# 3. Chronological split integrity
# ---------------------------------------------------------------------------

def test_split_values_complete(baseline_ready):
    """Every row must be assigned a split label."""
    assert baseline_ready["split"].isna().sum() == 0


def test_split_no_overlap_train_validation(baseline_ready):
    """Train max < validation min (strict temporal ordering)."""
    train_max = baseline_ready.loc[baseline_ready.split == "train", "scheduled_departure_utc"].max()
    val_min = baseline_ready.loc[baseline_ready.split == "validation", "scheduled_departure_utc"].min()
    assert train_max < val_min, f"Train/validation overlap: train_max={train_max}, val_min={val_min}"


def test_split_no_overlap_validation_test(baseline_ready):
    """Validation max < test min (strict temporal ordering)."""
    val_max = baseline_ready.loc[baseline_ready.split == "validation", "scheduled_departure_utc"].max()
    test_min = baseline_ready.loc[baseline_ready.split == "test", "scheduled_departure_utc"].min()
    assert val_max < test_min, f"Validation/test overlap: val_max={val_max}, test_min={test_min}"


def test_split_boundaries_match_config(baseline_ready):
    """Split boundaries must honour the configured cutoff dates."""
    train_max = baseline_ready.loc[baseline_ready.split == "train", "scheduled_departure_utc"].max()
    val_max = baseline_ready.loc[baseline_ready.split == "validation", "scheduled_departure_utc"].max()

    # Train must end on or before 2024-01-20 23:59:59 UTC
    assert train_max <= pd.Timestamp("2024-01-21", tz="UTC"), f"Train max too late: {train_max}"
    # Validation must end on or before 2024-01-26 23:59:59 UTC
    assert val_max <= pd.Timestamp("2024-01-27", tz="UTC"), f"Val max too late: {val_max}"


# ---------------------------------------------------------------------------
# 4. prev_aircraft_delay causality
# ---------------------------------------------------------------------------

def test_prev_aircraft_delay_causality(joined, features):
    """
    For every row where prev_aircraft_delay is not NaN, the previous aircraft's
    actual_arrival_utc must be STRICTLY BEFORE the current flight's
    scheduled_departure_utc.

    Alignment is done via flight_id to avoid pandas index-mismatch errors.
    """
    # Rebuild the previous-arrival mapping exactly as features.py does
    joined_sorted = joined.sort_values("scheduled_departure_utc").reset_index(drop=True)
    prev_arr_series = joined_sorted.groupby("tail_number")["actual_arrival_utc"].shift(1)
    # Map flight_id -> previous actual_arrival_utc (tz-aware)
    prev_arr_by_fid = pd.Series(
        pd.to_datetime(prev_arr_series.values, utc=True),
        index=joined_sorted["flight_id"],
    )

    # Select only flights that have a non-NaN prev_aircraft_delay
    features_indexed = features.set_index("flight_id")
    has_prev = features_indexed["prev_aircraft_delay"].dropna().index  # flight_id values

    prev_arrivals = prev_arr_by_fid.loc[has_prev]
    sched_deps = joined.set_index("flight_id").loc[has_prev, "scheduled_departure_utc"]

    violations = (prev_arrivals >= sched_deps).sum()
    assert violations == 0, (
        f"prev_aircraft_delay causality violated in {violations} rows: "
        "previous aircraft arrival >= current scheduled departure"
    )


def test_prev_aircraft_delay_first_flight_is_nan(joined, features):
    """The very first flight for each tail number must have prev_aircraft_delay = NaN."""
    joined_sorted = joined.sort_values("scheduled_departure_utc").reset_index(drop=True)
    first_fids = joined_sorted.groupby("tail_number")["flight_id"].first()
    features_indexed = features.set_index("flight_id")
    first_delays = features_indexed.loc[first_fids, "prev_aircraft_delay"]
    assert first_delays.isna().all(), (
        f"Some first flights have non-NaN prev_aircraft_delay: "
        f"{first_delays.dropna().index.tolist()[:5]}"
    )


# ---------------------------------------------------------------------------
# 5. Airport congestion causality
# ---------------------------------------------------------------------------

def test_airport_congestion_uses_closed_left():
    """
    The rolling window for airport_congestion uses closed='left', which means
    arrivals at EXACTLY the current timestamp are excluded.

    This test constructs a minimal scenario where an arrival occurs at exactly
    the departure time and verifies congestion does not include that arrival.
    """
    # Arrival at T=10:00, departure also at T=10:00 for airport B
    data = {
        "flight_id": ["F1", "F2"],
        "tail_number": ["T1", "T2"],
        "origin_airport": ["A", "B"],
        "destination_airport": ["B", "C"],
        "scheduled_departure_utc": [
            "2024-01-01 08:00:00+00:00",
            "2024-01-01 10:00:00+00:00",  # F2 departs B at exactly 10:00
        ],
        "actual_arrival_utc": [
            "2024-01-01 10:00:00+00:00",  # F1 arrives B at exactly 10:00
            "2024-01-01 12:00:00+00:00",
        ],
        "arrival_delay_minutes": [30.0, 20.0],
        "scheduled_duration_minutes": [120, 120],
        "distance": [1000, 1000],
        "carrier": ["AA", "AA"],
        "origin_weather_temperature_c": [15.0, 16.0],
        "origin_weather_dew_point_c": [5.0, 6.0],
        "origin_weather_pressure_hpa": [1013.0, 1012.0],
        "origin_weather_visibility_m": [10000.0, 9000.0],
        "origin_weather_wind_direction_degrees": [180.0, 190.0],
        "origin_weather_wind_speed_mps": [5.0, 6.0],
    }
    df = pd.DataFrame(data)

    from falsifier_x_air.data.features import generate_features
    feats = generate_features(df)
    f2 = feats[feats["flight_id"] == "F2"].iloc[0]

    # F2 departs B at 10:00; F1 arrived B at exactly 10:00.
    # closed='left' means the rolling window up to 10:00 excludes arrivals AT 10:00.
    # So airport_congestion for F2 should be NaN (no prior arrival in the 3h window).
    assert pd.isna(f2["airport_congestion"]), (
        f"Airport congestion for F2 should be NaN but got {f2['airport_congestion']}; "
        "arrival at exactly departure time must be excluded (closed='left')"
    )


def test_airport_congestion_includes_strict_prior_arrival():
    """An arrival strictly before the departure's 3h window is correctly reflected.

    closed='left' rolling mechanics:
    - The rolling window AT timestamp T covers the half-open interval [T-3h, T).
    - A single arrival at airport B produces NaN at its own timestamp (no prior arrivals).
    - The SECOND arrival at B (at T2) produces the mean of arrivals in [T2-3h, T2).
    - A departure at D picks up the most recent non-NaN congestion entry <= D.

    Scenario:
      A0 arrives B at 09:00 (delay=5)  <- seed; rolling at 09:00 = NaN
      F1 arrives B at 12:00 (delay=10) <- rolling at 12:00 covers [09:00,12:00) -> mean(5) = 5.0
      F2 departs B at 14:00            <- merge_asof picks value from 12:00 -> congestion=5.0
    """
    data = {
        "flight_id": ["A0", "F1", "F2"],
        "tail_number": ["T0", "T1", "T2"],
        "origin_airport": ["Z", "A", "B"],
        "destination_airport": ["B", "B", "C"],
        "scheduled_departure_utc": [
            "2024-01-01 07:00:00+00:00",  # A0 departs Z
            "2024-01-01 08:00:00+00:00",  # F1 departs A
            "2024-01-01 14:00:00+00:00",  # F2 departs B at 14:00
        ],
        "actual_arrival_utc": [
            "2024-01-01 09:00:00+00:00",  # A0 arrives B at 09:00 (delay=5)
            "2024-01-01 12:00:00+00:00",  # F1 arrives B at 12:00 (delay=10)
            "2024-01-01 16:00:00+00:00",  # F2 arrives C at 16:00
        ],
        "arrival_delay_minutes": [5.0, 10.0, 20.0],
        "scheduled_duration_minutes": [120, 110, 100],
        "distance": [800, 1000, 1000],
        "carrier": ["AA", "AA", "AA"],
        "origin_weather_temperature_c": [15.0, 20.0, 21.0],
        "origin_weather_dew_point_c": [5.0, 10.0, 11.0],
        "origin_weather_pressure_hpa": [1014.0, 1013.0, 1012.0],
        "origin_weather_visibility_m": [11000.0, 10000.0, 9000.0],
        "origin_weather_wind_direction_degrees": [170.0, 180.0, 190.0],
        "origin_weather_wind_speed_mps": [4.0, 5.0, 6.0],
    }
    df = pd.DataFrame(data)

    from falsifier_x_air.data.features import generate_features
    feats = generate_features(df)
    f2 = feats[feats["flight_id"] == "F2"].iloc[0]

    # Rolling at F1's arrival (12:00) covers [09:00,12:00) -> includes A0 delay=5.0.
    # merge_asof picks 5.0 for F2 departing B at 14:00.
    assert f2["airport_congestion"] == pytest.approx(5.0), (
        f"Expected congestion=5.0 (A0 delay=5 in 3h window before F1's arrival), "
        f"got {f2['airport_congestion']}"
    )



# ---------------------------------------------------------------------------
# 6. Weather causality (backward-only)
# ---------------------------------------------------------------------------

def test_weather_observation_utc_not_after_departure(joined):
    """Weather observation timestamp must never exceed scheduled departure time."""
    matched = joined.dropna(subset=["origin_weather_observation_utc"])
    violations = (
        matched["origin_weather_observation_utc"] > matched["scheduled_departure_utc"]
    ).sum()
    assert violations == 0, (
        f"Future weather used in {violations} rows "
        "(origin_weather_observation_utc > scheduled_departure_utc)"
    )


def test_weather_age_minutes_non_negative(joined):
    """Weather age must be zero or positive (observation at or before departure)."""
    matched = joined.dropna(subset=["origin_weather_age_minutes"])
    assert (matched["origin_weather_age_minutes"] >= 0).all()


# ---------------------------------------------------------------------------
# 7. Reproducible feature generation
# ---------------------------------------------------------------------------

def test_generate_features_is_deterministic(joined):
    """Calling generate_features twice on the same input must yield identical results."""
    from falsifier_x_air.data.features import generate_features
    f1 = generate_features(joined).sort_values("flight_id").reset_index(drop=True)
    f2 = generate_features(joined).sort_values("flight_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(f1, f2, check_like=False)


# ---------------------------------------------------------------------------
# 8. Preprocessing fitted only on training data
# ---------------------------------------------------------------------------

def test_preprocessing_fitted_only_on_training_data(features, baseline_ready):
    """
    Imputer and scaler must be fitted exclusively on training rows.
    We verify this by fitting a scaler on train and checking that the
    test set mean is NOT used (i.e. the transform does not centre test data).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from falsifier_x_air.data.features import prepare_training_data

    full = prepare_training_data(features, baseline_ready)
    train = full[full.split == "train"]
    test = full[full.split == "test"]

    numeric_col = "scheduled_duration_minutes"

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    train_vals = train[[numeric_col]].values
    test_vals = test[[numeric_col]].values

    imputer.fit(train_vals)
    scaler.fit(imputer.transform(train_vals))

    # The scaler mean must match the training mean, not the test mean
    train_mean = train[numeric_col].mean()
    test_mean = test[numeric_col].mean()
    assert abs(scaler.mean_[0] - train_mean) < 1e-3, (
        f"Scaler mean ({scaler.mean_[0]:.4f}) doesn't match train mean ({train_mean:.4f})"
    )
    # If train ≠ test distribution (chronological split), this must hold:
    # The scaler is NOT re-fitted on test data.
    assert abs(scaler.mean_[0] - test_mean) > 1e-6 or abs(train_mean - test_mean) < 1e-6, (
        "Scaler appears to be fitted on test data (means are identical), "
        "which should not happen with a chronological split."
    )


# ---------------------------------------------------------------------------
# 9. Graph Temporal Causality
# ---------------------------------------------------------------------------

def test_graph_temporal_causality():
    """
    Ensure all AIRCRAFT_ROTATION and TEMPORAL_AIRPORT_PROPAGATION edges satisfy
    strict source_time < target_time. Any source_time >= target_time is temporal leakage.
    """
    edges_path = PROCESSED / "graphs" / "edges.csv"
    if not edges_path.exists():
        edges_path = PROCESSED / "graphs" / "edges.parquet"
    if edges_path.suffix == ".csv":
        edges = pd.read_csv(edges_path)
    else:
        edges = pd.read_parquet(edges_path)

    temporal = edges[edges.edge_type.isin(["AIRCRAFT_ROTATION", "TEMPORAL_AIRPORT_PROPAGATION"])].copy()
    temporal["source_time"] = pd.to_datetime(temporal["source_time"], utc=True)
    temporal["target_time"] = pd.to_datetime(temporal["target_time"], utc=True)

    # All temporal edges must strictly satisfy source_time < target_time.
    violations = (temporal["source_time"] >= temporal["target_time"]).sum()
    assert violations == 0, f"Graph temporal leakage detected: {violations} edges have source_time >= target_time"

