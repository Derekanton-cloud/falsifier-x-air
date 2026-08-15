"""BTS ingestion and canonical, timezone-aware flight records."""

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

import pandas as pd


REQUIRED_BTS_COLUMNS = frozenset({"FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "Cancelled", "Diverted"})
OPTIONAL_BTS_COLUMNS = ("Tail_Number", "OriginAirportID", "DestAirportID", "DepTime", "ArrTime", "DepDelay", "ArrDelay", "CRSElapsedTime", "Distance", "CancellationCode", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")


def validate_bts_archive(path: Path) -> dict[str, object]:
    """Validate a manually obtained official BTS CSV/ZIP before processing it."""
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"BTS input is missing or empty: {path}")
    if path.suffix.lower() == ".zip":
        if not zipfile.is_zipfile(path):
            raise ValueError(f"BTS ZIP validation failed (not a ZIP archive): {path}")
        with zipfile.ZipFile(path) as archive:
            csv_members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(f"BTS archive must contain exactly one CSV; found {len(csv_members)}")
            with archive.open(csv_members[0]) as handle:
                columns = set(pd.read_csv(handle, nrows=0).columns)
    elif path.suffix.lower() == ".csv":
        columns = set(pd.read_csv(path, nrows=0).columns)
    else:
        raise ValueError(f"Unsupported BTS input type: {path.suffix}; expected .zip or .csv")
    missing = REQUIRED_BTS_COLUMNS - columns
    if missing:
        raise ValueError(f"BTS input missing required columns: {sorted(missing)}")
    return {"path": str(path), "bytes": path.stat().st_size, "columns": sorted(columns)}


def _bts_csv_member(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"BTS archive must contain exactly one CSV; found {len(members)}")
    return members[0]


@dataclass(frozen=True)
class FlightQualityReport:
    rows_read: int
    rows_output: int
    duplicates: int
    invalid_timestamps: int
    cancelled: int
    diverted: int


def load_airports(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Airport metadata not found: {path}. Provide airport_code, latitude, longitude, timezone (IANA name).")
    airports = pd.read_csv(path, dtype={"airport_code": "string", "airport_id": "string"})
    required = {"airport_code", "latitude", "longitude", "timezone"}
    missing = required - set(airports.columns)
    if missing:
        raise ValueError(f"Airport metadata missing required columns: {sorted(missing)}")
    if airports[list(required)].isna().any().any() or airports["airport_code"].duplicated().any():
        raise ValueError("Airport metadata has missing required values or duplicate airport_code entries")
    return airports.drop_duplicates("airport_code").set_index("airport_code", drop=False)


def _local_timestamp(flight_dates: pd.Series, times: pd.Series, zones: pd.Series) -> pd.Series:
    """Parse BTS local HHMM values; invalid/missing times remain NaT."""
    time_text = pd.to_numeric(times, errors="coerce").astype("Int64").astype("string").str.zfill(4)
    base = pd.to_datetime(flight_dates.astype("string") + " " + time_text.str[:2] + ":" + time_text.str[2:], errors="coerce")
    values = []
    for timestamp, zone in zip(base, zones):
        if pd.isna(timestamp) or pd.isna(zone):
            values.append(pd.NaT)
        else:
            try:
                values.append(timestamp.tz_localize(ZoneInfo(str(zone)), ambiguous="NaT", nonexistent="NaT"))
            except (ValueError, KeyError):
                values.append(pd.NaT)
    return pd.Series(values, index=flight_dates.index, dtype="datetime64[ns, UTC]").dt.tz_convert("UTC")


def clean_bts_files(paths: list[Path], airports: pd.DataFrame, max_rows: int, airport_filter: tuple[str, ...] = (), carrier_filter: tuple[str, ...] = ()) -> tuple[pd.DataFrame, FlightQualityReport]:
    """Read BTS files in chunks, filter early, and retain raw local times plus UTC timestamps."""
    frames: list[pd.DataFrame] = []
    read_rows = 0
    for path in paths:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive, archive.open(_bts_csv_member(path)) as handle:
                header = pd.read_csv(handle, nrows=0)
        else:
            header = pd.read_csv(path, nrows=0)
        missing = REQUIRED_BTS_COLUMNS - set(header.columns)
        if missing:
            raise ValueError(f"{path.name} missing BTS columns: {sorted(missing)}")
        usecols = list(REQUIRED_BTS_COLUMNS | (set(OPTIONAL_BTS_COLUMNS) & set(header.columns)))
        if path.suffix.lower() == ".zip":
            archive = zipfile.ZipFile(path)
            handle = archive.open(_bts_csv_member(path))
        else:
            archive = None
            handle = path
        try:
            chunks = pd.read_csv(handle, usecols=usecols, chunksize=100_000, low_memory=False)
            for chunk in chunks:
                read_rows += len(chunk)
                if airport_filter:
                    chunk = chunk[chunk["Origin"].isin(airport_filter) | chunk["Dest"].isin(airport_filter)]
                if carrier_filter:
                    chunk = chunk[chunk["Reporting_Airline"].isin(carrier_filter)]
                frames.append(chunk)
                if sum(len(frame) for frame in frames) >= max_rows:
                    break
        finally:
            if archive is not None:
                handle.close()
                archive.close()
        if sum(len(frame) for frame in frames) >= max_rows:
            break
    raw = pd.concat(frames, ignore_index=True).head(max_rows) if frames else pd.DataFrame()
    if raw.empty:
        return raw, FlightQualityReport(read_rows, 0, 0, 0, 0, 0)
    raw["FlightDate"] = pd.to_datetime(raw["FlightDate"], errors="coerce").dt.date
    airport_lookup = airports.reset_index(drop=True)[["airport_code", "timezone"]]
    raw = raw.merge(airport_lookup, left_on="Origin", right_on="airport_code", how="left")
    raw = raw.rename(columns={"timezone": "origin_timezone"}).drop(columns="airport_code")
    raw = raw.merge(airport_lookup, left_on="Dest", right_on="airport_code", how="left")
    raw = raw.rename(columns={"timezone": "destination_timezone"}).drop(columns="airport_code")
    raw["scheduled_departure_utc"] = _local_timestamp(raw["FlightDate"], raw["CRSDepTime"], raw["origin_timezone"])
    raw["scheduled_arrival_utc"] = _local_timestamp(raw["FlightDate"], raw["CRSArrTime"], raw["destination_timezone"])
    def numeric(name: str) -> pd.Series:
        return pd.to_numeric(raw[name], errors="coerce") if name in raw else pd.Series(pd.NA, index=raw.index, dtype="Float64")
    duration = numeric("CRSElapsedTime")
    arrival_before_departure = raw["scheduled_arrival_utc"] < raw["scheduled_departure_utc"]
    raw.loc[arrival_before_departure & duration.notna(), "scheduled_arrival_utc"] = raw.loc[arrival_before_departure & duration.notna(), "scheduled_departure_utc"] + pd.to_timedelta(duration[arrival_before_departure & duration.notna()], unit="m")
    raw["actual_departure_utc"] = raw["scheduled_departure_utc"] + pd.to_timedelta(numeric("DepDelay"), unit="m")
    raw["actual_arrival_utc"] = raw["scheduled_arrival_utc"] + pd.to_timedelta(numeric("ArrDelay"), unit="m")
    raw["flight_id"] = raw["FlightDate"].astype("string") + "_" + raw["Reporting_Airline"].astype("string") + "_" + raw["Flight_Number_Reporting_Airline"].astype("string") + "_" + raw["Origin"].astype("string") + "_" + raw["CRSDepTime"].astype("string")
    duplicates = int(raw.duplicated("flight_id").sum())
    invalid = int(raw["scheduled_departure_utc"].isna().sum())
    canonical = pd.DataFrame({
        "flight_id": raw["flight_id"], "flight_date": raw["FlightDate"], "carrier": raw["Reporting_Airline"], "flight_number": raw["Flight_Number_Reporting_Airline"], "tail_number": raw.get("Tail_Number", pd.Series(pd.NA, index=raw.index)),
        "origin_airport": raw["Origin"], "destination_airport": raw["Dest"], "origin_timezone": raw["origin_timezone"], "destination_timezone": raw["destination_timezone"], "origin_airport_id": raw.get("OriginAirportID", pd.Series(pd.NA, index=raw.index)), "destination_airport_id": raw.get("DestAirportID", pd.Series(pd.NA, index=raw.index)),
        "scheduled_departure_local": raw["CRSDepTime"], "scheduled_arrival_local": raw["CRSArrTime"], "actual_departure_local": raw.get("DepTime", pd.Series(pd.NA, index=raw.index)), "actual_arrival_local": raw.get("ArrTime", pd.Series(pd.NA, index=raw.index)), "scheduled_departure_utc": raw["scheduled_departure_utc"], "scheduled_arrival_utc": raw["scheduled_arrival_utc"],
        "actual_departure_utc": raw["actual_departure_utc"], "actual_arrival_utc": raw["actual_arrival_utc"], "scheduled_duration_minutes": duration,
        "departure_delay_minutes": numeric("DepDelay"), "arrival_delay_minutes": numeric("ArrDelay"),
        "cancelled": raw["Cancelled"].astype(bool), "diverted": raw["Diverted"].astype(bool), "cancellation_code": raw.get("CancellationCode", pd.Series(pd.NA, index=raw.index)), "distance": numeric("Distance"),
        "carrier_delay": numeric("CarrierDelay"), "weather_delay": numeric("WeatherDelay"), "nas_delay": numeric("NASDelay"), "security_delay": numeric("SecurityDelay"), "late_aircraft_delay": numeric("LateAircraftDelay"),
    }).drop_duplicates("flight_id")
    canonical = canonical[canonical["scheduled_departure_utc"].notna()].sort_values("scheduled_departure_utc").reset_index(drop=True)
    return canonical, FlightQualityReport(read_rows, len(canonical), duplicates, invalid, int(canonical.cancelled.sum()), int(canonical.diverted.sum()))
