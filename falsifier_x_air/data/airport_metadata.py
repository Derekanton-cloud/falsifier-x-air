"""Historical BTS Master Coordinate selection and coordinate-to-IANA mapping."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import pandas as pd
from timezonefinder import TimezoneFinder


STUDY_DATE = pd.Timestamp("2024-01-15")


def _csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Master Coordinate archive must contain exactly one CSV; found {len(members)}")
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle)


def select_historical_airports(master: pd.DataFrame, observed_codes: set[str], study_date: pd.Timestamp = STUDY_DATE) -> pd.DataFrame:
    required = {"AIRPORT", "LATITUDE", "LONGITUDE", "AIRPORT_START_DATE", "AIRPORT_THRU_DATE", "AIRPORT_IS_CLOSED", "AIRPORT_IS_LATEST", "AIRPORT_SEQ_ID"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master Coordinate data missing fields: {sorted(missing)}")
    candidates = master[master["AIRPORT"].isin(observed_codes)].copy()
    candidates["AIRPORT_START_DATE"] = pd.to_datetime(candidates["AIRPORT_START_DATE"], errors="coerce")
    candidates["AIRPORT_THRU_DATE"] = pd.to_datetime(candidates["AIRPORT_THRU_DATE"], errors="coerce")
    valid = candidates[
        (candidates["AIRPORT_START_DATE"] <= study_date)
        & (candidates["AIRPORT_THRU_DATE"].isna() | (candidates["AIRPORT_THRU_DATE"] >= study_date))
        & candidates["AIRPORT_IS_CLOSED"].fillna(0).eq(0)
    ].copy()
    selected = valid.sort_values(["AIRPORT", "AIRPORT_IS_LATEST", "AIRPORT_START_DATE", "AIRPORT_SEQ_ID"], ascending=[True, False, False, False]).drop_duplicates("AIRPORT")
    unresolved = observed_codes - set(selected["AIRPORT"])
    if unresolved:
        raise ValueError(f"No open Master Coordinate record valid on {study_date.date()} for airports: {sorted(unresolved)}")
    return selected


def validate_airports(frame: pd.DataFrame, observed_codes: set[str]) -> None:
    required = {"airport_code", "latitude", "longitude", "timezone"}
    if set(frame.columns) != required:
        raise ValueError(f"airports.csv schema must be exactly {sorted(required)}")
    if frame["airport_code"].isna().any() or frame["airport_code"].duplicated().any():
        raise ValueError("Airport codes must be non-null and unique")
    if set(frame.airport_code) != observed_codes:
        raise ValueError("Observed BTS airport codes and airport metadata codes differ")
    if frame[["latitude", "longitude", "timezone"]].isna().any().any():
        raise ValueError("Airport metadata contains missing coordinates or timezone")
    if not frame.latitude.between(-90, 90).all() or not frame.longitude.between(-180, 180).all():
        raise ValueError("Airport metadata contains invalid coordinates")
    for zone in frame.timezone:
        try:
            ZoneInfo(zone)
        except Exception as error:
            raise ValueError(f"Invalid IANA timezone: {zone}") from error


def build_airport_metadata(bts_codes: set[str], master_path: Path, output_path: Path, provenance_path: Path) -> pd.DataFrame:
    master = _csv_from_zip(master_path)
    selected = select_historical_airports(master, bts_codes)
    finder = TimezoneFinder()
    output = pd.DataFrame({"airport_code": selected["AIRPORT"], "latitude": pd.to_numeric(selected["LATITUDE"]), "longitude": pd.to_numeric(selected["LONGITUDE"])})
    output["timezone"] = [finder.timezone_at(lng=longitude, lat=latitude) for latitude, longitude in zip(output.latitude, output.longitude)]
    validate_airports(output, bts_codes)
    output = output.sort_values("airport_code").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    provenance = {"study_period": "2024-01", "airport_count_observed": len(bts_codes), "airport_count_matched": len(output),
                  "coordinate_source": "BTS TranStats Master Coordinate", "coordinate_source_file": str(master_path),
                  "coordinate_source_url": "https://www.transtats.bts.gov/tables.asp?QO_VQ=IMI&QO_anzr=N8vn6v10",
                  "timezone_source": "IANA tz database via timezonefinder", "timezone_source_url": "https://www.iana.org/time-zones",
                  "timezonefinder_version": version("timezonefinder"), "selection_rules": ["AirportStartDate <= 2024-01-15", "AirportThruDate is null or >= 2024-01-15", "AirportIsClosed == 0", "prefer AirportIsLatest, then newest AirportStartDate, then greatest AirportSeqID"],
                  "unmatched_airports": [], "timezone_failures": [], "generated_at": datetime.now(timezone.utc).isoformat()}
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output
