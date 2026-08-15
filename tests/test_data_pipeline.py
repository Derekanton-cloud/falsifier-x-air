import pandas as pd
import pytest
import zipfile

from falsifier_x_air.data.flights import clean_bts_files, validate_bts_archive
from falsifier_x_air.data.graph_data import build_dynamic_edges, chronological_split
from falsifier_x_air.data.weather import align_weather, match_airports_to_stations, parse_noaa_csv
from falsifier_x_air.data.config import load_config
from falsifier_x_air.data.noaa_metadata import build_noaa_station_metadata, validate_noaa_stations


@pytest.fixture
def airports() -> pd.DataFrame:
    return pd.DataFrame({"airport_code": ["ORD", "JFK"], "latitude": [41.98, 40.64], "longitude": [-87.9, -73.78], "timezone": ["America/Chicago", "America/New_York"]}).set_index("airport_code", drop=False)


def test_bts_midnight_rollover_and_duplicate_removal(tmp_path, airports):
    raw = pd.DataFrame({"FlightDate": ["2024-01-01", "2024-01-01"], "Reporting_Airline": ["AA", "AA"], "Flight_Number_Reporting_Airline": ["1", "1"], "Origin": ["ORD", "ORD"], "Dest": ["JFK", "JFK"], "CRSDepTime": [2350, 2350], "CRSArrTime": [30, 30], "Cancelled": [0, 0], "Diverted": [0, 0], "CRSElapsedTime": [100, 100], "DepDelay": [0, 0], "ArrDelay": [5, 5], "Tail_Number": ["N1", "N1"]})
    path = tmp_path / "bts.csv"
    raw.to_csv(path, index=False)
    flights, report = clean_bts_files([path], airports, 10)
    assert len(flights) == 1 and report.duplicates == 1
    assert flights.loc[0, "scheduled_arrival_utc"] > flights.loc[0, "scheduled_departure_utc"]


def test_station_matching_is_radius_limited_and_deterministic(airports):
    stations = pd.DataFrame({"station_id": ["near", "far"], "latitude": [41.99, 10.0], "longitude": [-87.91, 10.0], "availability_score": [0.5, 1.0]})
    matches, report = match_airports_to_stations(airports.reset_index(drop=True).query("airport_code == 'ORD'"), stations, 20)
    assert matches.iloc[0].station_id == "near" and report.matched == 1


def test_station_matching_preserves_numeric_noaa_identifier_as_text(airports):
    stations = pd.DataFrame({"station_id": [72530094846], "latitude": [41.99], "longitude": [-87.91]})
    matches, _ = match_airports_to_stations(airports.reset_index(drop=True).query("airport_code == 'ORD'"), stations, 20)
    assert matches.iloc[0].station_id == "72530094846"


def test_weather_alignment_is_backward_only(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({"STATION": ["S1", "S1"], "DATE": ["2024-01-01T09:00:00Z", "2024-01-01T11:00:00Z"], "TMP": ["0010,1", "0020,1"], "DEW": ["0005,1", "0010,1"], "WND": ["090,1,N,0010,1", "090,1,N,0010,1"], "VIS": ["010000,1", "010000,1"], "SLP": ["10130,1", "10130,1"]}).to_csv(path, index=False)
    weather = parse_noaa_csv([path])
    flights = pd.DataFrame({"flight_id": ["F1"], "origin_airport": ["ORD"], "destination_airport": ["JFK"], "scheduled_departure_utc": pd.to_datetime(["2024-01-01T10:00:00Z"])})
    matches = pd.DataFrame({"airport_code": ["ORD"], "station_id": ["S1"]})
    joined = align_weather(flights, weather, matches, 90)
    assert joined.loc[0, "origin_weather_temperature_c"] == 1.0
    assert joined.loc[0, "origin_weather_observation_utc"] <= joined.loc[0, "scheduled_departure_utc"]
    assert joined.loc[0, "origin_weather_age_minutes"] == 60.0


def test_dynamic_edges_respect_time_and_ignore_missing_tail():
    flights = pd.DataFrame({"flight_id": ["F1", "F2", "F3"], "origin_airport": ["A", "B", "B"], "destination_airport": ["B", "C", "D"], "tail_number": ["N1", "N1", None], "scheduled_departure_utc": pd.to_datetime(["2024-01-01T08:00Z", "2024-01-01T10:00Z", "2024-01-01T11:00Z"]), "scheduled_arrival_utc": pd.to_datetime(["2024-01-01T09:00Z", "2024-01-01T11:00Z", "2024-01-01T12:00Z"]), "actual_arrival_utc": pd.to_datetime(["2024-01-01T09:00Z", "2024-01-01T11:00Z", "2024-01-01T12:00Z"])})
    edges = build_dynamic_edges(flights, 20, 240, 240)
    rotation = edges[edges.edge_type == "AIRCRAFT_ROTATION"]
    assert len(rotation) == 1 and (rotation.source_time < rotation.target_time).all()


def test_chronological_splits_do_not_overlap():
    flights = pd.DataFrame({"scheduled_departure_utc": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-22T00:00Z", "2024-01-28T00:00Z"])})
    split = chronological_split(flights, "2024-01-20", "2024-01-26")
    assert list(split.astype(str)) == ["train", "validation", "test"]


def test_invalid_configuration_boundaries_fail_clearly(tmp_path):
    config = tmp_path / "data.toml"
    config.write_text("""[dataset]\nschema_version='1'\nstart_date='2024-02-01'\nend_date='2024-01-01'\nmax_flights=1\nairports=[]\ncarriers=[]\n[paths]\nraw_bts='a'\nraw_noaa='b'\ninterim='c'\nprocessed='d'\nmetadata='e'\nairport_metadata='f'\nstation_metadata='g'\nmaster_coordinate='h'\n[sources]\nbts_prezip_url='x'\nnoaa_isd_base_url='y'\n[weather]\nstation_radius_km=1\nmatching_tolerance_minutes=1\n[graph]\nsnapshot_minutes=1\nminimum_rotation_minutes=1\nmaximum_rotation_minutes=2\nairport_propagation_minutes=1\n[splits]\ntrain_end='2024-01-01'\nvalidation_end='2024-01-02'\n""")
    with pytest.raises(ValueError, match="start_date"):
        load_config(config)


def test_bts_archive_validator_rejects_html_error_page(tmp_path):
    fake_download = tmp_path / "download.zip"
    fake_download.write_text("<html>not a BTS archive</html>")
    with pytest.raises(ValueError, match="not a ZIP"):
        validate_bts_archive(fake_download)


def test_bts_archive_with_readme_is_processed_as_its_csv_member(tmp_path, airports):
    source = tmp_path / "source.csv"
    pd.DataFrame({"FlightDate": ["2024-01-01"], "Reporting_Airline": ["AA"], "Flight_Number_Reporting_Airline": ["1"], "Origin": ["ORD"], "Dest": ["JFK"], "CRSDepTime": [900], "CRSArrTime": [1200], "Cancelled": [0], "Diverted": [0]}).to_csv(source, index=False)
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(source, "official.csv")
        handle.writestr("readme.html", "official archive documentation")
    flights, _ = clean_bts_files([archive], airports, 10)
    assert flights.flight_id.tolist() == ["2024-01-01_AA_1_ORD_900"]


def test_noaa_metadata_uses_only_valid_official_history_records(tmp_path):
    flights = tmp_path / "flights.csv"
    pd.DataFrame({"origin_airport": ["ORD"], "destination_airport": ["JFK"]}).to_csv(flights, index=False)
    airport_path = tmp_path / "airports.csv"
    pd.DataFrame({"airport_code": ["ORD", "JFK"], "latitude": [41.98, 40.64], "longitude": [-87.9, -73.78], "timezone": ["America/Chicago", "America/New_York"]}).to_csv(airport_path, index=False)
    history = tmp_path / "isd-history.csv"
    pd.DataFrame({"USAF": ["111111", "222222", "333333", "444444"], "WBAN": ["00001", "00002", "00003", "00004"], "LAT": [41.99, 40.65, 0.0, 40.65], "LON": [-87.91, -73.79, 0.0, -73.79], "BEGIN": [20200101, 20200101, 20200101, 20200101], "END": [20250101, 20231231, 20250101, 20250101]}).to_csv(history, index=False)
    stations, matches = build_noaa_station_metadata(flights, airport_path, history, tmp_path / "noaa_stations.csv", tmp_path / "matches.csv", tmp_path / "provenance.json", "2024-01-01", "2024-01-31", 75)
    assert stations.station_id.tolist() == ["11111100001", "44444400004"]
    assert matches.airport_code.tolist() == ["JFK", "ORD"]
    with pytest.raises(ValueError, match="schema"):
        validate_noaa_stations(stations.assign(extra=1))
