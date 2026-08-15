"""Configuration contracts for the real-data pipeline."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataPipelineConfig:
    root: Path
    start_date: date
    end_date: date
    max_flights: int
    airports: tuple[str, ...]
    carriers: tuple[str, ...]
    raw_bts: Path
    raw_noaa: Path
    interim: Path
    processed: Path
    metadata: Path
    airport_metadata: Path
    station_metadata: Path
    master_coordinate: Path
    bts_prezip_url: str
    noaa_isd_base_url: str
    station_radius_km: float
    matching_tolerance_minutes: int
    snapshot_minutes: int
    minimum_rotation_minutes: int
    maximum_rotation_minutes: int
    airport_propagation_minutes: int
    train_end: date
    validation_end: date
    schema_version: str

    def ensure_directories(self) -> None:
        for path in (self.raw_bts, self.raw_noaa, self.interim, self.processed, self.metadata):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("dataset.start_date must be on or before dataset.end_date")
        if not self.start_date <= self.train_end < self.validation_end <= self.end_date:
            raise ValueError("Split boundaries must satisfy start_date <= train_end < validation_end <= end_date")
        if self.max_flights <= 0 or self.station_radius_km <= 0 or self.matching_tolerance_minutes < 0:
            raise ValueError("max_flights and station_radius_km must be positive; matching tolerance cannot be negative")
        if not 0 <= self.minimum_rotation_minutes <= self.maximum_rotation_minutes:
            raise ValueError("Rotation windows must satisfy 0 <= minimum_rotation_minutes <= maximum_rotation_minutes")
        if self.airport_propagation_minutes <= 0 or self.snapshot_minutes <= 0:
            raise ValueError("Graph window sizes must be positive")


def load_config(path: Path | str = "configs/data.toml") -> DataPipelineConfig:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = path.parent.parent
    dataset, paths, sources, weather, graph, splits = (raw[name] for name in ("dataset", "paths", "sources", "weather", "graph", "splits"))
    config = DataPipelineConfig(
        root=root, start_date=date.fromisoformat(dataset["start_date"]), end_date=date.fromisoformat(dataset["end_date"]),
        max_flights=int(dataset["max_flights"]), airports=tuple(dataset["airports"]), carriers=tuple(dataset["carriers"]),
        raw_bts=root / paths["raw_bts"], raw_noaa=root / paths["raw_noaa"], interim=root / paths["interim"],
        processed=root / paths["processed"], metadata=root / paths["metadata"], airport_metadata=root / paths["airport_metadata"],
        station_metadata=root / paths["station_metadata"], master_coordinate=root / paths["master_coordinate"], bts_prezip_url=sources["bts_prezip_url"], noaa_isd_base_url=sources["noaa_isd_base_url"],
        station_radius_km=float(weather["station_radius_km"]), matching_tolerance_minutes=int(weather["matching_tolerance_minutes"]),
        snapshot_minutes=int(graph["snapshot_minutes"]), minimum_rotation_minutes=int(graph["minimum_rotation_minutes"]),
        maximum_rotation_minutes=int(graph["maximum_rotation_minutes"]), airport_propagation_minutes=int(graph["airport_propagation_minutes"]),
        train_end=date.fromisoformat(splits["train_end"]), validation_end=date.fromisoformat(splits["validation_end"]), schema_version=dataset["schema_version"],
    )
    config.validate()
    return config
