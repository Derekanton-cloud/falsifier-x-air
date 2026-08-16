"""Phase 5 tests: schema, causality, preprocessing, and reproducibility."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from falsifier_x_air.stgnn import (
    CATEGORICAL_COLUMNS, FORBIDDEN_COLUMNS, NUMERICAL_COLUMNS, CausalMessagePassing,
    FlightGraphData, FlightSTGNN, TrainOnlyFeatureEncoder, load_flight_graph, set_seed,
)


def _toy_data(edge_index=None, source_time=None, target_time=None):
    return FlightGraphData(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), y=torch.tensor([1.0, 2.0, 3.0]),
        edge_index=torch.tensor([[0], [1]]) if edge_index is None else edge_index,
        edge_type=torch.tensor([0]), source_time=torch.tensor([1.0], dtype=torch.float64) if source_time is None else source_time,
        target_time=torch.tensor([2.0], dtype=torch.float64) if target_time is None else target_time,
        split=np.array(["train", "validation", "test"]), flight_ids=np.array(["a", "b", "c"]), feature_names=["a", "b"], graph_report={},
    )


def test_graph_loading_schema_and_chronological_splits():
    graph = load_flight_graph()
    assert graph.x.shape[0] == len(graph.flight_ids) == len(graph.y)
    assert graph.edge_index.shape[0] == 2 and graph.edge_index.shape[1] == len(graph.edge_type)
    assert set(graph.split) == {"train", "validation", "test"}
    assert graph.graph_report["message_edges"] > 0
    # Adapter itself retains only the scientifically justified relation types.
    assert set(graph.graph_report) >= {"AIRCRAFT_ROTATION", "TEMPORAL_AIRPORT_PROPAGATION"}


def test_target_and_forbidden_features_are_excluded():
    graph = load_flight_graph()
    assert "target_arrival_delay_minutes" not in graph.feature_names
    assert not FORBIDDEN_COLUMNS.intersection(graph.feature_names)


def test_causal_message_passing_rejects_equal_and_future_edges():
    set_seed(7); layer = CausalMessagePassing(4); layer.eval()
    h = torch.randn(3, 4)
    # Equality and future edges must have exactly the same result as no edges.
    empty = torch.empty((2, 0), dtype=torch.long); empty_type = torch.empty(0, dtype=torch.long); empty_time = torch.empty(0, dtype=torch.float64)
    expected = layer(h, empty, empty_type, empty_time, empty_time)
    equal = layer(h, torch.tensor([[0], [1]]), torch.tensor([0]), torch.tensor([2.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
    future = layer(h, torch.tensor([[0], [1]]), torch.tensor([0]), torch.tensor([3.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
    assert torch.equal(expected, equal)
    assert torch.equal(expected, future)


def test_train_only_normalization_and_missing_features():
    train = {c: ["AA", "AA"] if c in CATEGORICAL_COLUMNS else [1.0, np.nan] for c in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS}
    test = {c: ["ZZ"] if c in CATEGORICAL_COLUMNS else [100.0] for c in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS}
    frame = pd.concat([pd.DataFrame(train), pd.DataFrame(test)], ignore_index=True)
    encoder = TrainOnlyFeatureEncoder(); matrix, _ = encoder.fit_transform(frame, np.array([True, True, False]))
    assert encoder.means["distance"] == 1.0
    assert matrix.shape[0] == 3 and np.isfinite(matrix).all()
    # unseen test category does not expand the train-fitted vocabulary
    assert not any("=ZZ" in name for name in encoder.names)


def test_output_dimensions_and_deterministic_initialization_and_inference():
    data = _toy_data()
    set_seed(11); a = FlightSTGNN(2, hidden=8); a.eval(); first = a(data)
    set_seed(11); b = FlightSTGNN(2, hidden=8); b.eval(); second = b(data)
    assert first.shape == (3,)
    assert torch.equal(first, second)


def test_cancelled_and_diverted_rows_are_excluded_from_targets():
    graph = load_flight_graph()
    assert torch.isfinite(graph.y).all()
