import json
from types import SimpleNamespace
import numpy as np
import pytest
from emomm_sequence.training.sequence_selector import reference_paths


def test_cached_reference_cannot_silently_change_endpoints(tmp_path):
    directory = tmp_path / "cache" / "teachers"
    directory.mkdir(parents=True)
    problem = SimpleNamespace(ids=np.array(["a", "b", "c"]), n=3, start=0, target=2, cost=lambda path: .2)
    file = directory / "task_000001.json"
    file.write_text(json.dumps({"track_ids": ["a", "b", "c"], "paths": [[1, 0, 2]], "costs": [.2]}))
    with pytest.raises(ValueError, match="endpoint protocol"):
        reference_paths(problem, {"query_id": "task_000001"}, {}, tmp_path)
    file.write_text(json.dumps({"track_ids": ["a", "b", "c"], "paths": [[0, 1, 2]], "costs": [.2]}))
    assert reference_paths(problem, {"query_id": "task_000001"}, {}, tmp_path) == [[0, 1, 2]]


def test_cached_reference_cannot_silently_change_objective(tmp_path):
    directory = tmp_path / "cache" / "teachers"
    directory.mkdir(parents=True)
    problem = SimpleNamespace(ids=np.array(["a", "b", "c"]), n=3, start=0, target=2, cost=lambda path: .3)
    (directory / "task_000001.json").write_text(json.dumps({"track_ids": ["a", "b", "c"], "paths": [[0, 1, 2]], "costs": [.2]}))
    with pytest.raises(ValueError, match="objective changed"):
        reference_paths(problem, {"query_id": "task_000001"}, {}, tmp_path)
