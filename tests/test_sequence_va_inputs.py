import json
import numpy as np
from emomm_sequence.data.sequence_tasks import Bank, make_task_problem
from emomm_sequence.data.sequence_va_tasks import VAInputs


def test_va_validation_cannot_select_calibration_songs_and_mode_keeps_endpoints(tmp_path):
    cache = tmp_path / "cache" / "va_extension"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(2)
    ids = np.array([f"d{i}" for i in range(7)])
    bank = Bank("deam", "validation", ids, ids, rng.normal(size=(7, 8)).astype("float32"))
    mean = np.linspace(.2, .8, 7)[:, None] * np.ones((1, 2))
    quantiles = np.stack((mean - .05, mean, mean + .05), axis=-1)
    np.savez(cache / "deam_validation.npz", track_ids=ids, quantiles=quantiles,
             cqr_lower=mean - .1, cqr_upper=mean + .1,
             preference_text_ids=np.array([], dtype=int), preference_distances=np.zeros((7, 0)))
    np.savez(cache / "text.npz", predicted_va=np.array([[.2, .2], [.8, .8]]))
    (cache / "manifest.json").write_text(json.dumps({"validation_eligibility": {"deam": {"allowed_bank_indices": [1, 3, 5]}}}))
    texts = {"embeddings": rng.normal(size=(2, 8)).astype("float32")}
    task = {"source": "deam", "split": "validation", "indices": list(range(7)), "start_text_id": 0,
            "target_text_id": 1, "preference_text_id": None, "num_tracks": 3, "budget": 6}
    ends = []
    for mode in ("point", "raw_interval", "cqr_interval"):
        selected, problem, actual = VAInputs(tmp_path, mode).make(bank, task, texts)
        assert selected.ids.tolist() == ["d1", "d3", "d5"]
        assert actual["budget"] == 3
        assert problem.x.shape == (3, 15)
        assert problem.query.shape == (3, 15)
        _, frozen = make_task_problem(bank, actual, texts)
        assert (problem.start, problem.target) == (frozen.start, frozen.target)
        ends.append((problem.start, problem.target))
    assert len(set(ends)) == 1
