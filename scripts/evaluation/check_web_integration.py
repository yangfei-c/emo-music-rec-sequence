"""Exercise the real second-page API without writing simulated human feedback."""
import argparse
import json
import sys
import socket
import threading
import time
from pathlib import Path

import torch
import requests
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "EmoMM-Web/backend"))
from emomm_sequence.inference.runtime import EmoMMSequenceRuntime
from emomm_sequence.training.records import write_json
from app.main import create_app
from app import dependencies, routes


class LocalClient:
    """Real local HTTP avoids dependence on a mismatched test-client package."""
    def __init__(self, app):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="off"))

    def __enter__(self):
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("Local test server did not start")
            time.sleep(.05)
        return self

    def get(self, path, **kwargs):
        return requests.get(f"http://127.0.0.1:{self.port}{path}", timeout=120, **kwargs)

    def post(self, path, **kwargs):
        return requests.post(f"http://127.0.0.1:{self.port}{path}", timeout=120, **kwargs)

    def __exit__(self, *args):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/system_route2.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/route2_uncertainty_20260926/evaluation")
    parser.add_argument("--check-first-work", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    runtime = EmoMMSequenceRuntime(ROOT, config_path=args.config)
    app = create_app()
    app.dependency_overrides.clear()
    routes.get_sequence_runtime = lambda: runtime
    events = []
    routes.record_event = lambda kind, payload: events.append({"kind": kind, **payload})
    dependencies.record_event = lambda kind, payload: events.append({"kind": kind, **payload})
    with LocalClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/strategies").status_code == 200
        first_work_status = None
        if args.check_first_work:
            first = client.post("/recommend", json={"text": "我现在有点低落。", "strategy": "comfort", "strength": 0.5})
            if first.status_code != 200:
                raise RuntimeError("First-work regression: " + first.text)
            first_work_status = first.status_code
        assert client.post("/sequence/recommend", json={"request_text": "我很烦", "method": "removed_method"}).status_code == 422
        response = client.post("/sequence/recommend", json={"request_text": "我现在很烦，想慢慢平静下来，最好听钢琴音乐。", "method": "ours", "scope": "unseen_library"})
        if response.status_code != 200:
            raise RuntimeError(response.text)
        value = response.json()
        assert len(value["tracks"]) == 5
        track = value["tracks"][0]["track_id"]
        audio = client.get(f"/media/sequence/{track}", headers={"Range": "bytes=0-1023"})
        assert audio.status_code in {200, 206} and len(audio.content) > 0
        payload = {"session_id": value["session_id"], "song_id": track, "play_duration_seconds": 12, "recommendation_mode": "sequence", "sequence_method": "ours", "step_index": 1, "sequence_length": 5}
        assert client.post("/playback", json=payload).status_code == 200
        assert client.post("/feedback", json={**payload, "music_preference": 3, "regulation_effect": 3}).status_code == 200
        assert all(event["model_version"] == runtime.system.config["run_id"] for event in events)
        assert all(event["retriever_checkpoint_sha256"] == value["retriever_checkpoint_sha256"] for event in events)
        assert all(event["text_va_checkpoint_sha256"] == value["text_va_checkpoint_sha256"] for event in events)
        assert all(event["system_config_sha256"] == value["system_config_sha256"] for event in events)
    report = {"status": "passed", "routes": ["/health", "/strategies", "/sequence/recommend", "/media/sequence/{track_id}", "/playback", "/feedback"], "invalid_old_method_rejected": True, "audio_http_status": audio.status_code, "track_count": len(value["tracks"]), "model_version": value["model_version"], "simulated_feedback_written_to_real_log": False, "captured_test_events": len(events)}
    report["first_work_http_status"] = first_work_status
    write_json(args.output / "web_integration.json", report)
    write_json(args.output / "web_example.json", value)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
