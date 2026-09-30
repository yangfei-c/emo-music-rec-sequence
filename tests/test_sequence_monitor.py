"""Behavioral checks for the read-only experiment monitor."""
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone
import os

import psutil

_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluation/monitor_sequence_routes.py"
_SPEC = importlib.util.spec_from_file_location("sequence_monitor", _PATH)
monitor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(monitor)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pid_requires_matching_creation_timestamp():
    created = datetime.fromtimestamp(psutil.Process().create_time(), timezone.utc).isoformat()
    assert monitor.process_identity(os.getpid(), created)["alive"]
    assert not monitor.process_identity(os.getpid(), "2000-01-01T00:00:00+00:00")["alive"]
    assert monitor.process_identity(os.getpid(), None)["verification"] == "missing_creation_time"


def test_missing_tasks_never_count_as_complete(tmp_path):
    result = monitor.check_once(tmp_path, generate_report=False)
    assert result["state"] == "waiting_for_pipeline"
    assert not result["whole_blueprint_complete"]


def test_completed_pipeline_requires_successful_steps_and_files(tmp_path):
    progress = tmp_path / "pipeline/progress.json"
    save(progress, {"status": "completed", "scope": "pilot", "steps": []})
    assert monitor.check_once(tmp_path, generate_report=False)["state"] == "completion_unverified"
    save(progress, {"status": "completed", "scope": "pilot", "steps": [
        {"exit_code": 0, "expected_outputs": ["result.json"]}]})
    assert monitor.check_once(tmp_path, generate_report=False)["state"] == "completion_unverified"
    save(tmp_path / "result.json", {"result": 1})
    result = monitor.check_once(tmp_path, generate_report=False)
    assert result["state"] == "completed_verified"
    assert result["scope"] == "pilot"
    assert not result["whole_blueprint_complete"]


def test_dead_process_overrides_running_progress_and_notification_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "process_identity", lambda *args: {"alive": False, "verification": "not_running"})
    save(tmp_path / "pipeline/progress.json", {"status": "running", "pid": 123, "created_utc": "2026-01-01T00:00:00Z"})
    first = monitor.check_once(tmp_path, generate_report=False)
    second = monitor.check_once(tmp_path, generate_report=False)
    assert first["state"] == second["state"] == "stopped_unexpectedly"
    events = (tmp_path / "monitoring/notifications.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    assert len((tmp_path / "monitoring/checks.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_followups_prevent_early_completion_and_missing_planned_steps(tmp_path):
    save(tmp_path / "result.json", {})
    primary = {"status": "completed", "scope": "pilot", "steps": [{"name": "a", "exit_code": 0}],
               "expected_outputs": ["result.json"]}
    save(tmp_path / "pipeline/progress.json", primary)
    created = datetime.fromtimestamp(psutil.Process().create_time(), timezone.utc).isoformat()
    save(tmp_path / "followups/progress.json", {"status": "waiting_for_primary", "scope": "pilot",
                                               "pid": os.getpid(), "created_utc": created})
    assert monitor.check_once(tmp_path, generate_report=False)["state"] == "running"
    save(tmp_path / "followups/progress.json", {**primary, "planned_steps": ["a", "b"]})
    assert monitor.check_once(tmp_path, generate_report=False)["state"] == "completion_unverified"
