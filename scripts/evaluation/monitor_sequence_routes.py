"""Read-only, identity-checked monitoring of the two sequence experiment routes.

Progress files are claims, not process evidence. A running record is considered
alive only when both its PID and OS process creation timestamp match. No job is
ever restarted or killed. Notifications are local JSONL events, never messages.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLETE = {"completed", "complete", "success", "succeeded", "finished"}
FAILED = {"failed", "error", "crashed", "timeout", "timed_out", "incomplete_timeout", "cancelled"}
PENDING = {"pending", "queued", "planned", "not_started", "waiting"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # Do not assume a local timestamp is UTC: identity must be explicit.
        if parsed.tzinfo is None:
            return None
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("Progress must be a JSON object")
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def process_identity(pid: Any, created_utc: Any) -> dict[str, Any]:
    """Verify process identity from the OS, never from a heartbeat file."""
    result: dict[str, Any] = {"pid": pid, "claimed_created_utc": created_utc, "alive": False}
    try:
        numeric_pid = int(pid)
        if numeric_pid <= 0:
            raise ValueError("PID must be positive")
    except (ValueError, TypeError):
        return {**result, "verification": "missing_pid"}
    try:
        import psutil
    except ImportError:
        return {**result, "verification": "unverified", "error": "psutil is unavailable"}
    try:
        process = psutil.Process(numeric_pid)
        actual_created = process.create_time()
        result["actual_created_utc"] = datetime.fromtimestamp(actual_created, timezone.utc).isoformat()
        expected = parse_time(created_utc)
        if expected is None:
            return {**result, "verification": "missing_creation_time"}
        # JSON writers may round to whole seconds; PID reuse differs much more.
        if abs(actual_created - expected) > 1.0:
            return {**result, "verification": "identity_mismatch"}
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return {**result, "verification": "not_running"}
        return {**result, "verification": "verified", "alive": True}
    except psutil.NoSuchProcess:
        return {**result, "verification": "not_running"}
    except (psutil.AccessDenied, OSError) as error:
        return {**result, "verification": "unverified", "error": str(error)}


def resolve_output(value: Any, root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    # Repository-relative artifact paths and experiment-root-relative paths are
    # both accepted; no glob expansion or commands are evaluated.
    if path.parts and path.parts[0] in {"artifacts", "data", "configs"}:
        return REPO_ROOT / path
    return root / path


def completion_evidence(record: dict[str, Any], root: Path) -> dict[str, Any]:
    steps = record.get("steps")
    reasons: list[str] = []
    outputs: list[str] = []
    if not isinstance(steps, list) or not steps:
        reasons.append("No completed step manifest was declared")
        steps = []
    if record.get("planned_steps"):
        observed = {step.get("name") for step in steps if isinstance(step, dict) and step.get("exit_code") == 0}
        if set(record["planned_steps"]) - observed:
            reasons.append("One or more planned steps lack a successful completion record")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            reasons.append(f"Step {index} is not an object")
            continue
        if step.get("exit_code") != 0:
            reasons.append(f"Step {index} lacks exit_code=0")
        listed = step.get("expected_outputs", [])
        if isinstance(listed, list):
            outputs.extend(str(item) for item in listed)
        else:
            reasons.append(f"Step {index} expected_outputs is not a list")
    listed = record.get("expected_outputs", [])
    if isinstance(listed, list):
        outputs.extend(str(item) for item in listed)
    else:
        reasons.append("Pipeline expected_outputs is not a list")
    if not outputs:
        reasons.append("No expected output artifacts were declared")
    missing = [str(resolve_output(value, root)) for value in dict.fromkeys(outputs)
               if not resolve_output(value, root).is_file()]
    if missing:
        reasons.append("One or more expected output files are missing")
    if record.get("scope") not in {"pilot", "full"}:
        reasons.append("Pipeline scope must explicitly be pilot or full")
    return {"verified": not reasons, "reasons": reasons, "missing_outputs": missing,
            "expected_output_count": len(set(outputs)), "step_count": len(steps),
            "scope": record.get("scope"), "whole_blueprint_complete": False}


def inspect_record(path: Path, root: Path, now: float, stale_seconds: float) -> dict[str, Any]:
    item: dict[str, Any] = {"path": path.relative_to(root).as_posix()}
    try:
        record = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {**item, "state": "unreadable", "error": str(error)}
    declared = str(record.get("status", "unknown")).lower()
    process = process_identity(record.get("pid"), record.get("created_utc"))
    item.update({"declared_status": declared, "process": process,
                 "scope": record.get("scope"), "active_command": record.get("active_command"),
                 "progress": {key: record[key] for key in ("epoch", "episodes", "episode", "stage", "phase") if key in record}})
    updated = parse_time(record.get("updated_utc") or record.get("updated_at")) or path.stat().st_mtime
    item["seconds_since_update"] = max(0.0, now - updated)
    if declared == "incomplete_timeout":
        item["state"] = "incomplete_resumable"
    elif declared in FAILED:
        item["state"] = "failed"
        item["error"] = record.get("error") or record.get("message")
    elif declared in COMPLETE:
        if item["path"] in {"pipeline/progress.json", "followups/progress.json"}:
            evidence = completion_evidence(record, root)
            item["completion"] = evidence
            item["state"] = "completed_verified" if evidence["verified"] else "completion_unverified"
        else:
            item["state"] = "completed_declared"
    elif process["alive"]:
        item["state"] = "stalled" if item["seconds_since_update"] > stale_seconds else "running"
    elif declared in PENDING and process["verification"] == "missing_pid":
        item["state"] = "not_started"
    elif process["verification"] == "not_running":
        item["state"] = "stopped_unexpectedly"
    else:
        item["state"] = process["verification"]
    limit = record.get("max_run_seconds") or record.get("timeout_seconds")
    start = parse_time(record.get("started_utc") or record.get("created_utc"))
    if isinstance(limit, (int, float)) and limit > 0 and start is not None and process["alive"]:
        item["timeout_advisory"] = now - start > limit
    return item


def state_signature(snapshot: dict[str, Any]) -> str:
    # Epochs and timestamps do not generate notifications. A changed command is
    # a phase transition and therefore useful even while the pipeline is alive.
    state = {"state": snapshot["state"], "scope": snapshot.get("scope"),
             "records": [{"path": item["path"], "state": item["state"],
                          "active_command": item.get("active_command"),
                          "timeout_advisory": item.get("timeout_advisory", False)}
                         for item in snapshot["records"]]}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def maybe_report(snapshot: dict[str, Any], root: Path, monitoring: Path) -> None:
    if snapshot["state"] != "completed_verified" or snapshot.get("scope") not in {"pilot", "full"}:
        return
    script = REPO_ROOT / "scripts/evaluation/report_sequence_routes.py"
    if not script.is_file():
        snapshot["report"] = {"state": "waiting_for_report_script"}
        return
    profile = snapshot["scope"]
    expected = [root / "reports" / profile / name for name in ("summary.json", "results.csv", "实验结果.md")]
    if all(path.is_file() for path in expected):
        snapshot["report"] = {"state": "exists", "paths": [str(path) for path in expected]}
        return
    attempts_path = monitoring / "report_attempt.json"
    previous = read_json(attempts_path) if attempts_path.is_file() else {}
    signature = snapshot["state_signature"]
    if previous.get("completion_signature") == signature:
        snapshot["report"] = previous
        return
    command = [sys.executable, str(script), "--root", str(root), "--profile", profile]
    attempt = {"completion_signature": signature, "started_utc": utc_now(), "command": command,
               "state": "started", "automatic_retry": False}
    write_json(attempts_path, attempt)
    with (monitoring / f"report_{profile}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    attempt.update({"finished_utc": utc_now(), "exit_code": result.returncode,
                    "state": "completed" if result.returncode == 0 and all(path.is_file() for path in expected) else "failed"})
    write_json(attempts_path, attempt)
    snapshot["report"] = attempt


@contextlib.contextmanager
def check_lock(path: Path):
    """Prevent a scheduled check and a manual check from writing concurrently."""
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            yield
        finally:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_once(root: Path, stale_seconds: float = 5400, generate_report: bool = True) -> dict[str, Any]:
    root = root.resolve()
    monitoring = root / "monitoring"
    monitoring.mkdir(parents=True, exist_ok=True)
    with check_lock(monitoring / "check.lock"):
        records = [inspect_record(path, root, time.time(), stale_seconds)
                   for path in sorted(root.rglob("progress.json")) if monitoring not in path.parents]
        pipeline = next((item for item in records if item["path"] == "pipeline/progress.json"), None)
        followups = next((item for item in records if item["path"] == "followups/progress.json"), None)
        active_scope = followups or pipeline
        snapshot: dict[str, Any] = {"schema_version": 1, "checked_utc": utc_now(), "root": str(root),
                                    "monitor_pid": os.getpid(), "records": records,
                                    "state": active_scope["state"] if active_scope else "waiting_for_pipeline",
                                    "scope": active_scope.get("scope") if active_scope else None,
                                    "observed_controller": active_scope["path"] if active_scope else None,
                                    "whole_blueprint_complete": False,
                                    "policy": {"restart": False, "kill": False, "external_notifications": False}}
        snapshot["state_signature"] = state_signature(snapshot)
        latest = monitoring / "latest.json"
        try:
            previous = read_json(latest) if latest.is_file() else {}
        except (OSError, ValueError):
            previous = {}
        if generate_report:
            maybe_report(snapshot, root, monitoring)
        append_json(monitoring / "checks.jsonl", snapshot)
        if previous.get("state_signature") != snapshot["state_signature"]:
            append_json(monitoring / "notifications.jsonl", {
                "created_utc": utc_now(), "event": "state_changed", "previous_state": previous.get("state"),
                "state": snapshot["state"], "scope": snapshot["scope"],
                "state_signature": snapshot["state_signature"], "records": records,
                "report": snapshot.get("report"), "delivery": "local_file_only"})
        write_json(latest, snapshot)
        return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/sequence_routes_v1")
    parser.add_argument("--interval-seconds", type=float, default=1800)
    parser.add_argument("--stale-seconds", type=float, default=5400)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.stale_seconds <= 0:
        parser.error("Intervals must be positive")
    if args.once:
        result = check_once(args.root, args.stale_seconds, not args.no_report)
        print(json.dumps({"state": result["state"], "scope": result["scope"], "records": len(result["records"])}, ensure_ascii=False))
        return
    import psutil
    watcher = args.root.resolve() / "monitoring/watcher.json"
    watcher.parent.mkdir(parents=True, exist_ok=True)
    if watcher.is_file():
        existing = read_json(watcher)
        if process_identity(existing.get("pid"), existing.get("created_utc"))["alive"]:
            raise SystemExit("A verified watcher already exists; no duplicate was started")
    metadata = {"status": "running", "pid": os.getpid(),
                "created_utc": datetime.fromtimestamp(psutil.Process().create_time(), timezone.utc).isoformat(),
                "interval_seconds": args.interval_seconds, "mechanism": "local_hidden_watcher",
                "persistence": "Only while this process and the current Windows session remain alive"}
    write_json(watcher, metadata)
    try:
        while True:
            started = time.monotonic()
            try:
                check_once(args.root, args.stale_seconds, not args.no_report)
            except Exception as error:
                # Retrying observation is permitted; experiments themselves are
                # never retried or modified by this monitor.
                append_json(watcher.parent / "monitor_errors.jsonl", {"created_utc": utc_now(), "error": repr(error)})
            time.sleep(max(0.1, args.interval_seconds - (time.monotonic() - started)))
    except KeyboardInterrupt:
        metadata.update({"status": "stopped", "stopped_utc": utc_now()})
        write_json(watcher, metadata)


if __name__ == "__main__":
    main()
