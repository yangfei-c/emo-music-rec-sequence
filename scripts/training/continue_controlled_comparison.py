"""Two-process scheduling of the unchanged fixed comparison, adopting the first run.

Windows-only continuation used after stopping ONLY the original coordinator.
The adopted training child is neither restarted nor modified. All updates are
recorded separately from the original sequential execution log.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.training.records import sha256, write_json

REPORT = ROOT / "results/controlled_comparison_20260913"
OUT = ROOT / "artifacts/runs/controlled_comparison_20260913"
LOCK = threading.Lock()
RECORDS = []


def save():
    with LOCK:
        write_json(REPORT / "execution_parallel.json", RECORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-pid", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    for path, digest in plan["code_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError(f"Frozen experiment source changed: {path}")
    prior = json.loads((REPORT / "execution.json").read_text(encoding="utf-8"))
    if len(prior) != 1 or prior[0]["pid"] != args.adopt_pid or prior[0]["name"] != "no_va":
        raise ValueError("Adopted process is not the recorded first variant")
    if (REPORT / "execution_parallel.json").exists():
        raise FileExistsError("Refusing to overwrite continuation records")
    write_json(REPORT / "scheduling_amendment.json", {
        "date_utc": datetime.now(timezone.utc).isoformat(), "reason": "two concurrent small-head training jobs to reduce elapsed time",
        "scientific_protocol_changed": False, "frozen_plan_sha256": sha256(REPORT / "plan.json"),
        "training_child_adopted": args.adopt_pid, "coordinator_only_stopped": 39636,
        "concurrency": 2, "new_script_sha256": sha256(Path(__file__)),
        "resource_check": "7.7 GB host RAM and 1.36 GB GPU memory available with first job running",
    })

    def worker(variant):
        name = variant["name"]
        record = dict(prior[0]) if name == "no_va" else {
            "name": name, "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": variant["command"], "status": "running",
        }
        with LOCK:
            RECORDS.append(record)
        save()
        started_utc = datetime.fromisoformat(record["started_utc"])
        log_path = REPORT / "logs" / f"{name}.log"
        process = None
        stream = None
        handle = None
        if name == "no_va":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, False, args.adopt_pid)
            if not handle:
                raise OSError("Cannot observe adopted training process")
        else:
            stream = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(variant["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
        save()
        print(f"MONITOR {name} pid={record['pid']}", flush=True)
        try:
            while True:
                if process is not None:
                    code = process.poll()
                else:
                    status = ctypes.c_ulong()
                    if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                        raise OSError("Cannot get adopted process status")
                    code = None if status.value == 259 else int(status.value)
                elapsed = (datetime.now(timezone.utc) - started_utc).total_seconds()
                if code is not None:
                    break
                if elapsed > plan["timeout_seconds_per_variant"]:
                    if process is not None:
                        process.terminate()
                        process.wait(timeout=30)
                    else:
                        # The only target is the explicitly recorded training PID.
                        subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {args.adopt_pid}"], check=True)
                    record.update(status="timeout", duration_seconds=elapsed)
                    save()
                    raise TimeoutError(name)
                print(f"ALIVE {name} elapsed={elapsed:.0f}s bytes={log_path.stat().st_size}", flush=True)
                time.sleep(30)
            record.update(exit_code=code, duration_seconds=elapsed,
                          status="completed" if code == 0 else "failed",
                          finished_utc=datetime.now(timezone.utc).isoformat())
            save()
            if code or not (OUT / name / "aggregate.json").exists():
                raise RuntimeError(f"Incomplete training: {name}, code {code}")
            print(f"DONE {name} {elapsed:.0f}s", flush=True)
        finally:
            if stream is not None:
                stream.close()
            if handle:
                kernel.CloseHandle(handle)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, variant) for variant in plan["variants"]]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
