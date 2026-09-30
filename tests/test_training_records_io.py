"""Progress writes survive transient Windows reader locks without hiding errors."""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

import pytest

from emomm_sequence.training import records


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_replace_failure_preserves_old_json_until_success(tmp_path, monkeypatch, winerror):
    target = tmp_path / "progress.json"
    target.write_text('{"step": 1}', encoding="utf-8")
    replace = Path.replace
    attempts = []
    delays = []

    def temporarily_locked(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text(encoding="utf-8")) == {"step": 1}
        if len(attempts) < 3:
            error = PermissionError("reader has not released its handle")
            error.winerror = winerror
            raise error
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr(records.time, "sleep", delays.append)
    records.write_json(target, {"step": 2, "label": "继续"})
    assert len(attempts) == 3
    assert delays == [0.05, 0.1]
    assert json.loads(target.read_text(encoding="utf-8")) == {"step": 2, "label": "继续"}
    assert not target.with_suffix(".json.tmp").exists()


def test_persistent_windows_error_is_bounded_and_preserves_both_versions(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"step": 1}', encoding="utf-8")
    error = PermissionError("persistent access denial")
    error.winerror = 5
    attempts = []
    delays = []

    def always_denied(*args):
        attempts.append(args)
        raise error

    monkeypatch.setattr(Path, "replace", always_denied)
    monkeypatch.setattr(records.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as caught:
        records.write_json(target, {"step": 2})
    assert caught.value is error
    assert len(attempts) == 10
    assert len(delays) == 9 and sum(delays) < 4
    assert json.loads(target.read_text(encoding="utf-8")) == {"step": 1}
    assert json.loads(target.with_suffix(".json.tmp").read_text(encoding="utf-8")) == {"step": 2}


def test_unrelated_io_error_is_not_retried(tmp_path, monkeypatch):
    error = OSError(28, "no space left on device")
    delays = []

    def fail(*args):
        raise error

    monkeypatch.setattr(Path, "replace", fail)
    monkeypatch.setattr(records.time, "sleep", delays.append)
    with pytest.raises(OSError) as caught:
        records.write_json(tmp_path / "progress.json", {"step": 2})
    assert caught.value is error
    assert delays == []


@pytest.mark.skipif(os.name != "nt", reason="requires Windows sharing semantics")
def test_actual_windows_reader_handle_can_release_during_retry(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"step": 1}', encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ; permit other readers/writers, but deliberately omit FILE_SHARE_DELETE.
    handle = kernel.CreateFileW(str(target), 0x80000000, 0x3, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    released = False
    delays = []

    def release_reader(delay):
        nonlocal released
        delays.append(delay)
        assert json.loads(target.read_text(encoding="utf-8")) == {"step": 1}
        assert kernel.CloseHandle(handle)
        released = True

    monkeypatch.setattr(records.time, "sleep", release_reader)
    try:
        records.write_json(target, {"step": 2})
    finally:
        if not released:
            kernel.CloseHandle(handle)
    assert released and len(delays) == 1
    assert json.loads(target.read_text(encoding="utf-8")) == {"step": 2}
