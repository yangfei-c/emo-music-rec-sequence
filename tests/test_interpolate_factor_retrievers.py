from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/training/interpolate_factor_retrievers.py"
)
SPEC = importlib.util.spec_from_file_location("interpolate_factor_retrievers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_interpolate_only_changes_requested_factor() -> None:
    base = {
        "text_adapters.semantic.up.weight": torch.tensor([1.0]),
        "text_adapters.preference.up.weight": torch.tensor([2.0]),
        "music_adapters.preference.up.weight": torch.tensor([4.0]),
        "logit_scales.preference": torch.tensor(6.0),
        "text_va_head.weight": torch.tensor([8.0]),
    }
    candidate = {key: value + 10.0 for key, value in base.items()}
    state = MODULE.interpolate_state(base, candidate, "preference", 0.25)
    torch.testing.assert_close(
        state["text_adapters.preference.up.weight"], torch.tensor([4.5])
    )
    torch.testing.assert_close(
        state["music_adapters.preference.up.weight"], torch.tensor([6.5])
    )
    torch.testing.assert_close(state["logit_scales.preference"], torch.tensor(8.5))
    torch.testing.assert_close(
        state["text_adapters.semantic.up.weight"], torch.tensor([1.0])
    )
    torch.testing.assert_close(state["text_va_head.weight"], torch.tensor([8.0]))
