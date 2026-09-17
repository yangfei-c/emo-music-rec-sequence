from __future__ import annotations

import pytest

from emomm_sequence.training.temperature import controlled_temperature


def test_learned_temperature_has_no_override() -> None:
    assert (
        controlled_temperature(
            "learned", start=0.07, end=0.07, epoch=1, total_epochs=6
        )
        is None
    )


def test_fixed_and_linear_temperature_controls() -> None:
    assert controlled_temperature(
        "fixed", start=0.04, end=0.20, epoch=4, total_epochs=6
    ) == pytest.approx(0.04)
    values = [
        controlled_temperature(
            "linear", start=0.01, end=0.05, epoch=epoch, total_epochs=5
        )
        for epoch in range(1, 6)
    ]
    assert values == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])


def test_temperature_control_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        controlled_temperature("mystery", start=0.1, end=0.2, epoch=1, total_epochs=2)
    with pytest.raises(ValueError):
        controlled_temperature("fixed", start=0.0, end=0.2, epoch=1, total_epochs=2)
    with pytest.raises(ValueError):
        controlled_temperature("linear", start=0.1, end=0.2, epoch=3, total_epochs=2)
