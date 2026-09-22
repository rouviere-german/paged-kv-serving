from __future__ import annotations

import numpy as np
import pytest

from paged_kv_serving.engine.monitoring import PromptLengthMonitor

pytestmark = pytest.mark.unit


def reference_lengths(seed: int = 0, size: int = 400) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(20, 60, size=size)]


def test_stable_traffic_does_not_report_drift() -> None:
    monitor = PromptLengthMonitor(reference_lengths(), min_samples=50)
    rng = np.random.default_rng(1)
    for value in rng.integers(20, 60, size=200):
        monitor.observe(int(value))
    assert monitor.score() < monitor.threshold
    assert not monitor.drifted()


def test_a_shift_in_prompt_lengths_is_reported() -> None:
    monitor = PromptLengthMonitor(reference_lengths(), min_samples=50)
    rng = np.random.default_rng(2)
    for value in rng.integers(300, 400, size=200):
        monitor.observe(int(value))
    assert monitor.drifted()
    assert monitor.score() > monitor.threshold


def test_the_score_is_zero_until_enough_samples_arrive() -> None:
    monitor = PromptLengthMonitor(reference_lengths(), min_samples=32)
    for _ in range(31):
        monitor.observe(1000)
    assert monitor.score() == 0.0
    assert len(monitor) == 31
    monitor.observe(1000)
    assert monitor.score() > 0.0


def test_the_window_forgets_old_traffic() -> None:
    monitor = PromptLengthMonitor(reference_lengths(), window=64, min_samples=32)
    for _ in range(64):
        monitor.observe(500)
    assert monitor.drifted()
    rng = np.random.default_rng(3)
    for value in rng.integers(20, 60, size=64):
        monitor.observe(int(value))
    assert len(monitor) == 64
    assert not monitor.drifted(), "the shifted window should have rolled out"


def test_construction_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="at least two prompt lengths"):
        PromptLengthMonitor([10])
    with pytest.raises(ValueError, match="window, bins and min_samples"):
        PromptLengthMonitor(reference_lengths(), window=1)
    monitor = PromptLengthMonitor(reference_lengths())
    assert monitor.reference_size == 400
    with pytest.raises(ValueError, match="prompt_length must be"):
        monitor.observe(-1)


def test_a_learning_monitor_freezes_its_first_prompts_as_the_reference() -> None:
    # PSI over 10 bins has a noise floor of roughly 9/n per sample, so both samples need a few
    # hundred prompts before 0.2 means drift rather than chance (the served defaults are 256).
    monitor = PromptLengthMonitor.learning(400, window=256, min_samples=32)
    rng = np.random.default_rng(4)
    for value in rng.integers(20, 60, size=400):
        monitor.observe(int(value))
    assert monitor.reference_ready
    assert monitor.reference_size == 400
    assert len(monitor) == 0, "reference prompts are not part of the live window"
    assert monitor.score() == 0.0
    for value in rng.integers(20, 60, size=256):
        monitor.observe(int(value))
    assert not monitor.drifted()
    for _ in range(256):
        monitor.observe(900)
    assert monitor.drifted()
    assert monitor.reference_size == 400, "the reference never moves once frozen"


def test_a_learning_monitor_needs_a_real_reference() -> None:
    with pytest.raises(ValueError, match="at least two prompt lengths"):
        PromptLengthMonitor.learning(1)
    monitor = PromptLengthMonitor.learning(10, min_samples=2)
    for _ in range(9):
        monitor.observe(10)
    assert not monitor.reference_ready
    assert monitor.score() == 0.0
