from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_budget import hypothesis_examples
from paged_kv_serving.engine.sampling import greedy_token, sample_token, top_p_filter
from paged_kv_serving.engine.sequence import SamplingParams

pytestmark = pytest.mark.unit


def test_greedy_is_deterministic_and_breaks_ties_at_the_lowest_id() -> None:
    logits = np.asarray([1.0, 3.0, 3.0, 0.0], dtype=np.float32)
    assert greedy_token(logits) == 1
    params = SamplingParams(max_new_tokens=1, temperature=0.0)
    rng = np.random.default_rng(0)
    assert sample_token(logits, params, rng) == 1
    # Greedy consumes no randomness, so a preempted request replays identically.
    assert rng.bit_generator.state == np.random.default_rng(0).bit_generator.state


def test_top_p_keeps_the_token_that_crosses_the_threshold() -> None:
    probabilities = np.asarray([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    filtered = top_p_filter(probabilities, 0.7)
    assert filtered[0] > 0
    assert filtered[1] > 0
    assert filtered[2] == 0
    assert filtered[3] == 0
    assert np.isclose(filtered.sum(), 1.0, atol=1e-6)
    # A single token above the threshold still forms a valid nucleus.
    dominant = np.asarray([0.95, 0.05], dtype=np.float32)
    assert top_p_filter(dominant, 0.5)[0] == 1.0
    assert top_p_filter(probabilities, 1.0) is probabilities
    with pytest.raises(ValueError, match="top_p must be"):
        top_p_filter(probabilities, 0.0)


def test_sampling_depends_only_on_the_sequence_generator() -> None:
    logits = np.asarray([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    params = SamplingParams(max_new_tokens=1, temperature=1.0, top_p=0.9, seed=7)
    first = [sample_token(logits, params, np.random.default_rng(7)) for _ in range(5)]
    assert len(set(first)) == 1, "the same generator state must give the same token"
    stream_a = np.random.default_rng(7)
    stream_b = np.random.default_rng(7)
    assert [sample_token(logits, params, stream_a) for _ in range(6)] == [
        sample_token(logits, params, stream_b) for _ in range(6)
    ]


def test_temperature_widens_the_distribution() -> None:
    logits = np.asarray([0.0, 4.0, 0.0, 0.0], dtype=np.float32)
    cold = SamplingParams(max_new_tokens=1, temperature=0.1, seed=1)
    hot = SamplingParams(max_new_tokens=1, temperature=5.0, seed=1)
    rng_cold, rng_hot = np.random.default_rng(1), np.random.default_rng(1)
    cold_draws = [sample_token(logits, cold, rng_cold) for _ in range(200)]
    hot_draws = [sample_token(logits, hot, rng_hot) for _ in range(200)]
    assert len(set(cold_draws)) < len(set(hot_draws))
    assert set(cold_draws) == {1}


def test_logits_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="1-D array"):
        sample_token(
            np.zeros((2, 2), dtype=np.float32),
            SamplingParams(max_new_tokens=1, temperature=1.0),
            np.random.default_rng(0),
        )


@settings(max_examples=hypothesis_examples(40), deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False), min_size=2, max_size=64
    ),
    top_p=st.floats(min_value=0.01, max_value=1.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_sampled_ids_are_always_inside_the_nucleus(
    values: list[float], top_p: float, seed: int
) -> None:
    logits = np.asarray(values, dtype=np.float32)
    params = SamplingParams(max_new_tokens=1, temperature=1.0, top_p=top_p, seed=seed)
    token = sample_token(logits, params, np.random.default_rng(seed))
    assert 0 <= token < logits.size
    scaled = logits - logits.max()
    probabilities = np.exp(scaled.astype(np.float64))
    probabilities = (probabilities / probabilities.sum()).astype(np.float32)
    assert top_p_filter(probabilities, top_p)[token] > 0.0


def test_sampling_parameter_validation() -> None:
    for kwargs, message in (
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"seed": -1}, "seed"),
    ):
        with pytest.raises(ValueError, match=message):
            SamplingParams(**kwargs)
    assert SamplingParams(temperature=0.0).greedy
    assert not SamplingParams(temperature=0.1).greedy


@pytest.mark.parametrize("temperature", [1e-300, 1e-45, 1e-12, 1e-6])
def test_a_vanishing_temperature_is_greedy_not_a_division_by_zero(temperature: float) -> None:
    """Review regression: 1e-300 became 0.0 in float32, the division produced inf and NaN, and
    the sampler returned the last vocabulary id instead of the argmax."""
    rng = np.random.default_rng(3)
    logits = rng.standard_normal(512).astype(np.float32)
    params = SamplingParams(max_new_tokens=1, temperature=temperature, seed=3)
    assert params.greedy
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        assert sample_token(logits, params, np.random.default_rng(3)) == int(np.argmax(logits))


def test_the_smallest_sampling_temperature_is_still_finite() -> None:
    logits = np.asarray([1.0, 1.0 + 1e-3, 0.5], dtype=np.float32)
    params = SamplingParams(max_new_tokens=1, temperature=1e-4, seed=0)
    assert not params.greedy
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        token = sample_token(logits, params, np.random.default_rng(0))
    assert token in (0, 1)
