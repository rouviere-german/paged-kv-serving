from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_budget import hypothesis_examples
from paged_kv_serving.engine.config import ModelConfig
from paged_kv_serving.engine.model import (
    LanguageModel,
    LayerWeights,
    TransformerWeights,
    apply_rope,
    random_weights,
    rms_norm,
    rope_tables,
    silu,
    softmax_last,
)

pytestmark = pytest.mark.unit

CONFIG = ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=64)


def test_rms_norm_scales_to_unit_root_mean_square() -> None:
    x = np.asarray([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    normed = rms_norm(x, np.ones(4, dtype=np.float32), 1e-12)
    assert np.isclose(np.sqrt(np.mean(normed**2)), 1.0, atol=1e-5)


def test_silu_and_softmax_stay_finite_on_extreme_inputs() -> None:
    extreme = np.asarray([-1e4, 0.0, 1e4], dtype=np.float32)
    assert np.all(np.isfinite(silu(extreme)))
    probabilities = softmax_last(extreme)
    assert np.all(np.isfinite(probabilities))
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)


def test_rope_preserves_norm_and_encodes_relative_position() -> None:
    cos, sin = rope_tables(32, 8, 10_000.0)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((4, 2, 8)).astype(np.float32)
    rotated = apply_rope(vectors, np.asarray([0, 1, 2, 3], dtype=np.int64), cos, sin)
    assert np.allclose(
        np.linalg.norm(rotated, axis=-1), np.linalg.norm(vectors, axis=-1), atol=1e-5
    )
    # The dot product of a rotated query and key depends only on the position difference.
    query = rng.standard_normal((1, 1, 8)).astype(np.float32)
    key = rng.standard_normal((1, 1, 8)).astype(np.float32)
    pairs = [(0, 3), (5, 8), (11, 14)]
    products = [
        float(
            np.sum(
                apply_rope(query, np.asarray([a], dtype=np.int64), cos, sin)
                * apply_rope(key, np.asarray([b], dtype=np.int64), cos, sin)
            )
        )
        for a, b in pairs
    ]
    assert max(products) - min(products) < 1e-4


def test_rope_requires_an_even_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        rope_tables(8, 7, 10_000.0)


def test_random_weights_are_deterministic_and_validate() -> None:
    first = random_weights(CONFIG, seed=1)
    second = random_weights(CONFIG, seed=1)
    assert np.array_equal(first.token_embedding, second.token_embedding)
    assert not np.array_equal(first.token_embedding, random_weights(CONFIG, seed=2).token_embedding)
    first.validate(CONFIG)


def test_weight_validation_rejects_wrong_shapes_dtypes_and_non_finite_values() -> None:
    weights = random_weights(CONFIG, seed=1)
    with pytest.raises(ValueError, match="expected 2 layers"):
        TransformerWeights(
            weights.token_embedding, weights.layers[:1], weights.final_norm
        ).validate(CONFIG)
    wrong_dtype = dataclasses.replace(
        weights, token_embedding=weights.token_embedding.astype(np.float64)
    )
    with pytest.raises(ValueError, match="expected float32"):
        wrong_dtype.validate(CONFIG)
    wrong_shape = dataclasses.replace(weights, final_norm=np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError, match="expected shape"):
        wrong_shape.validate(CONFIG)
    broken = weights.token_embedding.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        dataclasses.replace(weights, token_embedding=broken).validate(CONFIG)


def test_tensors_round_trip_and_reject_missing_or_extra_names() -> None:
    weights = random_weights(CONFIG, seed=3)
    tensors = weights.to_tensors()
    assert len(tensors) == 2 + CONFIG.n_layers * len(LayerWeights.__slots__)
    restored = TransformerWeights.from_tensors(dict(tensors), CONFIG)
    assert np.array_equal(restored.layers[1].qkv_proj, weights.layers[1].qkv_proj)
    with pytest.raises(ValueError, match="missing tensors"):
        TransformerWeights.from_tensors(
            {k: v for k, v in tensors.items() if k != "final_norm"}, CONFIG
        )
    with pytest.raises(ValueError, match="unexpected tensors"):
        TransformerWeights.from_tensors({**tensors, "surprise": np.zeros(1, np.float32)}, CONFIG)


def test_model_config_validation_and_parameter_count() -> None:
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(vocab_size=8, d_model=10, n_layers=1, n_heads=4, d_ff=8, max_seq_len=8)
    with pytest.raises(ValueError, match="must be >= 1"):
        ModelConfig(vocab_size=0, d_model=8, n_layers=1, n_heads=2, d_ff=8, max_seq_len=8)
    with pytest.raises(ValueError, match="rope_theta"):
        ModelConfig(
            vocab_size=8, d_model=8, n_layers=1, n_heads=2, d_ff=8, max_seq_len=8, rope_theta=1.0
        )
    with pytest.raises(ValueError, match="norm_eps"):
        ModelConfig(
            vocab_size=8, d_model=8, n_layers=1, n_heads=2, d_ff=8, max_seq_len=8, norm_eps=0.0
        )
    assert CONFIG.head_dim == 8
    assert CONFIG.parameter_count > 0
    assert ModelConfig.from_dict(CONFIG.to_dict()) == CONFIG
    with pytest.raises(ValueError, match="unknown model config fields"):
        ModelConfig.from_dict({**CONFIG.to_dict(), "extra": 1})


def test_reference_forward_rejects_out_of_range_input() -> None:
    model = LanguageModel(CONFIG, random_weights(CONFIG, seed=4))
    with pytest.raises(ValueError, match="non-empty"):
        model.reference_forward([])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.reference_forward([CONFIG.vocab_size])
    with pytest.raises(ValueError, match="longer than max_seq_len"):
        model.reference_forward(list(range(CONFIG.vocab_size)) * 4)


@settings(max_examples=hypothesis_examples(20), deadline=None)
@given(
    tokens=st.lists(st.integers(min_value=0, max_value=31), min_size=1, max_size=24),
)
def test_reference_forward_is_causal(tokens: list[int]) -> None:
    """Logits at position i must not depend on tokens after i: that is what causality means."""
    model = LanguageModel(CONFIG, random_weights(CONFIG, seed=5))
    full = model.reference_forward(tokens)
    for cut in {1, len(tokens)}:
        prefix = model.reference_forward(tokens[:cut])
        assert np.max(np.abs(prefix[cut - 1] - full[cut - 1])) < 1e-4


def test_reference_forward_shape_and_finiteness() -> None:
    model = LanguageModel(CONFIG, random_weights(CONFIG, seed=6))
    logits = model.reference_forward([1, 2, 3, 4])
    assert logits.shape == (4, CONFIG.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))
    assert model.rope[0].shape == (CONFIG.max_seq_len, CONFIG.head_dim // 2)
    assert np.isclose(model.attention_scale, CONFIG.head_dim**-0.5)
