from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_budget import hypothesis_examples
from paged_kv_serving.engine.corpus import DEMO_CORPUS
from paged_kv_serving.engine.tokenizer import EOS_TOKEN, ByteBPETokenizer

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def trained() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(DEMO_CORPUS, vocab_size=400)


def test_training_is_deterministic() -> None:
    first = ByteBPETokenizer.train(DEMO_CORPUS, vocab_size=320)
    second = ByteBPETokenizer.train(DEMO_CORPUS, vocab_size=320)
    assert first.merges == second.merges
    assert first.vocab_size == second.vocab_size


def test_merges_shorten_text_from_the_training_corpus(trained: ByteBPETokenizer) -> None:
    text = DEMO_CORPUS[0]
    assert trained.vocab_size == 400
    assert len(trained.encode(text)) < len(text.encode("utf-8"))


@settings(max_examples=hypothesis_examples(60), deadline=None)
@given(text=st.text(max_size=200))
def test_every_string_round_trips(text: str, trained: ByteBPETokenizer) -> None:
    """Byte fallback means encoding is total: there is no unknown token and nothing is lost."""
    assert trained.decode(trained.encode(text)) == text


def test_special_tokens_are_added_and_skipped_on_decode(trained: ByteBPETokenizer) -> None:
    ids = trained.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == trained.bos_token_id
    assert ids[-1] == trained.eos_token_id
    assert trained.decode(ids) == "hello"
    assert trained.decode(ids, skip_special=False) == "hello"
    assert trained.specials[EOS_TOKEN] == trained.eos_token_id


def test_non_ascii_text_survives(trained: ByteBPETokenizer) -> None:
    text = "grüße, 世界 🙂"
    assert trained.decode(trained.encode(text)) == text


def test_a_partial_token_stream_decodes_without_raising(trained: ByteBPETokenizer) -> None:
    """Streaming hands out one token at a time, so half a UTF-8 sequence must not crash."""
    ids = trained.encode("世界")
    partial = "".join(trained.decode([token]) for token in ids)
    assert isinstance(partial, str)
    assert trained.decode(ids) == "世界"


def test_serialisation_round_trips_through_json(tmp_path: Path, trained: ByteBPETokenizer) -> None:
    path = tmp_path / "tokenizer.json"
    trained.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    restored = ByteBPETokenizer.load(path)
    assert restored.merges == trained.merges
    assert restored.encode("paged attention") == trained.encode("paged attention")


def test_unsupported_or_corrupt_vocabularies_are_refused() -> None:
    with pytest.raises(ValueError, match="unsupported tokenizer schema"):
        ByteBPETokenizer.from_dict({"schema_version": 99, "merges": [], "specials": {}})
    with pytest.raises(ValueError, match="does not exist yet"):
        ByteBPETokenizer([(9999, 1)])
    with pytest.raises(ValueError, match="out-of-range id"):
        ByteBPETokenizer((), {"<|a|>": 10})
    with pytest.raises(ValueError, match="unique"):
        ByteBPETokenizer((), {"<|a|>": 256, "<|b|>": 256})


def test_training_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="vocab_size must be at least"):
        ByteBPETokenizer.train(["hello"], vocab_size=10)
    with pytest.raises(ValueError, match="corpus is empty"):
        ByteBPETokenizer.train([], vocab_size=300)
    # A corpus with no repeated pair stops early instead of inventing merges.
    sparse = ByteBPETokenizer.train(["abcdef"], vocab_size=400)
    assert sparse.vocab_size < 400


def test_token_bytes_bounds_are_checked(trained: ByteBPETokenizer) -> None:
    assert trained.token_bytes(65) == b"A"
    with pytest.raises(ValueError, match="outside the vocabulary"):
        trained.token_bytes(trained.vocab_size)
