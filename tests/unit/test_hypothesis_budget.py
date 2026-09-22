"""The CI hypothesis profile must actually reach every property test.

Review regression: every property test pinned ``max_examples`` with a literal, and an explicit
``@settings(max_examples=...)`` overrides the loaded profile, so ``HYPOTHESIS_PROFILE=ci`` changed
nothing and the merge gate ran the same 15-40 examples as a laptop.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hypothesis_budget import hypothesis_examples

pytestmark = pytest.mark.unit

TESTS = Path(__file__).resolve().parents[1]


def _literal_example_counts(path: Path) -> list[int]:
    """Line numbers where ``max_examples`` is given as a literal rather than a budget."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "max_examples" and isinstance(keyword.value, ast.Constant):
                lines.append(node.lineno)
    return lines


def test_no_property_test_pins_its_own_example_count() -> None:
    offenders = {
        str(path.relative_to(TESTS)): lines
        for path in sorted(TESTS.rglob("test_*.py"))
        if (lines := _literal_example_counts(path))
    }
    assert offenders == {}, "use hypothesis_examples(base) so the CI profile can scale it"


def test_the_budget_scales_with_the_profile_and_the_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYPOTHESIS_SCALE", raising=False)
    assert hypothesis_examples(40) in {40, 200}  # dev or ci, whichever this run loaded
    monkeypatch.setenv("HYPOTHESIS_SCALE", "3")
    assert hypothesis_examples(40) == 120
    monkeypatch.setenv("HYPOTHESIS_SCALE", "0")
    assert hypothesis_examples(40) == 1
