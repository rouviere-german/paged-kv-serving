"""Example budgets for property tests that follow the active hypothesis profile.

An explicit ``@settings(max_examples=...)`` overrides whatever profile is loaded, so a test that
pins its own count silently ignores ``HYPOTHESIS_PROFILE=ci``. Property tests therefore take their
count from :func:`hypothesis_examples`: the local budget in the ``dev`` profile, five times that
in ``ci`` (the merge gate), and whatever ``HYPOTHESIS_SCALE`` says when it is set.
"""

from __future__ import annotations

import os

__all__ = ["PROFILE", "hypothesis_examples"]

PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "dev")
_SCALE = {"dev": 1, "ci": 5}


def hypothesis_examples(base: int) -> int:
    """``base`` examples locally, scaled up under the CI profile."""
    override = os.environ.get("HYPOTHESIS_SCALE")
    scale = int(override) if override else _SCALE.get(PROFILE, 1)
    return max(1, base * scale)
