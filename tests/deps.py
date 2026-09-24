"""Helpers for tests written while a dependency owned by another agent may still be a
scaffold stub (``NotImplementedError``). The orchestrator runs the merged suite; skips produced
here must be gone at the end."""

import pytest


def call_or_skip(fn, *args, **kwargs):
    """``fn(*args, **kwargs)``, or ``pytest.skip`` when ``fn`` is still a scaffold stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"dependency not implemented yet: {getattr(fn, '__qualname__', fn)} ({e})")
