"""Tests for strata/entrypoint.py: merging the user engine config with image paths."""

from pathlib import Path

import entrypoint
import pytest

ROOT = Path("/opt/strata")
TUNING = ROOT / "hipblaslt-tuning.txt"


def _user(**extra) -> dict:
    """Build a minimal valid user config."""
    return {"args": ["--spec", "4"], "tokenizer": "/data/tok", "model_name": "flash", **extra}


def test_merge_fills_image_paths():
    """The engine, libraries, backend, and log come from the image."""
    merged = entrypoint.merge_config(_user(), ROOT, TUNING)

    assert merged["exe"] == "/opt/strata/engine/strata"
    assert merged["cwd"] == "/opt/strata"
    assert merged["lib_dirs"] == ["/opt/strata/lib"]
    assert merged["backend"] == "hip"
    assert merged["args"] == ["--spec", "4"]
    assert merged["env"] == {"STRATA_HIPBLASLT_TUNING": str(TUNING)}


def test_merge_keeps_user_env_and_tuning_override():
    """User env passes through, and a user tuning table wins over the image's."""
    user = _user(env={"STRATA_RESIDENT_PIN": "0", "STRATA_HIPBLASLT_TUNING": "/data/t.txt"})

    merged = entrypoint.merge_config(user, ROOT, TUNING)

    assert merged["env"] == {"STRATA_RESIDENT_PIN": "0", "STRATA_HIPBLASLT_TUNING": "/data/t.txt"}


def test_merge_without_tuning_table_sets_no_tuning_env():
    """An image without a tuning table leaves the variable unset."""
    assert entrypoint.merge_config(_user(), ROOT, None)["env"] == {}


@pytest.mark.parametrize(
    ("user", "message"),
    [
        ({"args": []}, "missing tokenizer, model_name"),
        (_user(exe="/bin/strata"), "sets exe; the image provides these"),
        (_user(cwd="/x", log="/y"), "sets cwd, log"),
    ],
    ids=["missing-keys", "sets-exe", "sets-several"],
)
def test_merge_rejects_invalid_config(user, message):
    """Missing required keys and image-owned keys raise ValueError."""
    with pytest.raises(ValueError, match=message):
        entrypoint.merge_config(user, ROOT, TUNING)
