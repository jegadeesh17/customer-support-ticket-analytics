"""Tests for Hugging Face model resolution helpers."""

import os
import sys
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def test_ensure_models_returns_existing_dir(tmp_path):
    from src.model_assets import MODEL_FILES, ensure_models

    for filename in MODEL_FILES:
        (tmp_path / filename).write_bytes(b"bundle")
    result = ensure_models(str(tmp_path))
    assert result == str(tmp_path)


def test_ensure_models_skips_without_repo(tmp_path):
    from src.model_assets import ensure_models

    with patch.dict(os.environ, {}, clear=True):
        result = ensure_models(str(tmp_path))
    assert result == str(tmp_path)


def test_ensure_models_downloads_missing_files(tmp_path):
    from src.model_assets import MODEL_FILES, ensure_models

    with patch.dict(os.environ, {"HF_MODEL_REPO": "demo/support-models"}, clear=False):
        with patch("huggingface_hub.hf_hub_download") as mocked:
            ensure_models(str(tmp_path))
    assert mocked.call_count == len(MODEL_FILES)
