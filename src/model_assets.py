"""Resolve sklearn model bundles from local disk or Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_FILES = (
    "classification_model.pkl",
    "regression_model.pkl",
    "satisfaction_model.pkl",
)


def hf_repo_id() -> str | None:
    return os.getenv("HF_MODEL_REPO")


def ensure_models(models_dir: str) -> str:
    """Download any missing model bundles from HF Hub when configured."""
    missing = [
        filename
        for filename in MODEL_FILES
        if not os.path.exists(os.path.join(models_dir, filename))
    ]
    if not missing:
        return models_dir

    repo = hf_repo_id()
    if not repo:
        return models_dir

    from huggingface_hub import hf_hub_download

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    for filename in missing:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
    return models_dir
