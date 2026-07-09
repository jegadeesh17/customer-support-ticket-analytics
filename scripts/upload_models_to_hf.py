#!/usr/bin/env python3
"""Upload sklearn model bundles to Hugging Face Hub (one-time setup)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from huggingface_hub import HfApi, create_repo

from src.model_assets import MODEL_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload support analytics models to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/support-ops-models")
    parser.add_argument("--private", action="store_true", help="Create a private model repo")
    args = parser.parse_args()

    models_dir = os.path.join(ROOT, "models")
    missing = [name for name in MODEL_FILES if not os.path.exists(os.path.join(models_dir, name))]
    if missing:
        raise SystemExit(
            "Missing model files: "
            + ", ".join(missing)
            + "\nRun: python src/train_models.py"
        )

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    for filename in MODEL_FILES:
        api.upload_file(
            path_or_fileobj=os.path.join(models_dir, filename),
            path_in_repo=filename,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Uploaded {filename}")

    print(f"Models available at https://huggingface.co/{args.repo_id}")
    print(f"Set HF_MODEL_REPO={args.repo_id} in Cloud Run / Streamlit secrets.")


if __name__ == "__main__":
    main()
