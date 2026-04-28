"""Download Mistral-Small-3.2-24B-Instruct-2506 from Hugging Face.

Uses hf_transfer for fast parallel downloads. Skips files that we don't need
(e.g. consolidated single-file weights, GGUFs, ONNX) to save bandwidth/disk.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

REPO_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
LOCAL_DIR = "/home/alex/ablated/models/mistral-small-3.2-24b"


def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        # Skip giant single-file consolidated checkpoints — we only need the
        # sharded safetensors that transformers loads from.
        ignore_patterns=[
            "consolidated*",
            "*.gguf",
            "*.bin",
            "original/*",
            "*.pt",
            "*.onnx*",
        ],
        max_workers=8,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
