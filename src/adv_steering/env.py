from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_LOADED = False


def load_project_env() -> None:
    global _LOADED
    if _LOADED:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path, override=False)
    _LOADED = True


def get_hf_token() -> str | None:
    load_project_env()
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
