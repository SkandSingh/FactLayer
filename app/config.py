"""Application configuration, loaded from the environment (and `.env`).

Import this module to get module-level settings constants; values are
read once at import time via `python-dotenv`'s `load_dotenv()`.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_CONCURRENT_LLM_CALLS: int = _get_int("MAX_CONCURRENT_LLM_CALLS", 4)
LLM_TIMEOUT_SECONDS: int = _get_int("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 3)
FACTLAYER_DB_PATH: str = os.environ.get("FACTLAYER_DB_PATH", "factlayer.db")
UPLOAD_MAX_BYTES: int = _get_int("UPLOAD_MAX_BYTES", 52428800)
