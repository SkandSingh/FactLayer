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


def _parse_keys(csv_env_var: str, singular_fallback_env_var: str | None = None) -> list[str]:
    """Parse a comma-separated list of API keys from an env var.

    Whitespace around each key is stripped and empty entries are dropped.
    If the comma-separated var is empty/unset and a singular fallback var
    name is given, falls back to a single-element list built from that
    var (if it's non-empty), so existing single-key setups keep working.
    """
    raw = os.environ.get(csv_env_var, "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys and singular_fallback_env_var:
        single = os.environ.get(singular_fallback_env_var, "").strip()
        if single:
            keys = [single]
    return keys


GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_CONCURRENT_LLM_CALLS: int = _get_int("MAX_CONCURRENT_LLM_CALLS", 4)
LLM_TIMEOUT_SECONDS: int = _get_int("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 3)
FACTLAYER_DB_PATH: str = os.environ.get("FACTLAYER_DB_PATH", "factlayer.db")
UPLOAD_MAX_BYTES: int = _get_int("UPLOAD_MAX_BYTES", 52428800)

# Multi-key / multi-provider LLM pool support. GEMINI_API_KEYS/GROK_API_KEYS
# are comma-separated lists of API keys; calls round-robin across every key
# from every provider so each additional key adds its own rate-limit quota
# to the pool. GEMINI_API_KEYS falls back to the existing singular
# GEMINI_API_KEY var when unset, so single-key setups are unaffected.
GEMINI_API_KEYS: list[str] = _parse_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
GROK_API_KEYS: list[str] = _parse_keys("GROK_API_KEYS")
GROK_MODEL: str = os.environ.get("GROK_MODEL", "grok-4-fast")
