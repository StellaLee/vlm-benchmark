"""Put src/ on sys.path and load .env, so scripts run without an editable install
or manual `export`s."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load_dotenv(path):
    """Minimal .env loader (no dependency). Existing env vars win, so a real
    `export` always overrides the file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv(os.path.join(_ROOT, ".env"))
