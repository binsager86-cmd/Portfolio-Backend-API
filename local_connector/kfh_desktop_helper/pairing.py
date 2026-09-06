"""One-time pairing nonce for the desktop helper.

The dev/test tool (kfh_gate5b.local_api) uses a single nonce shared via an
env var because there is exactly one developer running it. The desktop
helper is installed by many different account owners, so each installation
generates its own random nonce on first run, persists it locally, and the
owner pastes it once into Saham's KFH connection settings. Saham then sends
it back on every request as proof this specific browser is paired with this
specific helper installation - it is never transmitted anywhere else and
never leaves the owner's own machine except to their own browser.
"""

from __future__ import annotations

import secrets
import stat
from contextlib import suppress
from pathlib import Path

PAIRING_FILE_NAME = "kfh_helper_pairing.txt"
MIN_NONCE_LENGTH = 32


def default_pairing_dir() -> Path:
    return Path.home() / ".saham"


def load_or_create_pairing_nonce(pairing_dir: Path | None = None) -> str:
    """Return this installation's persisted nonce, generating one on first run."""
    directory = pairing_dir or default_pairing_dir()
    directory.mkdir(parents=True, exist_ok=True)
    pairing_file = directory / PAIRING_FILE_NAME
    if pairing_file.exists():
        existing = pairing_file.read_text(encoding="utf-8").strip()
        if len(existing) >= MIN_NONCE_LENGTH:
            return existing
    nonce = secrets.token_urlsafe(32)
    pairing_file.write_text(nonce, encoding="utf-8")
    _restrict_to_owner(pairing_file)
    return nonce


def _restrict_to_owner(path: Path) -> None:
    """Best-effort owner-read-write-only (0600). Windows ACLs aren't
    chmod-based, so this is a no-op there; POSIX (macOS/Linux) gets it."""
    with suppress(NotImplementedError, OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
