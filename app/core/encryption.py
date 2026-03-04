"""
Field-level encryption — Fernet symmetric encryption for sensitive DB fields.

Usage:
    from app.core.encryption import encrypt_field, decrypt_field

    encrypted = encrypt_field("sk-my-api-key-123")
    original  = decrypt_field(encrypted)

The key is read from settings.FIELD_ENCRYPTION_KEY.
If the key is empty (dev), encryption is a no-op (values stored as-is with a prefix).
"""

import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "enc::"  # Marker prefix so we know a value is encrypted


def _get_fernet():
    """Lazy-load Fernet to avoid import cost when encryption is disabled."""
    settings = get_settings()
    if not settings.FIELD_ENCRYPTION_KEY:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(settings.FIELD_ENCRYPTION_KEY.encode())
    except Exception as exc:
        logger.error("Failed to initialise Fernet cipher: %s", exc)
        return None


def encrypt_field(value: Optional[str]) -> Optional[str]:
    """Encrypt a string field for storage. Returns prefixed ciphertext."""
    if value is None:
        return None

    fernet = _get_fernet()
    if fernet is None:
        # No encryption key configured — store as-is (dev mode)
        return value

    try:
        token = fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        return f"{_PREFIX}{token}"
    except Exception as exc:
        logger.error("Encryption failed: %s", exc)
        return value  # fallback: store plaintext


def decrypt_field(value: Optional[str]) -> Optional[str]:
    """Decrypt a previously encrypted field. Returns plaintext."""
    if value is None:
        return None
    if not value.startswith(_PREFIX):
        # Not encrypted (legacy or dev value) — return as-is
        return value

    fernet = _get_fernet()
    if fernet is None:
        logger.warning("Cannot decrypt — no FIELD_ENCRYPTION_KEY configured")
        return value  # return the raw encrypted string

    try:
        ciphertext = value[len(_PREFIX):]
        return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        logger.error("Decryption failed: %s", exc)
        return None
