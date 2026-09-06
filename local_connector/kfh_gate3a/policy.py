"""Closed Gate 3A navigation and action policy."""

from __future__ import annotations

from enum import StrEnum
from urllib.parse import urlsplit

KFH_START_URL = "https://trading.kfhtrade.com/"
KFH_ALLOWED_ORIGINS = frozenset({"https://trading.kfhtrade.com"})


class KfhApprovedAction(StrEnum):
    LOGIN = "LOGIN"
    STATEMENTS = "STATEMENTS"
    PORTFOLIO = "PORTFOLIO"
    ACCOUNT_SUMMARY = "ACCOUNT_SUMMARY"
    LOGOUT = "LOGOUT"


APPROVED_ACTIONS = frozenset(KfhApprovedAction)


def require_approved_action(action: KfhApprovedAction) -> None:
    if action not in APPROVED_ACTIONS:
        raise PermissionError("KFH browser action is not allowlisted")


def is_allowed_kfh_url(url: str) -> bool:
    if url in {"about:blank", "data:,"} or url.startswith(("blob:", "data:")):
        return True
    parsed = urlsplit(url)
    origin = f"{parsed.scheme}://{parsed.netloc}".lower()
    return origin in KFH_ALLOWED_ORIGINS
