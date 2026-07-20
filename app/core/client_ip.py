"""Canonical client IP extraction for rate limits and audit logs."""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from app.core.config import get_settings


def _trusted_proxy_ips() -> set[str]:
    settings = get_settings()
    return {ip.strip() for ip in settings.TRUSTED_PROXY_IPS.split(",") if ip.strip()}


def get_client_ip(request: Optional[Request] = None) -> Optional[str]:
    if request is None:
        return None

    direct_ip = request.client.host if request.client else None
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and direct_ip and direct_ip in _trusted_proxy_ips():
        first_hop = forwarded.split(",", 1)[0].strip()
        return first_hop or direct_ip

    return direct_ip