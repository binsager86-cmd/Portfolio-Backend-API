"""
Push Notification Service — sends push notifications via Expo Push API.

Uses the Expo Push Notification service (https://exp.host/--/api/v2/push/send)
to deliver notifications to registered devices.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def send_push_notifications(
    tokens: list[str],
    title: str,
    body: str,
    data: Optional[dict] = None,
) -> dict:
    """
    Send push notifications to a list of Expo push tokens.

    Batches tokens in groups of 100 (Expo API limit).
    Returns summary of sent/failed counts.
    """
    if not tokens:
        return {"sent": 0, "failed": 0}

    messages = []
    for token in tokens:
        msg = {
            "to": token,
            "title": title,
            "body": body,
            "sound": "default",
        }
        if data:
            msg["data"] = data
        messages.append(msg)

    sent = 0
    failed = 0
    chunk_size = 100

    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i: i + chunk_size]
            try:
                resp = client.post(
                    EXPO_PUSH_URL,
                    json=chunk,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                result = resp.json()
                tickets = result.get("data", [])
                for ticket in tickets:
                    if ticket.get("status") == "ok":
                        sent += 1
                    else:
                        failed += 1
                        err = ticket.get("details", {}).get("error", "unknown")
                        logger.warning("Push ticket error: %s", err)
            except Exception as e:
                logger.warning("Expo push send failed: %s", e)
                failed += len(chunk)

    logger.info("Push notifications: sent=%d, failed=%d", sent, failed)
    return {"sent": sent, "failed": failed}


def notify_users_for_article(
    article_symbols: list[str],
    article_title: str,
    article_id: str,
    article_category: str,
) -> dict:
    """
    Send push notifications to all users who hold any of the article's symbols.

    Looks up user holdings from the stocks table, then finds their push tokens.
    """
    if not article_symbols:
        return {"sent": 0, "failed": 0, "reason": "no_symbols"}

    from app.core.database import SessionLocal
    from app.models.portfolio import Stock
    from app.models.push_token import PushToken

    db = SessionLocal()
    try:
        # Find all user_ids that hold any of the article's symbols
        symbols_upper = [s.strip().upper() for s in article_symbols if s.strip()]
        if not symbols_upper:
            return {"sent": 0, "failed": 0, "reason": "no_symbols"}

        from sqlalchemy import func
        user_ids = (
            db.query(Stock.user_id)
            .filter(func.upper(Stock.symbol).in_(symbols_upper))
            .distinct()
            .all()
        )
        user_id_list = [uid[0] for uid in user_ids]

        if not user_id_list:
            return {"sent": 0, "failed": 0, "reason": "no_holders"}

        # Get push tokens for those users
        tokens = (
            db.query(PushToken.token)
            .filter(PushToken.user_id.in_(user_id_list))
            .all()
        )
        token_list = [t[0] for t in tokens]

        if not token_list:
            return {"sent": 0, "failed": 0, "reason": "no_tokens"}

        symbols_str = ", ".join(symbols_upper)
        title = f"📰 {symbols_str} — New Announcement"
        body = article_title[:200] if article_title else "New market announcement"
        data = {
            "newsId": article_id,
            "type": "news",
            "category": article_category,
            "symbols": symbols_upper,
        }

        return send_push_notifications(token_list, title, body, data)
    except Exception as e:
        logger.warning("notify_users_for_article failed: %s", e)
        return {"sent": 0, "failed": 0, "error": str(e)}
    finally:
        db.close()
