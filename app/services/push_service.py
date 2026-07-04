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


async def send_push_notification(
    token: str,
    title: str,
    body: str,
    data: Optional[dict] = None,
    sound: str = "default",
    priority: str = "high",
    category: Optional[str] = None,
    android: Optional[dict] = None,
) -> bool:
    """Send a single rich push notification via Expo Push API."""
    if not token:
        return False

    message: dict = {
        "to": token,
        "title": title,
        "body": body,
        "sound": sound,
        "priority": priority,
    }

    if data:
        message["data"] = data
    if category:
        message["categoryId"] = category
    if android:
        # Expo supports channelId at top-level.
        if isinstance(android, dict) and android.get("channelId"):
            message["channelId"] = android["channelId"]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                EXPO_PUSH_URL,
                json=message,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            tickets = payload.get("data", [])
            if isinstance(tickets, dict):
                tickets = [tickets]
            ok = bool(tickets) and tickets[0].get("status") == "ok"
            if not ok:
                logger.warning("Single push ticket not ok: %s", payload)
            return ok
    except Exception as exc:
        logger.warning("Single push notification failed: %s", exc)
        return False


def send_push_notifications(
    tokens: list[str],
    title: str,
    body: str,
    data: Optional[dict] = None,
    channel_id: str = "default",
    subtitle: Optional[str] = None,
    badge: Optional[int] = None,
) -> dict:
    """
    Send push notifications to a list of Expo push tokens.

    Batches tokens in groups of 100 (Expo API limit).
    Returns summary of sent/failed counts.
    """
    if not tokens:
        return {"sent": 0, "failed": 0}

    # Filter obviously invalid tokens early to avoid unnecessary API calls.
    valid_tokens = [
        t for t in tokens
        if isinstance(t, str)
        and (t.startswith("ExpoPushToken[") or t.startswith("ExponentPushToken["))
    ]
    invalid_local = len(tokens) - len(valid_tokens)
    if invalid_local:
        logger.warning("Push notifications: skipped %d locally invalid token(s)", invalid_local)

    if not valid_tokens:
        return {"sent": 0, "failed": len(tokens)}

    messages = []
    for token in valid_tokens:
        msg = {
            "to": token,
            "title": title,
            "body": body,
            "sound": "default",
            "channelId": channel_id,
            "priority": "high",
        }
        if subtitle:
            msg["subtitle"] = subtitle
        if badge is not None:
            msg["badge"] = badge
        if data:
            msg["data"] = data
        messages.append(msg)

    sent = 0
    failed = invalid_local
    chunk_size = 100
    stop_due_to_invalid_credentials = False

    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(messages), chunk_size):
            if stop_due_to_invalid_credentials:
                # App-level credentials issue: remaining tickets will fail too.
                failed += len(messages) - i
                break

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
                if isinstance(tickets, dict):
                    tickets = [tickets]

                invalid_credentials_hits = 0
                for ticket in tickets:
                    if ticket.get("status") == "ok":
                        sent += 1
                    else:
                        failed += 1
                        err = ticket.get("details", {}).get("error", "unknown")
                        if err == "InvalidCredentials":
                            invalid_credentials_hits += 1
                        else:
                            logger.warning("Push ticket error: %s", err)

                if invalid_credentials_hits:
                    logger.error(
                        "Expo Push InvalidCredentials (%d ticket(s) in chunk). "
                        "Check Expo project push credentials (FCM/APNs) for the app. "
                        "Suppressing repeated chunk logs for this send call.",
                        invalid_credentials_hits,
                    )
                    stop_due_to_invalid_credentials = True
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

        # Honor each user's notification preferences — drop anyone who
        # disabled the "News Notifications" toggle in Settings.
        try:
            from app.services.notification_prefs import filter_users_by_pref
            user_id_list = filter_users_by_pref(db, user_id_list, "newsNotifications")
        except Exception as e:
            logger.warning("notification pref filter failed (sending to all): %s", e)

        if not user_id_list:
            return {"sent": 0, "failed": 0, "reason": "opted_out"}

        # Get push tokens for those users
        tokens = (
            db.query(PushToken.token)
            .filter(PushToken.user_id.in_(user_id_list))
            .all()
        )
        token_list = [t[0] for t in tokens]

        if not token_list:
            return {"sent": 0, "failed": 0, "reason": "no_tokens"}

        symbols_str = ", ".join(symbols_upper[:3])  # cap at 3 symbols to keep title short
        if len(symbols_upper) > 3:
            symbols_str += f" +{len(symbols_upper) - 3}"
        title = f"📰 {symbols_str}"
        body = article_title[:180] if article_title else "New market announcement"
        data = {
            "newsId": article_id,
            "type": "news",
            "category": article_category,
            "symbols": symbols_upper,
        }

        return send_push_notifications(
            token_list, title, body, data,
            channel_id="news",
            subtitle="New Announcement",
        )
    except Exception as e:
        logger.warning("notify_users_for_article failed: %s", e)
        return {"sent": 0, "failed": 0, "error": str(e)}
    finally:
        db.close()
