"""
Push Notification Service — sends push notifications via Expo Push API.

Uses the Expo Push Notification service (https://exp.host/--/api/v2/push/send)
to deliver notifications to registered devices.
"""

import logging
from pathlib import Path
from typing import Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def _send_fcm_notification(
    token: str,
    title: str,
    body: str,
    data: Optional[dict] = None,
    channel_id: str = "default",
) -> bool:
    """Send a notification through Firebase Admin when credentials are configured."""
    service_account_file = get_settings().FIREBASE_SERVICE_ACCOUNT_FILE.strip()
    if not service_account_file:
        logger.warning("FCM delivery skipped: FIREBASE_SERVICE_ACCOUNT_FILE is not configured")
        return False

    credential_path = Path(service_account_file)
    if not credential_path.is_file():
        logger.error("FCM delivery skipped: Firebase service-account file does not exist")
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, messaging

        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(credentials.Certificate(str(credential_path)))

        message = messaging.Message(
            token=token,
            notification=messaging.Notification(title=title, body=body),
            data={key: str(value) for key, value in (data or {}).items()},
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(channel_id=channel_id),
            ),
        )
        messaging.send(message)
        return True
    except Exception as exc:
        logger.warning("FCM notification failed: %s", exc)
        return False


async def send_push_notification(
    token: str,
    title: str,
    body: str,
    data: Optional[dict] = None,
    sound: str = "default",
    priority: str = "high",
    category: Optional[str] = None,
    android: Optional[dict] = None,
    token_provider: str = "expo",
) -> bool:
    """Send a single rich push notification via Expo Push API or FCM."""
    if not token:
        return False

    if token_provider == "fcm":
        channel_id = android.get("channelId", "default") if isinstance(android, dict) else "default"
        return _send_fcm_notification(token, title, body, data, channel_id)

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
    tokens: list[tuple[str, str]],
    title: str,
    body: str,
    data: Optional[dict] = None,
    channel_id: str = "default",
    subtitle: Optional[str] = None,
    badge: Optional[int] = None,
) -> dict:
    """
    Send push notifications to Expo and FCM device tokens.

    Batches tokens in groups of 100 (Expo API limit).
    Returns summary of sent/failed counts.
    """
    if not tokens:
        return {"sent": 0, "failed": 0}

    expo_tokens = [token for token, provider in tokens if provider == "expo"]
    fcm_tokens = [token for token, provider in tokens if provider == "fcm"]

    sent = 0
    failed = 0
    for token in fcm_tokens:
        if _send_fcm_notification(token, title, body, data, channel_id):
            sent += 1
        else:
            failed += 1

    if not expo_tokens:
        return {"sent": sent, "failed": failed}

    messages = []
    for token in expo_tokens:
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
            db.query(PushToken.token, PushToken.token_provider)
            .filter(PushToken.user_id.in_(user_id_list))
            .all()
        )
        token_list = [(token, provider) for token, provider in tokens]

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
