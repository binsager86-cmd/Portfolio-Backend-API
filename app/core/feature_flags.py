import json

from app.core.database import query_one

FLAGS = {
    "enable_new_ai_pipeline": False,
    "enable_whale_signals_v2": True,
    "cache_news_feed": True,
}


def get_flags(user_id: int | None = None) -> dict:
    if user_id is not None:
        custom = query_one("SELECT flags FROM user_preferences WHERE user_id = ?", (user_id,))
        if custom:
            raw_flags = custom["flags"] if "flags" in custom else custom[0]
            try:
                parsed = json.loads(raw_flags) if isinstance(raw_flags, str) else dict(raw_flags)
                return {**FLAGS, **parsed}
            except Exception:
                return FLAGS.copy()
    return FLAGS.copy()
