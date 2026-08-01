import asyncio
import time

from app.api.v1.fundamental_legacy import _ensure_schema, delete_stock
from app.core.database import exec_sql, query_val
from app.core.security import TokenData


def _user(user_id: int = 1) -> TokenData:
    return TokenData(user_id=user_id, username=f"u{user_id}")


def test_delete_stock_succeeds_when_extraction_cache_table_is_missing(_init_test_db):
    _ensure_schema()
    exec_sql("DROP TABLE IF EXISTS extraction_cache")

    now = int(time.time())
    exec_sql(
        """
        INSERT INTO analysis_stocks (
            user_id, symbol, company_name, exchange, currency, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, "DELTEST", "Delete Test Co", "KSE", "KWD", now, now),
    )
    stock_id = query_val(
        "SELECT id FROM analysis_stocks WHERE user_id = ? AND symbol = ?",
        (1, "DELTEST"),
    )

    result = asyncio.run(delete_stock(int(stock_id), current_user=_user()))

    assert result["status"] == "ok"
    assert query_val("SELECT id FROM analysis_stocks WHERE id = ?", (stock_id,)) is None
