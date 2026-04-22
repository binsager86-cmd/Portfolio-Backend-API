"""
Migrate news articles from local SQLite to production PostgreSQL
via the /api/v1/news/import endpoint.

Usage:
    python _migrate_news_to_prod.py
"""

import json
import os
import sqlite3
import time
from getpass import getpass
from pathlib import Path

import httpx

# ── Config ──────────────────────────────────────────────────────────
PROD_BASE = "https://backend-api-app-hfc2n.ondigitalocean.app"
LOCAL_DB = os.getenv(
    "LOCAL_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "dev_portfolio.db"),
)

PROD_USER = os.getenv("PROD_NEWS_IMPORT_USER", "")
PROD_PASS = os.getenv("PROD_NEWS_IMPORT_PASSWORD", "")

BATCH_SIZE = 500  # articles per request
# ────────────────────────────────────────────────────────────────────


def get_token(client: httpx.Client, username: str, password: str) -> str:
    resp = client.post(
        f"{PROD_BASE}/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def read_local_articles() -> list[dict]:
    conn = sqlite3.connect(LOCAL_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT news_id, title, summary, source, category, "
        "published_at, url, related_symbols, sentiment, impact, "
        "language, is_verified, attachments_json FROM news_articles"
    )
    articles = []
    for row in cur:
        attachments = []
        if row["attachments_json"]:
            try:
                attachments = json.loads(row["attachments_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        symbols = row["related_symbols"].split(",") if row["related_symbols"] else []

        articles.append({
            "id": row["news_id"],
            "title": row["title"] or "",
            "summary": row["summary"],
            "source": row["source"] or "boursa_kuwait",
            "category": row["category"] or "company_announcement",
            "publishedAt": row["published_at"],
            "url": row["url"],
            "relatedSymbols": symbols,
            "sentiment": row["sentiment"] or "neutral",
            "impact": row["impact"] or "informational",
            "language": row["language"] or "en",
            "isVerified": bool(row["is_verified"]),
            "attachments": attachments,
        })
    conn.close()
    return articles


def main():
    username = PROD_USER.strip() or input("Production username/email: ").strip()
    password = PROD_PASS or getpass("Production password: ")

    if not username or not password:
        raise RuntimeError(
            "Missing credentials. Set PROD_NEWS_IMPORT_USER and "
            "PROD_NEWS_IMPORT_PASSWORD or provide interactive input."
        )

    articles = read_local_articles()
    print(f"Local articles: {len(articles)}")

    with httpx.Client(timeout=120.0) as client:
        token = get_token(client, username, password)
        headers = {"Authorization": f"Bearer {token}"}
        print("Authenticated with production")

        total_inserted = 0
        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE

            try:
                resp = client.post(
                    f"{PROD_BASE}/api/v1/news/import",
                    json={"articles": batch},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                total_inserted += data.get("inserted", 0)
                print(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"sent={len(batch)}, inserted={data.get('inserted', 0)}, "
                    f"total_db={data.get('total', '?')}"
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    # Token expired, re-auth
                    print("  Token expired, re-authenticating...")
                    token = get_token(client, username, password)
                    headers = {"Authorization": f"Bearer {token}"}
                    # Retry this batch
                    resp = client.post(
                        f"{PROD_BASE}/api/v1/news/import",
                        json={"articles": batch},
                        headers=headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    total_inserted += data.get("inserted", 0)
                    print(
                        f"  Batch {batch_num}/{total_batches} (retry): "
                        f"inserted={data.get('inserted', 0)}, "
                        f"total_db={data.get('total', '?')}"
                    )
                else:
                    print(f"  Batch {batch_num} FAILED: {e}")
            except Exception as e:
                print(f"  Batch {batch_num} ERROR: {e}")

            # Small delay to avoid overwhelming production
            time.sleep(0.5)

        print(f"\n✅  Migration complete: {total_inserted} new articles inserted")


if __name__ == "__main__":
    main()
