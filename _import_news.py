"""Bulk import all Boursa Kuwait announcements into the news DB."""
import sys, time
sys.path.insert(0, '.')

import httpx
from datetime import datetime
from app.core.database import SessionLocal
from app.models.news import NewsArticle

BOURSA_API = "https://www.boursakuwait.com.kw/data-api/client-services"

TITLE_CAT = {
    # English
    "Dividend Distribution": "dividend",
    "Timetable of Corporate Actions": "dividend",
    "Financial Results": "financial",
    "Fund Financial Statement": "financial",
    "Credit Rating Disclosure": "financial",
    "Monthly Information": "financial",
    "Transcript of the Analysts Conference": "earnings",
    "Annual General Meeting Outcome": "regulatory",
    "General Assembly Meeting": "regulatory",
    "Postponed General Assembly Meeting": "regulatory",
    "General Assembly Meeting Date Amendment": "regulatory",
    "Unit Holders Assembly Meeting": "regulatory",
    "Board of Directors Meeting": "regulatory",
    "Board of Directors Meeting Results": "regulatory",
    "Rescheduling Board of Directors Meeting": "regulatory",
    "Board of Directors Meeting Date Amendment": "regulatory",
    "Formation of Board of Directors": "regulatory",
    "Election of New Board of Directors": "regulatory",
    "Board of Directors Membership Change": "regulatory",
    "Judicial Decision Disclosure": "regulatory",
    "Disclosure Amendment": "regulatory",
    "CMA approval to deal in treasury shares": "regulatory",
    "Disclosure regarding an unusual trade": "regulatory",
    "Official Holidays": "regulatory",
    "Market Maker Announcement": "market_news",
    "Off Market Trades": "market_news",
    "Material Information Disclosure": "company_announcement",
    "Supplementary Disclosure": "company_announcement",
    "*Other*": "company_announcement",
    # Arabic
    "البدء بتوزيع توزيع الارباح": "dividend",
    "الجدول الزمني لاستحقاقات الأسهم": "dividend",
    "النتائج المالية": "financial",
    "البيانات المالية للصندوق": "financial",
    "افصاح بشأن التصنيف الائتماني": "financial",
    "المعلومات الشهرية": "financial",
    "محضر مؤتمر المحللين": "earnings",
    "نتائج اجتماع الجمعية العامة": "regulatory",
    "اجتماع الجمعية العامة": "regulatory",
    "تأجيل الجمعية العامة": "regulatory",
    "جمعية حملة وحدات": "regulatory",
    "اجتماع مجلس الإدارة": "regulatory",
    "نتائج اجتماع مجلس الادارة": "regulatory",
    "تأجيل موعد اجتماع مجلس الادارة": "regulatory",
    "تشكيل مجلس الادارة": "regulatory",
    "فتح باب الترشيح لعضوية مجلس الادارة": "regulatory",
    "تغيير في مجلس الادارة": "regulatory",
    "نتائج اجتماع الهيئة الادارية": "regulatory",
    "اجتماع الهيئة الادارية": "regulatory",
    "افصاح بشأن الدعاوى والاحكام": "regulatory",
    "ايضاح بشأن التداول غير الاعتيادي": "regulatory",
    "افصاح تصحيحي": "regulatory",
    "تنفيذ جبري": "regulatory",
    "موافقة الهيئة على التعامل بأسهم الخزينة": "regulatory",
    "اعلان صانع السوق": "market_news",
    "الصفقات المتفق عليها": "market_news",
    "افصاح معلومات جوهرية": "company_announcement",
    "افصاح مكمل": "company_announcement",
    "*إعلانات اخرى*": "company_announcement",
}


def parse_date(raw):
    try:
        return datetime.strptime(raw[:14], "%Y%m%d%H%M%S")
    except Exception:
        return datetime.utcnow()


def assess_impact(cat):
    if cat in ("earnings", "dividend"):
        return "high"
    if cat in ("financial", "regulatory"):
        return "medium"
    return "informational"


db = SessionLocal()
existing_ids = set(r[0] for r in db.query(NewsArticle.news_id).all())
print(f"Already in DB: {len(existing_ids)}")

for lang_code, lang_label in [("E", "en"), ("A", "ar")]:
    print(f"\nFetching RT=3516 ({lang_label})...")
    t0 = time.time()
    r = httpx.get(BOURSA_API, params={"RT": "3516", "L": lang_code}, timeout=120, follow_redirects=True)
    data = r.json()
    print(f"Got {len(data)} items in {time.time() - t0:.1f}s")

    inserted = 0
    for item in data:
        nid = str(item.get("NewsId", ""))
        if not nid or nid in existing_ids:
            continue

        title = (item.get("Title") or "").strip()
        title_type = (item.get("TitleTypeDesc") or "").strip()
        ticker = (item.get("DisplayTicker") or "").strip()
        cat = TITLE_CAT.get(title_type, "company_announcement")

        article = NewsArticle(
            news_id=nid,
            title=title,
            summary=title,
            source="boursa_kuwait",
            category=cat,
            published_at=parse_date(str(item.get("PostedDate", ""))),
            url=None,
            related_symbols=ticker or None,
            sentiment="neutral",
            impact=assess_impact(cat),
            language=lang_label,
            is_verified=1,
            attachments_json=None,
            fetched_at=datetime.utcnow(),
        )
        db.add(article)
        existing_ids.add(nid)
        inserted += 1

        if inserted % 1000 == 0:
            db.commit()
            print(f"  {inserted} committed...")

    db.commit()
    print(f"Inserted {inserted} {lang_label} articles")

# Stats
from collections import Counter
from sqlalchemy import func

total = db.query(NewsArticle).count()
print(f"\n=== FINAL STATS ===")
print(f"Total articles: {total}")

cats = Counter(r[0] for r in db.query(NewsArticle.category).all())
print(f"Categories: {dict(cats)}")

for lang in ["en", "ar"]:
    lang_cats = {}
    for r in db.query(NewsArticle.category, func.count()).filter(
        NewsArticle.language == lang
    ).group_by(NewsArticle.category).all():
        lang_cats[r[0]] = r[1]
    print(f"  {lang}: {lang_cats}")

oldest = db.query(func.min(NewsArticle.published_at)).scalar()
newest = db.query(func.max(NewsArticle.published_at)).scalar()
print(f"Date range: {oldest} to {newest}")

db.close()
print("Done!")
