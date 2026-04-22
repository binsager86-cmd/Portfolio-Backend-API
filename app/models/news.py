"""
News Article model — persists Boursa Kuwait announcements for history browsing.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.core.database import Base


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source = Column(String(50), nullable=False, default="boursa_kuwait")
    category = Column(String(50), nullable=False, default="company_announcement")
    published_at = Column(DateTime, nullable=False, index=True)
    url = Column(Text, nullable=True)
    related_symbols = Column(Text, nullable=True)  # comma-separated
    sentiment = Column(String(20), nullable=False, default="neutral")
    impact = Column(String(20), nullable=False, default="informational")
    language = Column(String(5), nullable=False, default="en")
    is_verified = Column(Integer, nullable=False, default=1)  # boolean as int for SQLite compat
    attachments_json = Column(Text, nullable=True)  # JSON string
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    # Secondary dedupe fingerprint: md5(guid|link|title+pubdate+link). Indexed for
    # fast "have I seen this before?" checks when news_id is missing or unstable.
    content_hash = Column(String(32), nullable=True, index=True)
