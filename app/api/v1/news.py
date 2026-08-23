"""
News API v1 — Proxies Boursa Kuwait announcements and persists history.

Categories:
  • company_announcement — General company announcements & disclosures
  • financial            — Financial statements, reports, balance sheets
  • dividend             — Dividend declarations, payouts, ex-dates
  • earnings             — Earnings releases, profit results, EPS
  • market_news          — Market-wide updates, index moves, trading halts
  • regulatory           — CMA / regulatory notices, compliance, AGM/EGM

Endpoints:
  GET  /feed        — paginated live feed with optional symbol/category filtering
  GET  /history     — paginated stored history with date-range filtering
  GET  /item/{id}   — single news item by NewsId
  GET  /sources     — list available sources
  POST /fetch-all   — bulk-fetch all available Boursa announcements and persist
"""

import json
import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime
from html import unescape
from hashlib import md5
from typing import Any, Optional
from urllib.parse import quote, urldefrag, urljoin, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from kombu.exceptions import OperationalError

from app.api.deps import require_admin

from app.core.cache import cache_key, news_cache, get_cached, set_cached
from app.core.config import get_settings
from app.core.http_client import fetch_with_retry
import httpx  # kept for httpx.Client used in _bg_fetch_live (sync background thread)

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.security import TokenData
from app.models.news import NewsArticle

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/news", tags=["News"])


class NewsImportRequest(BaseModel):
    articles: list[dict[str, Any]] = Field(default_factory=list, max_length=10_000)

BOURSA_API = "https://www.boursakuwait.com.kw/data-api/client-services"
BOURSA_RSS_BASE = "https://rss.boursakuwait.com.kw"

# In-memory TTL cache for raw Boursa API responses.
# Delegates to app.core.cache.news_cache (15-min TTL, max 1000 entries).
# Key: (rt_code, lang_code)  →  Value: list[dict]
# Aliased for backward-compat with the X-Cache-Status check in news_feed.
_boursa_cache = news_cache

# Compatibility state used by legacy in-process helper paths.
_news_jobs: dict[str, dict] = {}

# Boursa Kuwait data-api request types for news / announcements
_BOURSA_RT_CODES = [
    "3507",  # General news & disclosures (current day only)
    "3508",  # Today's summary / company disclosures
]

# Full history endpoint — RT=3516 returns ALL announcements (~16k+ items since 2016)
# covering 173+ tickers.  Fields: NewsTypeDesc, TitleTypeDesc, Title,
# DisplayTicker, PostedDate, NewsId, RepeatedDate.
_BOURSA_HISTORY_RT = "3516"

_RSS_LIVE_FEEDS = ("FeedAllDetail.aspx",)
_PDF_HREF_RE = re.compile(r'''(?:href|src)=["']([^"']+\.pdf(?:\?[^"']*)?)["']''', re.IGNORECASE)
_resolved_document_cache: dict[str, str] = {}

# ── Arabic display name → English ticker mapping ─────────────────
# Boursa Kuwait returns Arabic company names as DisplayTicker for AR articles.
# We normalise to English tickers so portfolio symbol matching works
# regardless of the user's language.
_AR_DISPLAY_TO_EN_TICKER: dict[str, str] = {
    "الجزيرة": "JAZEERA", "مخازن": "MKHZN", "اهلي": "ABK",
    "الدولي": "KIB", "التجارية": "ALTIJARIA", "بيت الطاقة": "ENERGYH",
    "قيوين ا": "QIC", "بنك بوبيان": "BOUBYAN", "الامتياز": "ALIMTIAZ",
    "بنك وربة": "WARBABANK", "صالحية": "SRE", "كابلات": "CABLE",
    "زين": "ZAIN", "بيتك": "KFH", "يوباك": "UPAC", "مشاريع": "KPROJ",
    "صناعات": "NIND", "المباني": "MABANEE", "بيوت": "BEYOUT",
    "ترولي": "TROLLEY", "برقان": "BURG", "مزايا": "MAZAYA",
    "جي تي سي": "JTC", "أرزان": "ARZAN", "كويتية": "KINV",
    "ايفا فنادق": "IFAHR", "سفن": "SHIP", "جي اف اتش": "GFH",
    "وطني": "NBK", "ألف طاقة": "ALFTAQA", "ميزان": "MEZZAN",
    "سنام": "SANAM", "وطنية": "NRE", "ب ك تأمين": "BKIKWT",
    "الغانم": "ALG", "أعيان": "AAYAN", "متحدة": "URC",
    "خليج ب": "GBK", "المتكاملة": "INTEGRATED", "معادن": "MRC",
    "شمال الزور": "AZNOULA", "كامكو": "KAMCO", "المركز": "MARKAZ",
    "بوبيان ب": "BPCC", "استثمارات": "NINV", "تجاري": "CBK",
    "نور": "NOOR", "أس تي سي": "STC", "هيومن سوفت": "HUMANSOFT",
    "البورصة": "BOURSA", "عقارات ك": "KRE", "إنوفست": "INOVEST",
    "ألافكو": "ALAFCO", "دبي الاولى": "FIRSTDUBAI", "المتحد": "ALMUTAHED",
    "اسمنت ابيض": "RKWC", "اهلي متحد": "AUB", "لاند": "LAND",
    "فجيرة ا": "FCEM", "القرين": "ALQURAIN", "المدينة": "ALMADINA",
    "شارقة ا": "SCEM", "اسمنت خليج": "GCEM", "صيرفة": "EXCH",
    "اغذية": "NRE", "البناء": "SCEM", "إياس": "ALSALAM",
    "فلكس": "GFH", "نفائس": "RASIYAT", "الشامل": "FIRSTDUBAI",
    "راسيات": "RASIYAT", "ثريا": "THURAYA", "وربة كبيتل": "WARBACAP",
    "ديجتس": "DIGITUS", "كفيك": "KFIC", "فنادق": "KHOT",
    "امتيازات": "GFC", "أرجان": "ARGAN", "سينما": "KCIN",
    "م الأعمال": "KBT", "اسمنت": "KCEM", "إنجازات": "INJAZZAT",
    "يونيكاب": "UNICAP", "أصول": "OSOUL", "المنار": "ALMANAR",
    "المعدات": "EQUIPMENT", "خليج ت": "GINS", "بيان": "BAYANINV",
    "أسس": "OSOS", "كويت ت": "KINS", "فالمور": "VALMORE",
    "عمار": "AMAR", "التقدم": "ATC", "استهلاكيه": "NCCI",
    "عقار": "AQAR", "سكب ك": "KFOUC", "آسيا": "ASIYA",
    "الإماراتية": "EMIRATES", "وطنية م ب": "NICBM", "الأولى": "ALOLA",
    "البيت": "SECH", "نابيسكو": "NAPESCO", "منازل": "MANAZEL",
    "أركان": "ARKAN", "ساحل": "COAST", "كميفك": "KMEFIC",
    "فيوتشر كيد": "FUTUREKID", "بترولية": "IPG", "سنرجي": "SENERGY",
    "ورقية": "PAPER", "الخليجي": "GIH", "بورتلاند": "PCEM",
}

# ── NewsType code → category mapping ─────────────────────────────
# Boursa uses numeric NewsType codes. Mapping from official Boursa Kuwait table.
_NEWS_TYPE_MAP: dict[str, str] = {
    # General Announcement
    "1":   "company_announcement",   # General Assembly Meeting
    "2":   "company_announcement",   # General Assembly Meeting Date Amendment
    "3":   "company_announcement",   # Postponed General Assembly Meeting
    "7":   "company_announcement",   # Board of Directors Meeting
    "8":   "company_announcement",   # Board of Directors Meeting Results
    "10":  "company_announcement",   # Rescheduling Board of Directors Meeting
    "11":  "company_announcement",   # Board of Directors Meeting Date Amendment
    "12":  "company_announcement",   # Board of Directors Membership Change
    "13":  "company_announcement",   # Formation of Board of Directors
    "14":  "company_announcement",   # Election of New Board of Directors
    "15":  "company_announcement",   # Board of Directors Resignation
    "19":  "company_announcement",   # Unit Holders Assembly Meeting
    "21":  "company_announcement",   # Unit Holders Assembly Meeting Date Amendment
    "22":  "company_announcement",   # Postponed Unit Holders Assembly Meeting
    "31":  "company_announcement",   # Company Name Change
    "40":  "company_announcement",   # Other
    "89":  "company_announcement",   # Unit Holders Assembly Meeting
    "90":  "company_announcement",   # Unit Holders Assembly Meeting Date Amendment
    "91":  "company_announcement",   # Postponed Unit Holders Assembly Meeting
    "92":  "company_announcement",   # Unit Holders Assembly Outcome
    "93":  "company_announcement",   # Committee Members Meeting
    "94":  "company_announcement",   # Committee Members Meeting Results
    "95":  "company_announcement",   # Rescheduling Committee Members Meeting
    "96":  "company_announcement",   # Committee Members Meeting Date Amendment
    "97":  "company_announcement",   # Committee Members Membership Change
    "98":  "company_announcement",   # Committee Members Resignation
    "101": "earnings",               # Transcript of the Analysts Conference
    "110": "company_announcement",   # Annual General Meeting Outcome
    "130": "company_announcement",   # Fund Name Change
    # Dividends
    "4":   "dividend",               # Dividend Distribution
    "5":   "dividend",               # Dividend Distribution Date Amendment
    "20":  "dividend",               # Dividend Distribution
    "76":  "dividend",               # Trading without Bonus Shares
    "83":  "dividend",               # Cum AGM Date
    "84":  "dividend",               # Record AGM Date
    "85":  "dividend",               # Cum Dividend Date
    "86":  "dividend",               # Ex Dividend Date
    "87":  "dividend",               # Record Date
    "88":  "dividend",               # Payment Date
    "120": "dividend",               # Timetable of Corporate Actions
    # Financials
    "9":   "financial",              # Financial Results
    "17":  "financial",              # Monthly Information
    "18":  "financial",              # Fund Financial Statement
    "25":  "financial",              # Credit Rating Disclosure
    # Regulatory
    "16":  "regulatory",             # Board Recommendation for Voluntary Delisting
    "23":  "regulatory",             # Disclosure regarding an unusual trade
    "24":  "regulatory",             # Judicial Decision Disclosure
    "30":  "regulatory",             # CMA Voluntary Delisting Approval
    "34":  "regulatory",             # Material Information Disclosure
    "35":  "regulatory",             # Supplementary Disclosure
    "36":  "regulatory",             # Disclosure Amendment
    "60":  "regulatory",             # Capital Increase Call
    "99":  "regulatory",             # Committee Recommendation for Voluntary Delisting
    "132": "regulatory",             # CMA approval to deal in treasury shares
    "133": "regulatory",             # Sustainability Report
    "134": "regulatory",             # Interested person disclosure form
    "135": "regulatory",             # Change in interested person disclosure form
    "136": "regulatory",             # Group disclosure form
    "137": "regulatory",             # 5% ownership disclosure form
    "139": "regulatory",             # Corporate insiders disclosure form
    "140": "regulatory",             # Lawsuits and court judgments disclosure form
    "141": "financial",              # Credit rating disclosure form
    "143": "financial",              # Board meeting and financial results
    "144": "company_announcement",   # General Assembly meeting
    "145": "regulatory",             # Material information disclosure form
    "146": "financial",              # Public investment fund monthly information
    # Market
    "32":  "market_news",            # Ticker Name Change
    "39":  "market_news",            # Company Delisting Date
    "41":  "market_news",            # Fund Delisting Date
    "71":  "market_news",            # Suspension Date
    "72":  "market_news",            # Activation Date
    "74":  "market_news",            # Trades of 5% or Above
    "75":  "market_news",            # Trading after Capital Increase
    "77":  "market_news",            # Trading after Capital Decrease
    "78":  "market_news",            # Official Holidays
    "79":  "market_news",            # Listing in Regular Market
    "200": "market_news",            # Off Market Trades
    "300": "market_news",            # Market Maker Announcement
}

# ── Keyword-based category detection ─────────────────────────────
# Fallback when NewsType code is missing. Checked against title text.

_DIVIDEND_KEYWORDS = [
    # English
    "dividend distribution", "dividend", "bonus shares", "payout",
    "ex-date", "ex dividend", "record date", "payment date",
    "cum agm date", "record agm date", "cum dividend",
    "timetable of corporate actions", "corporate action confirmation",
    "corporate action schedule", "trading without bonus shares",
    # Arabic
    "توزيعات", "توزيع أرباح", "أسهم منحة", "تاريخ الاستحقاق",
    "تاريخ التوزيع", "تاريخ تداول السهم دون", "جائزة السهم",
    "الجدول الزمني لاستحقاقات", "البدء بتوزيع",
    "تغيير موعد البدء بتوزيع",
]

_EARNINGS_KEYWORDS = [
    # English — only Analysts Conference
    "transcript of the analysts conference",
    "analysts conference",
    # Arabic
    "محضر مؤتمر المحللين", "مؤتمر المحللين",
]

_FINANCIAL_KEYWORDS = [
    # English
    "financial results", "financial statement", "fund financial statement",
    "monthly information", "credit rating disclosure",
    "balance sheet", "income statement", "annual report",
    "interim report", "quarterly report",
    "audited", "consolidated statement",
    # Arabic
    "النتائج المالية", "البيانات المالية", "المعلومات الشهرية",
    "التصنيف الائتماني", "ميزانية", "قائمة الدخل",
    "تقرير سنوي",
]

_REGULATORY_KEYWORDS = [
    # English
    "disclosure regarding an unusual trade", "judicial decision",
    "material information disclosure", "supplementary disclosure",
    "disclosure amendment", "capital increase call",
    "voluntary delisting", "cma approval", "cma voluntary",
    "sustainability report",
    # Arabic
    "إفصاح بشأن التداول غير الاعتيادي", "الدعاوى والاحكام",
    "افصاح معلومات جوهرية", "افصاح مكمل", "افصاح تصحيحي",
    "استدعاء زيادة رأس المال", "الانسحاب الاختياري",
    "موافقة الهيئة على", "تقرير الاستدامة",
]

_MARKET_KEYWORDS = [
    # English
    "ticker name change", "delisting date", "suspension date",
    "activation date", "trades of 5%", "obligatory executions",
    "trading after capital increase", "trading after capital decrease",
    "official holidays", "listing in regular market",
    "off-market trade", "off market trade",
    # Arabic
    "تغيير رمز تداول", "موعد السحاب", "موعد انسحاب",
    "وقف التداول", "إعادة التداول", "تنفيذ صفقة",
    "تنفيذ جبري", "تداول أسهم الشركة بعد",
    "العطل الرسمية", "إدراج الشركة في السوق",
    "صفقة متفق عليها",
]


# ── TitleTypeDesc → category mapping (RT=3516) ───────────────────
# RT=3516 provides TitleTypeDesc directly. Mapped from official Boursa table.
_TITLE_TYPE_CATEGORY: dict[str, str] = {
    # ── English ──
    # General Announcement
    "General Assembly Meeting":                          "company_announcement",
    "General Assembly Meeting Date Amendment":            "company_announcement",
    "Postponed General Assembly Meeting":                 "company_announcement",
    "Board of Directors Meeting":                         "company_announcement",
    "Board of Directors Meeting Results":                 "company_announcement",
    "Rescheduling Board of Directors Meeting":            "company_announcement",
    "Board of Directors Meeting Date Amendment":          "company_announcement",
    "Board of Directors Membership Change":               "company_announcement",
    "Formation of Board of Directors":                    "company_announcement",
    "Election of New Board of Directors":                 "company_announcement",
    "Board of Directors Resignation":                     "company_announcement",
    "Unit Holders Assembly Meeting":                      "company_announcement",
    "Unit Holders Assembly Meeting Date Amendment":       "company_announcement",
    "Postponed Unit Holders Assembly Meeting":            "company_announcement",
    "Unit Holders Assembly Outcome":                      "company_announcement",
    "Company Name Change":                                "company_announcement",
    "Other":                                              "company_announcement",
    "Committee Members Meeting":                          "company_announcement",
    "Committee Members Meeting Results":                  "company_announcement",
    "Rescheduling Committee Members Meeting":             "company_announcement",
    "Committee Members Meeting Date Amendment":           "company_announcement",
    "Committee Members Membership Change":                "company_announcement",
    "Committee Members Resignation":                      "company_announcement",
    "Annual General Meeting Outcome":                     "company_announcement",
    "Fund Name Change":                                   "company_announcement",
    # Analysts Conference (earnings tab)
    "Transcript of the Analysts Conference":              "earnings",
    # Dividends
    "Dividend Distribution":                              "dividend",
    "Dividend Distribution Date Amendment":               "dividend",
    "Trading without Bonus Shares":                       "dividend",
    "Cum AGM Date":                                       "dividend",
    "Record AGM Date":                                    "dividend",
    "Cum Dividend Date":                                  "dividend",
    "Ex Dividend Date":                                   "dividend",
    "Record Date":                                        "dividend",
    "Payment Date":                                       "dividend",
    "Timetable of Corporate Actions":                     "dividend",
    # Financials
    "Financial Results":                                  "financial",
    "Monthly Information":                                "financial",
    "Fund Financial Statement":                           "financial",
    "Credit Rating Disclosure":                           "financial",
    # Regulatory
    "Board of Directors Recommendation for Voluntary Delisting": "regulatory",
    "Disclosure regarding an unusual trade":              "regulatory",
    "Judicial Decision Disclosure":                       "regulatory",
    "Material Information Disclosure":                    "regulatory",
    "Supplementary Disclosure":                           "regulatory",
    "Disclosure Amendment":                               "regulatory",
    "CMA Voluntary Delisting Approval":                   "regulatory",
    "Capital Increase Call":                              "regulatory",
    "Committee Members Recommendation for Voluntary Delisting": "regulatory",
    "CMA approval to deal in treasury shares":           "regulatory",
    "Sustainability Report":                              "regulatory",
    # Market
    "Ticker Name Change":                                 "market_news",
    "Company Delisting Date":                             "market_news",
    "Fund Delisting Date":                                "market_news",
    "Suspension Date":                                    "market_news",
    "Activation Date":                                    "market_news",
    "Obligatory Executions":                              "market_news",
    "Trades of 5% or Above":                              "market_news",
    "Trading after Capital Increase":                     "market_news",
    "Trading after Capital Decrease":                     "market_news",
    "Official Holidays":                                  "market_news",
    "Listing in Regular Market":                          "market_news",
    "Off-Market Trade":                                   "market_news",
    "Market Maker Announcement":                          "market_news",

    # ── Arabic ──
    # General Announcement
    "اجتماع الجمعية العامة":                              "company_announcement",
    "تغيير موعد الجمعية العامة":                          "company_announcement",
    "تأجيل الجمعية العامة":                               "company_announcement",
    "اجتماع مجلس الادارة":                               "company_announcement",
    "اجتماع مجلس الإدارة":                               "company_announcement",
    "نتائج اجتماع مجلس الادارة":                         "company_announcement",
    "تأجيل موعد اجتماع مجلس الادارة":                    "company_announcement",
    "تغيير موعد اجتماع مجلس الادارة":                    "company_announcement",
    "تغيير في مجلس الادارة":                             "company_announcement",
    "تشكيل مجلس الادارة":                                "company_announcement",
    "فتح باب الترشيح لعضوية مجلس الادارة":               "company_announcement",
    "استقالة مجلس الادارة":                              "company_announcement",
    "جمعية حملة وحدات":                                   "company_announcement",
    "تغيير موعد جمعية حملة الوحدات":                     "company_announcement",
    "تأجيل موعد جمعية حملة الوحدات":                     "company_announcement",
    "نتائج اجتماع جمعية حملة الوحدات":                   "company_announcement",
    "انعقاد جمعية حملة الوحدات":                         "company_announcement",
    "تأجيل جمعية حملة الوحدات المؤجلة":                  "company_announcement",
    "تغيير اسم الشركة":                                  "company_announcement",
    "اجتماع الهيئة الادارية":                            "company_announcement",
    "اجتماع الهيئة الإدارية":                            "company_announcement",
    "نتائج اجتماع الهيئة الادارية":                      "company_announcement",
    "نتائج اجتماع الهيئة الإدارية":                      "company_announcement",
    "تأجيل موعد اجتماع الهيئة الادارية":                 "company_announcement",
    "تأجيل موعد اجتماع الهيئة الإدارية":                 "company_announcement",
    "تغيير موعد اجتماع الهيئة الادارية":                 "company_announcement",
    "تغيير موعد اجتماع الهيئة الإدارية":                 "company_announcement",
    "تغيير في الهيئة الادارية":                          "company_announcement",
    "تغيير في الهيئة الإدارية":                          "company_announcement",
    "استقالة الهيئة الادارية":                           "company_announcement",
    "استقالة الهيئة الإدارية":                           "company_announcement",
    "نتائج اجتماع الجمعية العامة":                       "company_announcement",
    "تغيير اسم الصندوق":                                 "company_announcement",
    "*إعلانات اخرى*":                                    "company_announcement",
    # Analysts Conference (earnings tab)
    "محضر مؤتمر المحللين":                               "earnings",
    # Dividends
    "البدء بتوزيع توزيع الارباح":                        "dividend",
    "تغيير موعد البدء بتوزيع الارباح":                   "dividend",
    "توزيع أرباح":                                       "dividend",
    "تداول أسهم الشركة بدون أسهم المنحة":                "dividend",
    "جائزة السهم لحضور الجمعية":                         "dividend",
    "تاريخ الاستحقاق لحضور الجمعية":                     "dividend",
    "تاريخ جائزة السهم":                                 "dividend",
    "تاريخ تداول السهم دون الاستحقاق":                   "dividend",
    "تاريخ الاستحقاق":                                   "dividend",
    "تاريخ التوزيع":                                     "dividend",
    "الجدول الزمني لاستحقاقات الأسهم":                   "dividend",
    # Financials
    "النتائج المالية":                                   "financial",
    "المعلومات الشهرية":                                 "financial",
    "البيانات المالية للصندوق":                          "financial",
    "البيانات المالية المصدوق":                          "financial",
    "افصاح بشأن التصنيف الائتماني":                     "financial",
    # Regulatory
    "توصية مجلس الادارة بالانسحاب الاختياري من البورصة": "regulatory",
    "ايضاح بشأن التداول غير الاعتيادي":                 "regulatory",
    "إفصاح بشأن التداول غير الاعتيادي":                 "regulatory",
    "افصاح بشأن الدعاوى والاحكام":                      "regulatory",
    "إفصاح بشأن الدعاوى والاحكام":                      "regulatory",
    "افصاح معلومات جوهرية":                              "regulatory",
    "افصاح مكمل":                                        "regulatory",
    "افصاح تصحيحي":                                      "regulatory",
    "موافقة هيئة أسواق المال بالانسحاب الاختياري من البورصة": "regulatory",
    "استدعاء زيادة رأس المال":                           "regulatory",
    "توصية الهيئة الإدارية بالانسحاب الاختياري من البورصة": "regulatory",
    "توصية الهيئة الادارية بالانسحاب الاختياري من البورصة": "regulatory",
    "موافقة الهيئة على التعامل بأسهم الخزينة":          "regulatory",
    "تقرير الاستدامة":                                   "regulatory",
    # Market
    "تغيير رمز تداول الشركة":                            "market_news",
    "موعد انسحاب الشركة من البورصة":                     "market_news",
    "موعد السحاب الشركة من البورصة":                     "market_news",
    "موعد انسحاب الصندوق من البورصة":                    "market_news",
    "وقف التداول على أسهم الشركة":                       "market_news",
    "إعادة التداول على أسهم الشركة":                     "market_news",
    "تنفيذ جبري":                                        "market_news",
    "تنفيذ صفقة 5% أو أكثر":                            "market_news",
    "تداول أسهم الشركة بعد زيادة رأس المال":             "market_news",
    "تداول أسهم الشركة بعد تخفيض رأس المال":             "market_news",
    "العطل الرسمية":                                     "market_news",
    "إدراج الشركة في السوق الرسمي":                      "market_news",
    "اعلان صانع السوق":                                  "market_news",
    "الصفقات المتفق عليها":                              "market_news",
    "صفقة متفق عليها":                                   "market_news",
}


def _parse_date(raw: str) -> str:
    """Convert Boursa date '20260406144720' → ISO 8601."""
    try:
        dt = datetime.strptime(raw[:14], "%Y%m%d%H%M%S")
        return dt.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _parse_rss_date(raw: str) -> str:
    try:
        return parsedate_to_datetime(raw).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _rss_urls(lang_code: str) -> list[str]:
    prefix = "/A/rss" if lang_code in ("A", "ar") else "/rss"
    return [f"{BOURSA_RSS_BASE}{prefix}/{feed}" for feed in _RSS_LIVE_FEEDS]


def _xml_text(element: ET.Element, name: str) -> str:
    found = element.find(name)
    return (found.text or "").strip() if found is not None else ""


def _extract_news_id(link: str, document_url: str, title: str, published_at: str) -> str:
    for candidate in (link, document_url):
        if not candidate:
            continue
        _, fragment = urldefrag(candidate)
        if fragment:
            return fragment.strip()
    return md5("|".join([title, document_url, published_at]).encode("utf-8")).hexdigest()


def _extract_symbol_from_title(title: str) -> str:
    prefix = title.split(" - ", 1)[0].strip().upper()
    if not prefix or any(ch.isspace() for ch in prefix):
        return ""
    return prefix if re.fullmatch(r"[A-Z0-9]{2,16}", prefix) else ""


def _resolve_boursa_document_url(document_url: str) -> str:
    """Return the direct PDF URL for Boursa disclosure HTML pages when available."""
    if not document_url:
        return ""
    if document_url.lower().split("?", 1)[0].endswith(".pdf"):
        return document_url
    cached = _resolved_document_cache.get(document_url)
    if cached:
        return cached
    if "ifsahdocs.boursakuwait.com.kw" not in document_url.lower():
        return document_url
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            resp = client.get(document_url)
            resp.raise_for_status()
        match = _PDF_HREF_RE.search(resp.text)
        if match:
            resolved = _quote_url_path(urljoin(document_url, unescape(match.group(1))))
            _resolved_document_cache[document_url] = resolved
            return resolved
    except Exception as exc:
        logger.debug("Boursa PDF resolution failed for %s: %s", document_url, exc)
    _resolved_document_cache[document_url] = document_url
    return document_url


def _quote_url_path(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        quote(parts.path, safe="/%"),
        parts.query,
        parts.fragment,
    ))


def _map_rss_item(raw: dict[str, str], lang: str = "en") -> dict:
    title = raw.get("title", "").strip()
    news_type = raw.get("type", "").strip()
    document_url = raw.get("url", "").strip()
    source_link = raw.get("link", "").strip()
    published_at = _parse_rss_date(raw.get("pubDate", ""))
    news_id = _extract_news_id(source_link, document_url, title, published_at)
    direct_url = _resolve_boursa_document_url(document_url) if document_url else source_link
    ticker = _extract_symbol_from_title(title)
    category = _classify_category(news_type, title)
    attachment_type = "pdf" if direct_url.lower().split("?", 1)[0].endswith(".pdf") else "link"

    return {
        "id": news_id,
        "title": title,
        "summary": raw.get("description", "").strip() or title,
        "source": "boursa_kuwait",
        "category": category,
        "publishedAt": published_at,
        "url": direct_url or source_link,
        "relatedSymbols": [ticker] if ticker else [],
        "sentiment": "neutral",
        "impact": _assess_impact(news_type, category),
        "language": "ar" if lang in ("A", "ar") else "en",
        "isVerified": True,
        "attachments": [{"type": attachment_type, "url": direct_url}] if direct_url else None,
    }


def _parse_rss_xml(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        items.append({
            "__source": "rss",
            "title": _xml_text(item, "title"),
            "type": _xml_text(item, "type"),
            "description": _xml_text(item, "description"),
            "link": _xml_text(item, "link"),
            "url": _xml_text(item, "url"),
            "pubDate": _xml_text(item, "pubDate"),
            "security": _xml_text(item, "security"),
        })
    return items


def _classify_category(news_type: str, title: str) -> str:
    """Determine the best category from NewsType code + title keywords."""
    # If we have a NewsType code, prefer the official mapping
    if news_type and news_type in _NEWS_TYPE_MAP:
        return _NEWS_TYPE_MAP[news_type]

    title_lower = title.lower()

    # Keyword fallback: earnings (analysts conf) > dividend > financial > regulatory > market
    if any(kw in title_lower for kw in _EARNINGS_KEYWORDS):
        return "earnings"
    if any(kw in title_lower for kw in _DIVIDEND_KEYWORDS):
        return "dividend"
    if any(kw in title_lower for kw in _FINANCIAL_KEYWORDS):
        return "financial"
    if any(kw in title_lower for kw in _REGULATORY_KEYWORDS):
        return "regulatory"
    if any(kw in title_lower for kw in _MARKET_KEYWORDS):
        return "market_news"

    return "company_announcement"


def _classify_from_title(title: str) -> str:
    """Determine category by checking if a known TitleTypeDesc appears in the title,
    then falling back to keyword-based classification."""
    if not title:
        return "company_announcement"

    # Check if any TitleTypeDesc key is contained within the title
    # Sort by length descending so longer (more specific) matches win
    for key in sorted(_TITLE_TYPE_CATEGORY, key=len, reverse=True):
        if key in title:
            return _TITLE_TYPE_CATEGORY[key]

    # Fallback to keyword-based classification
    return _classify_category("", title)


def _reclassify_stored_articles(db: Session) -> int:
    """Re-classify all stored articles using TitleTypeDesc matching + keywords.

    Called after keyword updates to fix historically mis-categorised items.
    Returns number of articles whose category changed.
    """
    rows = db.query(NewsArticle).all()
    changed = 0
    for row in rows:
        new_cat = _classify_from_title(row.title or "")
        new_impact = _assess_impact("", new_cat)
        if row.category != new_cat:
            row.category = new_cat
            row.impact = new_impact
            changed += 1
    if changed:
        db.commit()
        logger.info("Reclassified %d/%d stored articles", changed, len(rows))
    return changed


def _assess_impact(news_type: str, category: str) -> str:
    """Determine impact level based on type and category."""
    if category in ("earnings", "dividend"):
        return "high"
    if category == "financial":
        return "medium"
    if category == "regulatory":
        return "medium"
    if news_type == "110":
        return "medium"
    return "informational"


def _boursa_news_view_url(news_id: str, lang: str) -> str | None:
    """Public Boursa Kuwait news viewer page — works for any NewsId, even
    historical items where the Url field was not provided by the upstream
    API. Used as a fallback so the "Open Source" action is never a dead end.
    """
    if not news_id:
        return None
    lang_seg = "ar" if lang in ("ar", "A") else "en"
    return f"https://www.boursakuwait.com.kw/{lang_seg}/news/view#{news_id}"


def _map_item(raw: dict, lang: str = "en") -> dict:
    """Map a Boursa Kuwait announcement JSON to our NewsItem schema.

    Works for both RT=3507/3508 (has NewsType, Url) and RT=3516 (has
    TitleTypeDesc, NewsTypeDesc).
    """
    if raw.get("__source") == "rss":
        return _map_rss_item(raw, lang)

    news_type = str(raw.get("NewsType", "")).strip()
    title = (raw.get("Title") or "").strip()

    # RT=3516 items don't have NewsType but have TitleTypeDesc
    title_type = (raw.get("TitleTypeDesc") or "").strip()
    if title_type and title_type in _TITLE_TYPE_CATEGORY:
        category = _TITLE_TYPE_CATEGORY[title_type]
    else:
        category = _classify_category(news_type, title)

    ticker = (raw.get("DisplayTicker") or "").strip()

    # Resolve Arabic display names to English ticker codes so that
    # portfolio symbol matching works regardless of article language.
    if ticker and lang in ("A", "ar"):
        ticker = _AR_DISPLAY_TO_EN_TICKER.get(ticker, ticker)

    pdf_url = (raw.get("Url") or "").strip()
    direct_url = _resolve_boursa_document_url(pdf_url) if pdf_url else ""
    attachments = []
    if direct_url:
        attachment_type = "pdf" if direct_url.lower().split("?", 1)[0].endswith(".pdf") else "link"
        attachments.append({"type": attachment_type, "url": direct_url})

    news_id = str(raw.get("NewsId", ""))
    # Always provide an openable source URL: prefer the direct PDF, fall back
    # to Boursa Kuwait's public news-viewer page (works for every NewsId).
    source_url = direct_url or pdf_url or _boursa_news_view_url(news_id, lang)

    return {
        "id": news_id,
        "title": title,
        "summary": title,
        "source": "boursa_kuwait",
        "category": category,
        "publishedAt": _parse_date(str(raw.get("PostedDate", ""))),
        "url": source_url,
        "relatedSymbols": [ticker] if ticker else [],
        "sentiment": "neutral",
        "impact": _assess_impact(news_type, category),
        "language": "ar" if lang in ("A", "ar") else "en",
        "isVerified": True,
        "attachments": attachments if attachments else None,
    }


def _db_row_to_item(row: NewsArticle) -> dict:
    """Convert a DB row to the NewsItem response dict."""
    attachments = None
    if row.attachments_json:
        try:
            attachments = json.loads(row.attachments_json)
        except (json.JSONDecodeError, TypeError):
            attachments = None

    # Fall back to the Boursa Kuwait news-viewer page for historical items
    # that were imported without a direct PDF URL (RT=3516 doesn't return one).
    source_url = row.url or (
        _boursa_news_view_url(row.news_id, row.language)
        if row.source == "boursa_kuwait"
        else None
    )

    return {
        "id": row.news_id,
        "title": row.title,
        "summary": row.summary or row.title,
        "source": row.source,
        "category": row.category,
        "publishedAt": row.published_at.isoformat() if row.published_at else None,
        "url": source_url,
        "relatedSymbols": row.related_symbols.split(",") if row.related_symbols else [],
        "sentiment": row.sentiment,
        "impact": row.impact,
        "language": row.language,
        "isVerified": bool(row.is_verified),
        "attachments": attachments,
    }


def _content_hash(item: dict) -> str:
    """
    Compute a stable fingerprint for an article. Priority:
      1. guid (if present)
      2. url/link (if present)
      3. md5(title + publishedAt + url)

    Used as a fallback dedupe key in case the upstream NewsId is missing,
    reused, or changes for the same article.
    """
    guid = item.get("guid") or item.get("Guid")
    if guid:
        return md5(str(guid).encode("utf-8")).hexdigest()
    link = item.get("url") or item.get("link")
    if link:
        return md5(str(link).encode("utf-8")).hexdigest()
    payload = "|".join([
        str(item.get("title", "")),
        str(item.get("publishedAt", "")),
        str(link or ""),
    ])
    return md5(payload.encode("utf-8")).hexdigest()


def _update_existing_article(row: NewsArticle, item: dict) -> bool:
    """Refresh stored article metadata when an upstream RSS item improves it."""
    symbols_str = ",".join(item.get("relatedSymbols", [])) or None
    attachments_str = json.dumps(item["attachments"]) if item.get("attachments") else None
    changed = False
    for attr, value in (
        ("title", item.get("title", "")),
        ("summary", item.get("summary")),
        ("category", item.get("category", "company_announcement")),
        ("url", item.get("url")),
        ("related_symbols", symbols_str),
        ("impact", item.get("impact", "informational")),
        ("language", item.get("language", "en")),
        ("attachments_json", attachments_str),
    ):
        if value is not None and getattr(row, attr) != value:
            setattr(row, attr, value)
            changed = True
    if changed:
        row.fetched_at = datetime.now(timezone.utc)
    return changed


def _persist_articles(db: Session, items: list[dict]) -> int:
    """Upsert news items into the DB. Returns count of new articles inserted.

    Handles large batches (16k+) by chunking the existence check.
    """
    if not items:
        return 0

    # Batch the existence check in chunks of 500 to avoid SQL limits
    news_ids = [it["id"] for it in items if it.get("id")]
    existing_by_id: dict[str, NewsArticle] = {}
    chunk_size = 500
    for i in range(0, len(news_ids), chunk_size):
        chunk = news_ids[i: i + chunk_size]
        rows = db.query(NewsArticle).filter(
            NewsArticle.news_id.in_(chunk)
        ).all()
        existing_by_id.update({row.news_id: row for row in rows})

    # Secondary dedupe: collect existing content hashes to catch the same
    # article re-published with a different NewsId.
    item_hashes = {it.get("id", ""): _content_hash(it) for it in items if it.get("id")}
    hash_values = [h for h in item_hashes.values() if h]
    existing_hashes: set[str] = set()
    for i in range(0, len(hash_values), chunk_size):
        chunk = hash_values[i: i + chunk_size]
        rows = db.query(NewsArticle.content_hash).filter(
            NewsArticle.content_hash.in_(chunk)
        ).all()
        existing_hashes.update(r[0] for r in rows if r[0])

    inserted = 0
    updated = 0
    for it in items:
        nid = it.get("id", "")
        if not nid:
            continue
        existing_row = existing_by_id.get(nid)
        if existing_row:
            if _update_existing_article(existing_row, it):
                updated += 1
            continue
        chash = item_hashes.get(nid) or _content_hash(it)
        if chash in existing_hashes:
            continue

        symbols_str = ",".join(it.get("relatedSymbols", []))
        attachments_str = json.dumps(it["attachments"]) if it.get("attachments") else None

        try:
            pub_dt = datetime.fromisoformat(it["publishedAt"])
        except (ValueError, TypeError):
            pub_dt = datetime.now(timezone.utc)

        article = NewsArticle(
            news_id=nid,
            title=it.get("title", ""),
            summary=it.get("summary"),
            source=it.get("source", "boursa_kuwait"),
            category=it.get("category", "company_announcement"),
            published_at=pub_dt,
            url=it.get("url"),
            related_symbols=symbols_str or None,
            sentiment=it.get("sentiment", "neutral"),
            impact=it.get("impact", "informational"),
            language=it.get("language", "en"),
            is_verified=1 if it.get("isVerified", True) else 0,
            attachments_json=attachments_str,
            fetched_at=datetime.now(timezone.utc),
            content_hash=chash,
        )
        db.add(article)
        existing_hashes.add(chash)
        inserted += 1

        # Commit in batches of 500 for large imports
        if inserted % 500 == 0:
            db.commit()

    if inserted or updated:
        db.commit()

    return inserted


async def _fetch_boursa(params: dict) -> list[dict]:
    """
    Fetch data from Boursa Kuwait data-api via the shared retry-backed client.

    Uses fetch_with_retry from app.core.http_client (3 attempts, exponential
    back-off 1 s → 4 s) and caches results in news_cache (15-min TTL).
    """
    ck = (params.get("RT", ""), params.get("L", ""))
    cached = get_cached(_boursa_cache, ck)
    if cached is not None:
        return cached

    resp = await fetch_with_retry(BOURSA_API, params=params)
    data = resp.json()
    result = data if isinstance(data, list) else []
    set_cached(_boursa_cache, ck, result)
    return result


async def _fetch_live_boursa(lang_code: str) -> list[dict]:
    """Fetch current-day items from official Boursa RSS (fast, small payload)."""
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for url in _rss_urls(lang_code):
        try:
            resp = await fetch_with_retry(url)
            items = _parse_rss_xml(resp.text)
            for item in items:
                nid = _extract_news_id(item.get("link", ""), item.get("url", ""), item.get("title", ""), item.get("pubDate", ""))
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    merged.append(item)
        except Exception as e:
            logger.warning("Boursa RSS fetch failed url=%s: %s", url, e)
    return merged


def _bg_fetch_live(lang: str) -> None:
    """Background task: fetch current-day Boursa items and persist."""
    boursa_lang = "A" if lang == "ar" else "E"
    try:
        raw: list[dict] = []
        # Boursa's RSS host 403s requests without a browser-like User-Agent.
        with httpx.Client(timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioTracker/1.0)"}) as client:
            for url in _rss_urls(boursa_lang):
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    raw.extend(_parse_rss_xml(resp.text))
                except Exception as e:
                    logger.warning("BG RSS fetch failed url=%s: %s", url, e)
        if not raw:
            return
        # Deduplicate
        seen: set[str] = set()
        unique = []
        for item in raw:
            nid = _extract_news_id(item.get("link", ""), item.get("url", ""), item.get("title", ""), item.get("pubDate", ""))
            if nid and nid not in seen:
                seen.add(nid)
                unique.append(item)
        mapped = [_map_item(r, boursa_lang) for r in unique]
        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            count = _persist_articles(db, mapped)
            if count:
                logger.info("BG persisted %d live articles", count)
        except Exception as e:
            logger.warning("BG persist failed: %s", e)
        finally:
            db.close()
    except Exception as e:
        logger.warning("BG fetch failed: %s", e)


async def _fetch_all_boursa_sources(lang_code: str) -> list[dict]:
    """Fetch from ALL known Boursa RT codes including full history (RT=3516).

    Used by /fetch-all for bulk import. Returns ~16k+ items.
    """
    seen_ids: set[str] = set()
    merged: list[dict] = []

    # Full-history endpoint (RT=3516) — ~16k+ items since 2016
    try:
        items = await _fetch_boursa({"RT": _BOURSA_HISTORY_RT, "L": lang_code})
        for item in items:
            nid = str(item.get("NewsId", ""))
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                merged.append(item)
        logger.info("RT=%s returned %d items", _BOURSA_HISTORY_RT, len(items))
    except Exception as e:
        logger.warning("Boursa RT=%s fetch failed: %s", _BOURSA_HISTORY_RT, e)

    # Also fetch current-day sources from the official RSS feed.
    for url in _rss_urls(lang_code):
        try:
            resp = await fetch_with_retry(url)
            items = _parse_rss_xml(resp.text)
            for item in items:
                nid = _extract_news_id(item.get("link", ""), item.get("url", ""), item.get("title", ""), item.get("pubDate", ""))
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    merged.append(item)
        except Exception as e:
            logger.warning("Boursa RSS fetch failed url=%s: %s", url, e)

    return merged


@router.get("/feed")
async def news_feed(
    request: Request,
    response: Response,
    symbols: Optional[str] = Query(None, description="Comma-separated tickers"),
    categories: Optional[str] = Query(None, description="Comma-separated category filters"),
    cursor: Optional[str] = Query(None, description="Pagination cursor (offset)"),
    limit: int = Query(15, ge=1, le=100),
    lang: str = Query("en", description="Language: 'en' or 'ar'"),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Fetch paginated Boursa Kuwait news feed.

    Returns paginated articles from the DB. Live items from Boursa
    are fetched in the background and persisted for future requests.
    Full history is loaded via the /fetch-all endpoint.

    Conditional GET:
      Sets `Last-Modified` and `ETag` based on the most recent article matching
      the filters. Honours `If-None-Match` and `If-Modified-Since` and replies
      with `304 Not Modified` when nothing has changed, saving bandwidth.
    """
    feed_cache_key = cache_key(
        "news_feed",
        symbols or "",
        categories or "",
        cursor or "",
        limit,
        lang or "",
    )

    cached_payload = get_cached(news_cache, feed_cache_key)
    if cached_payload is not None:
        response.headers["Cache-Control"] = "private, max-age=15"
        response.headers["X-Cache-Status"] = "HIT"
        return cached_payload

    # Trigger background live fetch to keep DB fresh
    # Live data is fetched via /fetch-all or the cron scheduler;
    # the /feed endpoint just serves from DB for speed.

    # ── Build DB query with filters and pagination ──
    query = db.query(NewsArticle).order_by(NewsArticle.published_at.desc())

    if symbols:
        from sqlalchemy import or_
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        symbol_filters = [
            NewsArticle.related_symbols.ilike(f"%{sym}%") for sym in sym_list
        ]
        query = query.filter(or_(*symbol_filters))

    if categories:
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        query = query.filter(NewsArticle.category.in_(cat_list))

    # Filter by language so EN users see EN articles and AR users see AR
    if lang:
        query = query.filter(NewsArticle.language == lang)

    total = query.count()

    # ── Compute Last-Modified / ETag from latest matching article ──
    # Cheap query: just the newest published_at + news_id for this filter set.
    latest = (
        query.with_entities(NewsArticle.published_at, NewsArticle.news_id)
        .first()
    )
    last_modified_dt: Optional[datetime] = latest[0] if latest else None
    # format_datetime(..., usegmt=True) requires a tz-aware UTC datetime.
    if last_modified_dt is not None and last_modified_dt.tzinfo is None:
        last_modified_dt = last_modified_dt.replace(tzinfo=timezone.utc)
    latest_id: Optional[str] = latest[1] if latest else None
    # ETag combines newest id + total + paging coords so any change invalidates.
    etag_seed = f"{latest_id or ''}|{total}|{cursor or ''}|{limit}|{lang or ''}"
    etag = 'W/"' + md5(etag_seed.encode("utf-8")).hexdigest() + '"'

    inm = request.headers.get("if-none-match")
    if inm and inm == etag:
        response.status_code = 304
        response.headers["ETag"] = etag
        if last_modified_dt:
            response.headers["Last-Modified"] = format_datetime(last_modified_dt, usegmt=True)
        response.headers["Cache-Control"] = "private, max-age=15"
        return Response(status_code=304, headers=dict(response.headers))

    if last_modified_dt:
        ims = request.headers.get("if-modified-since")
        if ims:
            try:
                ims_dt = parsedate_to_datetime(ims)
                # parsedate_to_datetime returns tz-aware; normalise both sides.
                lm_naive = last_modified_dt.replace(tzinfo=None)
                ims_naive = ims_dt.replace(tzinfo=None) if ims_dt.tzinfo else ims_dt
                if lm_naive <= ims_naive:
                    response.status_code = 304
                    response.headers["ETag"] = etag
                    response.headers["Last-Modified"] = format_datetime(last_modified_dt, usegmt=True)
                    response.headers["Cache-Control"] = "private, max-age=15"
                    return Response(status_code=304, headers=dict(response.headers))
            except (TypeError, ValueError):
                pass  # malformed header — just serve normally

    # ── Cursor-based pagination (offset) ──
    offset = int(cursor) if cursor else 0
    rows = query.offset(offset).limit(limit).all()
    items = [_db_row_to_item(row) for row in rows]
    next_cursor = str(offset + limit) if offset + limit < total else None

    response.headers["ETag"] = etag
    if last_modified_dt:
        response.headers["Last-Modified"] = format_datetime(last_modified_dt, usegmt=True)
    response.headers["Cache-Control"] = "private, max-age=15"
    response.headers["X-Cache-Status"] = "MISS"

    payload = {
        "items": items,
        "nextPageCursor": next_cursor,
        "totalAvailable": total,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    set_cached(news_cache, feed_cache_key, payload)
    return payload


@router.get("/history")
async def news_history(
    symbols: Optional[str] = Query(None, description="Comma-separated tickers"),
    categories: Optional[str] = Query(None, description="Comma-separated category filters"),
    date_from: Optional[str] = Query(None, description="Start date (ISO 8601, e.g. 2025-01-01)"),
    date_to: Optional[str] = Query(None, description="End date (ISO 8601, e.g. 2025-06-30)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100),
    lang: Optional[str] = Query(None, description="Filter by language: 'en' or 'ar'"),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Browse stored news history with date-range and filter support.
    Returns articles previously fetched and saved to the database.
    """
    query = db.query(NewsArticle).order_by(NewsArticle.published_at.desc())

    # Date range filters
    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from)
            query = query.filter(NewsArticle.published_at >= dt_from)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format")

    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to)
            # Include the entire end date
            dt_to = dt_to.replace(hour=23, minute=59, second=59)
            query = query.filter(NewsArticle.published_at <= dt_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format")

    # Symbol filter
    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        from sqlalchemy import or_
        symbol_filters = [
            NewsArticle.related_symbols.ilike(f"%{sym}%") for sym in sym_list
        ]
        query = query.filter(or_(*symbol_filters))

    # Category filter
    if categories:
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        query = query.filter(NewsArticle.category.in_(cat_list))

    # Language filter
    if lang:
        query = query.filter(NewsArticle.language == lang)

    total = query.count()
    offset = (page - 1) * limit
    rows = query.offset(offset).limit(limit).all()

    items = [_db_row_to_item(row) for row in rows]
    total_pages = (total + limit - 1) // limit

    return {
        "items": items,
        "page": page,
        "totalPages": total_pages,
        "totalItems": total,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/item/{news_id}")
async def news_item(
    news_id: str,
    lang: str = Query("en", description="Language: 'en' or 'ar'"),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single news item by NewsId. Checks DB first, then live API."""
    # Try DB first
    row = db.query(NewsArticle).filter(NewsArticle.news_id == news_id).first()
    if row:
        return _db_row_to_item(row)

    # Fallback to live API
    boursa_lang = "A" if lang == "ar" else "E"
    try:
        raw_items = await _fetch_live_boursa(boursa_lang)
    except Exception as e:
        logger.warning("Boursa Kuwait API fetch failed: %s", e)
        raw_items = []

    for r in raw_items:
        if str(r.get("NewsId")) == news_id:
            item = _map_item(r, boursa_lang)
            # Persist for future lookups
            try:
                _persist_articles(db, [item])
            except Exception:
                pass
            return item

    raise HTTPException(status_code=404, detail="News item not found")


@router.get("/sources")
async def news_sources(
    current_user: TokenData = Depends(get_current_user),
):
    """List available news sources and categories."""
    return {
        "sources": ["boursa_kuwait"],
        "categories": [
            "company_announcement",
            "financial",
            "dividend",
            "earnings",
            "market_news",
            "regulatory",
        ],
    }


@router.post("/fetch-all")
async def fetch_all_history(
    lang: str = Query("en", description="Language: 'en' or 'ar'"),
    current_user: TokenData = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    [P2-4/B-6] Bulk-fetch all available announcements from every Boursa Kuwait
    data-api source and persist them to the database.

    Queues the job in Celery so ingestion runs off the FastAPI event loop.
    Returns immediately with a task id and status endpoint.
    """
    from app.tasks.news import fetch_and_ingest_news_task

    news_cache.clear()
    try:
        task = fetch_and_ingest_news_task.delay(lang=lang)
    except OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"News queue unavailable: {exc}")

    return {
        "task_id": task.id,
        "status": "queued",
        "check_url": f"/api/v1/news/status/{task.id}",
    }


@router.get("/status/{job_id}", include_in_schema=True)
async def fetch_all_status(
    job_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Return the current status of a Celery-backed /fetch-all task."""
    from app.core.celery_app import celery_app

    result = celery_app.AsyncResult(job_id)
    if result.state == "PENDING":
        return {"status": "pending", "task_id": job_id}
    if result.state == "SUCCESS":
        return {"status": "completed", "task_id": job_id, "result": result.result}
    if result.state == "FAILURE":
        return {"status": "failed", "task_id": job_id, "error": str(result.result)}
    return {"status": result.state, "task_id": job_id, "meta": result.info}


async def _async_fetch_all_history(lang: str = "en", job_id: str | None = None) -> None:
    """
    [P2-4/B-6] Async bulk-fetch coroutine launched by /fetch-all.

    Uses fetch_with_retry for all outbound calls so transient network failures
    are retried automatically. Updates _news_jobs[job_id] throughout.
    """
    import time as _time

    if job_id and job_id in _news_jobs:
        _news_jobs[job_id]["status"] = "running"
        _news_jobs[job_id]["started_at"] = int(_time.time())

    total_new = 0
    try:
        boursa_lang = "A" if lang == "ar" else "E"

        for lang_code in [boursa_lang]:
            seen_ids: set[str] = set()
            merged: list[dict] = []

            # Full-history endpoint (RT=3516)
            try:
                items = await _fetch_boursa({"RT": _BOURSA_HISTORY_RT, "L": lang_code})
                for item in items:
                    nid = str(item.get("NewsId", ""))
                    if nid and nid not in seen_ids:
                        seen_ids.add(nid)
                        merged.append(item)
                logger.info("Async fetch RT=%s lang=%s: %d items", _BOURSA_HISTORY_RT, lang_code, len(merged))
            except Exception as exc:
                logger.warning("Async fetch RT=%s lang=%s failed (all retries): %s", _BOURSA_HISTORY_RT, lang_code, exc)

            # Current-day sources from the official RSS feed.
            for url in _rss_urls(lang_code):
                try:
                    resp = await fetch_with_retry(url)
                    items = _parse_rss_xml(resp.text)
                    for item in items:
                        nid = _extract_news_id(item.get("link", ""), item.get("url", ""), item.get("title", ""), item.get("pubDate", ""))
                        if nid and nid not in seen_ids:
                            seen_ids.add(nid)
                            merged.append(item)
                except Exception as exc:
                    logger.warning("Async RSS fetch url=%s lang=%s failed: %s", url, lang_code, exc)

            if not merged:
                continue

            mapped = [_map_item(r, lang_code) for r in merged]

            from app.core.database import SessionLocal
            async_db = SessionLocal()
            try:
                count = _persist_articles(async_db, mapped)
                total_new += count
                if count:
                    logger.info("Async bulk-fetch persisted %d new articles (lang=%s)", count, lang_code)
            except Exception as exc:
                logger.warning("Async bulk-fetch persist failed: %s", exc)
            finally:
                async_db.close()

        logger.info("Async bulk-fetch complete. Total new articles: %d", total_new)
    except Exception as _job_exc:
        news_cache.clear()
        if job_id and job_id in _news_jobs:
            _news_jobs[job_id]["status"] = f"failed: {_job_exc}"
            _news_jobs[job_id]["completed_at"] = int(_time.time())
            _news_jobs[job_id]["total_new"] = total_new
        logger.error("Async bulk-fetch job %s failed: %s", job_id, _job_exc, exc_info=True)
    else:
        news_cache.clear()
        if job_id and job_id in _news_jobs:
            _news_jobs[job_id]["status"] = "completed"
            _news_jobs[job_id]["completed_at"] = int(_time.time())
            _news_jobs[job_id]["total_new"] = total_new


@router.post("/import")
async def import_articles(
    body: NewsImportRequest,
    current_user: TokenData = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Bulk-import articles from JSON. Expects {"articles": [...]}."""
    del current_user
    articles = body.articles
    if not articles:
        return {"inserted": 0, "total": db.query(NewsArticle).count()}
    count = _persist_articles(db, articles)
    return {"inserted": count, "total": db.query(NewsArticle).count()}


@router.post("/fetch-test")
async def fetch_test(
    current_user: TokenData = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Quick synchronous test: fetch RT=3507 (small) and report result."""
    del current_user
    if settings.ENVIRONMENT.lower() == "production":
        raise HTTPException(status_code=404, detail="Not found")

    results: dict = {"totalBefore": db.query(NewsArticle).count()}
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(
                BOURSA_API, params={"RT": "3507", "L": "E"}
            )
            results["status_code"] = resp.status_code
            data = resp.json()
            results["items_returned"] = len(data) if isinstance(data, list) else 0
            if isinstance(data, list) and data:
                mapped = [_map_item(r, "E") for r in data]
                count = _persist_articles(db, mapped)
                results["new_inserted"] = count
            results["totalAfter"] = db.query(NewsArticle).count()
    except Exception as e:
        logger.warning("News fetch-test failed: %s", e, exc_info=True)
        results["error"] = str(e)
    return results


@router.post("/reclassify")
async def reclassify_articles(
    current_user: TokenData = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Re-classify all stored articles using current keyword lists.
    Use after updating category keywords to fix historical data.
    """
    from collections import Counter
    del current_user

    # Get category distribution before
    before = Counter(
        r[0] for r in db.query(NewsArticle.category).all()
    )

    changed = _reclassify_stored_articles(db)

    # Get category distribution after
    after = Counter(
        r[0] for r in db.query(NewsArticle.category).all()
    )

    return {
        "totalArticles": db.query(NewsArticle).count(),
        "reclassified": changed,
        "before": dict(before),
        "after": dict(after),
    }
