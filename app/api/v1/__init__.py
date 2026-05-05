"""
API v1 — aggregates all versioned routers under /api/v1 prefix.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.cash import router as cash_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.ai import router as ai_router
from app.api.v1.pfm import router as pfm_router
from app.api.v1.cron import router as cron_router
from app.api.v1.integrity import router as integrity_router
from app.api.v1.dividends import router as dividends_router
from app.api.v1.securities import router as securities_router
from app.api.v1.backup import router as backup_router
from app.api.v1.stocks import router as stocks_router
from app.api.v1.tracker import router as tracker_router
from app.api.v1.fundamental import router as fundamental_router
from app.api.v1.trading import router as trading_router
from app.api.v1.admin import router as admin_router
from app.api.v1.news import router as news_router
from app.api.v1.market import router as market_router
from app.api.v1.notifications import router as notifications_router
from app.api.v1.trade_signals import router as trade_signals_router
from app.api.v1.users import router as users_router
from app.api.v1.compliance import router as compliance_router
from app.api.v1.system import router as system_router
from app.api.v1.orderbook import router as orderbook_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(auth_router)
v1_router.include_router(portfolio_router)
v1_router.include_router(cash_router)
v1_router.include_router(analytics_router)
v1_router.include_router(ai_router)
v1_router.include_router(pfm_router)
v1_router.include_router(cron_router)
v1_router.include_router(integrity_router)
v1_router.include_router(dividends_router)
v1_router.include_router(securities_router)
v1_router.include_router(backup_router)
v1_router.include_router(stocks_router)
v1_router.include_router(tracker_router)
v1_router.include_router(fundamental_router)
v1_router.include_router(trading_router)
v1_router.include_router(admin_router)
v1_router.include_router(news_router)
v1_router.include_router(market_router)
v1_router.include_router(notifications_router)
v1_router.include_router(trade_signals_router)
v1_router.include_router(users_router)
v1_router.include_router(compliance_router)
v1_router.include_router(system_router)
v1_router.include_router(orderbook_router)
