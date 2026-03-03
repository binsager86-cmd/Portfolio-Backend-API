"""
AI Service — Google Gemini integration for portfolio analysis.

Gathers portfolio data and sends structured prompts to Gemini
for AI-powered investment insights.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def analyze_portfolio(
    user_id: int,
    prompt: Optional[str] = None,
    include_holdings: bool = True,
    include_transactions: bool = False,
    include_performance: bool = True,
    language: str = "en",
) -> dict:
    """
    Run an AI analysis of the user's portfolio.

    Args:
        user_id: The authenticated user's ID.
        prompt: Optional custom prompt (overrides default template).
        include_holdings: Include current holdings data.
        include_transactions: Include recent transaction history.
        include_performance: Include performance metrics.
        language: Response language ('en' or 'ar').

    Returns:
        dict with 'analysis' (markdown), 'model', 'generated_at', 'cached'.

    Raises:
        ValueError if Gemini API key is not configured.
    """
    settings = get_settings()

    # Try per-user key first, then fall back to server-wide key
    api_key = settings.GEMINI_API_KEY
    try:
        from app.core.database import query_one, add_column_if_missing
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one(
            "SELECT gemini_api_key FROM users WHERE id = ?", (user_id,)
        )
        if row and row[0]:
            api_key = row[0]
    except Exception:
        pass  # Fall back to server-wide key

    if not api_key:
        raise ValueError(
            "AI analysis requires a Gemini API key. "
            "Add it in Settings or set GEMINI_API_KEY in .env."
        )

    # Gather portfolio context
    context_parts = []

    if include_holdings:
        from app.services.portfolio_service import build_portfolio_table
        from app.services.fx_service import PORTFOLIO_CCY

        for pname in PORTFOLIO_CCY:
            df = build_portfolio_table(pname, user_id)
            if not df.empty:
                summary = df[["symbol", "shares_qty", "avg_cost", "market_price",
                              "unrealized_pnl", "total_pnl", "currency"]].to_string(index=False)
                context_parts.append(f"## {pname} Portfolio Holdings\n{summary}")

    if include_performance:
        from app.services.portfolio_service import get_complete_overview
        overview = get_complete_overview(user_id)
        perf_summary = (
            f"## Portfolio Performance\n"
            f"Total Value: {overview.get('total_value', 0):,.2f} KWD\n"
            f"Total Gain: {overview.get('total_gain', 0):,.2f} KWD\n"
            f"ROI: {overview.get('roi_percent', 0):.2f}%\n"
            f"Net Deposits: {overview.get('net_deposits', 0):,.2f} KWD\n"
            f"Cash Balance: {overview.get('cash_balance', 0):,.2f} KWD"
        )
        context_parts.append(perf_summary)

    if include_transactions:
        from app.core.database import query_df
        tx_df = query_df(
            """SELECT stock_symbol, txn_type, shares, txn_date, purchase_cost, sell_value
               FROM transactions
               WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0
               ORDER BY txn_date DESC LIMIT 20""",
            (user_id,),
        )
        if not tx_df.empty:
            context_parts.append(f"## Recent Transactions\n{tx_df.to_string(index=False)}")

    context = "\n\n".join(context_parts)

    # Build prompt
    language_instruction = "Respond in Arabic." if language == "ar" else "Respond in English."

    if prompt:
        full_prompt = f"{language_instruction}\n\nPortfolio Data:\n{context}\n\nUser Question: {prompt}"
    else:
        full_prompt = (
            f"{language_instruction}\n\n"
            f"You are an expert investment analyst. Analyze the following portfolio data and provide:\n"
            f"1. Portfolio health assessment\n"
            f"2. Diversification analysis\n"
            f"3. Top performers and underperformers\n"
            f"4. Risk assessment\n"
            f"5. Actionable recommendations\n\n"
            f"Portfolio Data:\n{context}"
        )

    # Call Gemini API
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(full_prompt)

        return {
            "analysis": response.text,
            "model": "gemini-2.5-flash",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cached": False,
        }

    except ImportError:
        logger.error("google-generativeai package not installed")
        raise ValueError(
            "AI analysis requires the google-generativeai package. "
            "Install with: pip install google-generativeai"
        )
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise ValueError(f"AI analysis failed: {exc}")
