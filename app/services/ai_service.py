"""
AI Service — Google Gemini integration for portfolio analysis.

Gathers portfolio data and sends structured prompts to Gemini
for AI-powered investment insights.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

AI_GENERATION_TIMEOUT_SECONDS = 25.0


async def _run_with_timeout(func, *args, timeout: float = AI_GENERATION_TIMEOUT_SECONDS, **kwargs):
    """Run a blocking SDK call in a thread with a hard timeout."""
    return await asyncio.wait_for(
        asyncio.to_thread(func, *args, **kwargs),
        timeout=timeout,
    )


def _extract_response_text(resp: object) -> str:
    """Best-effort text extraction across Gemini SDK response shapes."""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text
    return ""


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

    # Call Gemini API (prefer new google-genai SDK used in requirements)
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = await _run_with_timeout(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        analysis_text = _extract_response_text(response)
        if not analysis_text:
            raise ValueError("Empty response from Gemini model.")

        return {
            "analysis": analysis_text,
            "model": "gemini-2.5-flash",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cached": False,
        }

    except ImportError:
        # Backward compatibility for environments still using legacy SDK
        try:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel("gemini-2.5-flash")
            response = await _run_with_timeout(model.generate_content, full_prompt)
            analysis_text = _extract_response_text(response)
            if not analysis_text:
                raise ValueError("Empty response from Gemini model.")

            return {
                "analysis": analysis_text,
                "model": "gemini-2.5-flash",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cached": False,
            }
        except ImportError:
            logger.error("Neither google-genai nor google-generativeai is installed")
            raise ValueError(
                "AI analysis requires Gemini SDK. Install with: pip install google-genai"
            )
    except asyncio.TimeoutError:
        logger.error("Gemini API timeout after %.1fs", AI_GENERATION_TIMEOUT_SECONDS)
        raise ValueError("AI provider timed out. Please retry in a moment.")
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise ValueError(f"AI analysis failed: {exc}")


# ── Whale Radar deep-thinking chat ───────────────────────────────────

WHALE_CHAT_TIMEOUT_SECONDS = 90.0


async def whale_chat(user_id: int, prompt: str) -> dict:
    """
    Run a deep-thinking Gemini analysis turn for the Whale Radar chat.

    Uses ``gemini-2.5-pro`` with the thinking budget set to dynamic
    (``-1``) so the model spends as long as it needs reasoning before
    answering. Falls back to ``gemini-2.5-flash`` (also with thinking)
    if pro is not available for the API key.

    Returns the same shape as ``analyze_portfolio``.
    """
    settings = get_settings()

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
        pass

    if not api_key:
        raise ValueError(
            "AI chat requires a Gemini API key. "
            "Add it in Settings or set GEMINI_API_KEY in .env."
        )

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise ValueError(
            "AI chat requires the google-genai SDK. "
            "Install with: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)

    # Dynamic thinking budget: model decides how much to think.
    thinking_cfg = genai_types.ThinkingConfig(
        thinking_budget=-1,
        include_thoughts=False,
    )
    config = genai_types.GenerateContentConfig(
        thinking_config=thinking_cfg,
        temperature=0.7,
        max_output_tokens=4096,
    )

    last_exc: Exception | None = None
    for model_id in ("gemini-2.5-pro", "gemini-2.5-flash"):
        try:
            response = await _run_with_timeout(
                client.models.generate_content,
                model=model_id,
                contents=prompt,
                config=config,
                timeout=WHALE_CHAT_TIMEOUT_SECONDS,
            )
            text = _extract_response_text(response)
            if not text:
                raise ValueError(f"Empty response from {model_id}.")
            return {
                "analysis": text,
                "model": model_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cached": False,
            }
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning("Whale chat timeout on %s after %.1fs", model_id, WHALE_CHAT_TIMEOUT_SECONDS)
            continue
        except Exception as exc:
            last_exc = exc
            logger.warning("Whale chat error on %s: %s", model_id, exc)
            continue

    raise ValueError(f"AI chat failed: {last_exc}")


# ── Cache-first AI analysis (Phase 9) ────────────────────────────────


async def generate_cached_analysis(
    user_id: int,
    context: dict,
    language: str = "en",
    endpoint: str = "portfolio_analysis",
) -> dict:
    """
    Cache-first portfolio analysis using Redis (DB 1) as a 24-hour result store.

    Flow:
      1. Compress context → deterministic hash
      2. Check Redis → return HIT immediately
      3. On MISS: call Gemini, store result, return with MISS metadata

    Returns a dict with keys:
      - ``analysis``    — markdown/text from Gemini
      - ``metadata``    — tokens, latency, model
      - ``cache_status`` — "HIT" | "MISS" | "ERROR"

    Never raises; errors are returned as cache_status="ERROR".
    """
    import time

    from app.core.ai_cache import get_ai_cache, set_ai_cache
    from app.core.ai_metrics import (
        record_ai_cache_hit,
        record_ai_cache_miss,
        record_ai_latency,
        record_ai_tokens,
    )
    from app.services.context_optimizer import compress_portfolio_context

    ctx_hash, ctx_payload = compress_portfolio_context(context)
    cache_key = f"ai:{user_id}:{language}:{ctx_hash}"

    # ── 1. Cache hit ──────────────────────────────────────────────────
    cached = await get_ai_cache(cache_key)
    if cached:
        logger.info("AI cache HIT for user %d (key=%s…)", user_id, cache_key[-12:])
        record_ai_cache_hit(endpoint)
        cached["cache_status"] = "HIT"
        return cached

    # ── 2. Cache miss → call Gemini ───────────────────────────────────
    record_ai_cache_miss(endpoint)

    settings = get_settings()
    api_key = settings.GEMINI_API_KEY
    try:
        from app.core.database import query_one, add_column_if_missing
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one("SELECT gemini_api_key FROM users WHERE id = ?", (user_id,))
        if row and row[0]:
            api_key = row[0]
    except Exception:
        pass

    if not api_key:
        return {
            "analysis": "AI analysis requires a Gemini API key.",
            "metadata": {"error": "no_api_key"},
            "cache_status": "ERROR",
        }

    language_instruction = "Respond in Arabic." if language == "ar" else "Respond in English."
    prompt = (
        f"{language_instruction}\n\n"
        f"You are an expert investment analyst. Analyze the following compressed portfolio data "
        f"and provide a concise assessment covering health, diversification, risk, and top recommendations.\n\n"
        f"Portfolio Data (JSON): {ctx_payload}"
    )

    start = time.perf_counter()
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = await _run_with_timeout(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt,
        )

        latency = time.perf_counter() - start
        analysis_text = _extract_response_text(response)

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) or getattr(usage, "total_prompt_token_count", 0)
        tokens_out = getattr(usage, "candidates_token_count", 0) or getattr(usage, "total_candidate_token_count", 0)

        result: dict = {
            "analysis": analysis_text,
            "metadata": {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": round(latency * 1000, 1),
                "model": "gemini-2.0-flash",
                "language": language,
            },
            "cache_status": "MISS",
        }

        # Record metrics
        record_ai_latency(latency, endpoint=endpoint, model="gemini-2.0-flash")
        record_ai_tokens(tokens_in, tokens_out, model="gemini-2.0-flash")

        # Store in Redis (24-hour TTL)
        await set_ai_cache(cache_key, result, ttl=86400)
        logger.info(
            "AI analysis cached for user %d: %d→%d tokens, %.2fs",
            user_id, tokens_in, tokens_out, latency,
        )
        return result

    except Exception as exc:
        latency = time.perf_counter() - start
        logger.error("Gemini API failed in generate_cached_analysis: %s", exc)
        return {
            "analysis": "⚠️ Analysis temporarily unavailable. Please try again later.",
            "metadata": {
                "error": str(exc),
                "latency_ms": round(latency * 1000, 1),
            },
            "cache_status": "ERROR",
        }
