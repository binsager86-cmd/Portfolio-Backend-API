"""
DNA Debug Script — run from the backend-api directory.

    python debug_dna.py ALSAFAT

Checks every layer of the DNA pipeline and prints exactly where it fails.
"""
import sys
import json
import math
import traceback

TICKER = (sys.argv[1] if len(sys.argv) > 1 else "ALSAFAT").upper()

# ── Bootstrap Django-style env so imports work ──────────────────────────────
import os
os.environ.setdefault("PYTHONPATH", ".")

print(f"\n{'='*60}")
print(f"  DNA Diagnostics for: {TICKER}")
print(f"{'='*60}\n")

# ── 1. DB store — does saved DNA exist? ─────────────────────────────────────
print("STEP 1 ▸ Checking DB store (ee_dna_profiles) …")
try:
    from app.services.eagle_eye.store import load_dna, ensure_tables
    ensure_tables()
    stored = load_dna(TICKER)
    if stored is None:
        print("  ✗ No DNA in DB for this ticker — will need on-demand build\n")
    else:
        keys = list(stored.keys())
        print(f"  ✓ DNA found in DB. Top-level keys: {keys}")
        missing = [k for k in ("window_profiles", "setup_examples", "default_window_days",
                               "setup_signals", "signal_stats", "history_status") if k not in stored]
        if missing:
            print(f"  ⚠  Missing new-format keys (will trigger rebuild): {missing}")
        else:
            print("  ✓ All required keys present")
        print()
except Exception as e:
    print(f"  ✗ load_dna crashed: {e}")
    traceback.print_exc()
    stored = None
    print()

# ── 2. OHLCV — is there enough price history? ───────────────────────────────
print("STEP 2 ▸ Loading OHLCV from DB …")
try:
    from app.services.eagle_eye.store import load_ohlcv
    from app.services.eagle_eye.config import CONFIG
    df = load_ohlcv(TICKER)
    print(f"  rows in cache: {len(df)}  (need >= {CONFIG.MIN_HISTORY_DAYS_REQUIRED})")
    if len(df) == 0:
        print("  ✗ No OHLCV cached — DNA build will attempt live TickerChart fetch")
    elif len(df) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
        print(f"  ✗ Insufficient rows ({len(df)}) — DNA will return None → 'unavailable'")
    else:
        print(f"  ✓ Sufficient history  ({df.index[0]} → {df.index[-1]})")
    print()
except Exception as e:
    print(f"  ✗ load_ohlcv crashed: {e}")
    traceback.print_exc()
    df = None
    print()

# ── 3. Indicators ───────────────────────────────────────────────────────────
print("STEP 3 ▸ Computing indicators …")
try:
    from app.services.eagle_eye.indicators import compute_all_indicators
    if df is not None and len(df) > 0:
        ind_df = compute_all_indicators(df)
        print(f"  ✓ {len(ind_df.columns)} indicator columns computed on {len(ind_df)} rows")
        # Check for all-NaN columns that matter to DNA
        nan_cols = [c for c in ("rsi", "macd_histogram", "adx") if c in ind_df.columns and ind_df[c].isna().all()]
        if nan_cols:
            print(f"  ⚠  All-NaN columns (will appear as None in chart bars): {nan_cols}")
        print()
    else:
        print("  ⚠  Skipped (no OHLCV)\n")
        ind_df = None
except Exception as e:
    print(f"  ✗ compute_all_indicators crashed: {e}")
    traceback.print_exc()
    ind_df = None
    print()

# ── 4. Move detection ────────────────────────────────────────────────────────
print("STEP 4 ▸ Detecting moves + fakeouts …")
try:
    from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves
    if df is not None and len(df) > 0:
        moves = detect_moves(TICKER, df)
        fakeouts = detect_fakeouts(TICKER, df)
        print(f"  ✓ {len(moves)} moves, {len(fakeouts)} fakeouts detected")
        print()
    else:
        print("  ⚠  Skipped (no OHLCV)\n")
        moves, fakeouts = [], []
except Exception as e:
    print(f"  ✗ move detector crashed: {e}")
    traceback.print_exc()
    moves, fakeouts = [], []
    print()

# ── 5. DNA extraction ────────────────────────────────────────────────────────
print("STEP 5 ▸ Running extract_dna …")
dna = None
try:
    from app.services.eagle_eye.recorder import record_all_events
    from app.services.eagle_eye.dna_extractor import (
        DNA_CONFIDENCE_FLOOR, DNA_DEFAULT_WINDOW_DAYS, DNA_WINDOW_OPTIONS,
        dna_to_dict, extract_dna,
    )
    if df is not None and len(df) > 0 and ind_df is not None:
        all_events = moves + fakeouts
        snapshots = record_all_events(all_events, ind_df)
        print(f"  snapshots recorded: {len(snapshots)}")
        dna = extract_dna(
            TICKER, snapshots, [],
            indicators_df=ind_df,
            horizon_days=DNA_DEFAULT_WINDOW_DAYS,
            min_setup_occurrences=DNA_CONFIDENCE_FLOOR,
            window_days=DNA_WINDOW_OPTIONS,
        )
        if dna is None:
            print(f"  ✗ extract_dna returned None (< 3 real events) — endpoint will return 'unavailable'")
        else:
            print(f"  ✓ DNA extracted: personality={dna.personality_tag}, events={dna.total_events_studied}")
        print()
    else:
        print("  ⚠  Skipped\n")
except Exception as e:
    print(f"  ✗ extract_dna crashed: {e}")
    traceback.print_exc()
    print()

# ── 6. Serialization — does dna_to_dict produce JSON-safe output? ────────────
print("STEP 6 ▸ Serialising DNA dict + checking for NaN / non-serialisable values …")
dna_dict = None
try:
    if dna is not None:
        dna_dict = dna_to_dict(dna)
        raw_json = json.dumps(dna_dict)   # will raise if NaN / numpy types slip through
        print(f"  ✓ JSON serialisation OK  ({len(raw_json):,} bytes)")

        # Scan for NaN in bar data specifically
        nan_found = False
        for ex in dna_dict.get("setup_examples", []):
            for bar in ex.get("bars", []):
                for k, v in bar.items():
                    try:
                        if v is not None and math.isnan(float(v)):
                            print(f"  ⚠  NaN in bar[{k}] for example {ex.get('setup_date')}")
                            nan_found = True
                    except (TypeError, ValueError):
                        pass
        if not nan_found:
            print("  ✓ No NaN values found in bar data")
        print()
    else:
        print("  ⚠  Skipped (DNA is None)\n")
except (ValueError, TypeError) as e:
    print(f"  ✗ JSON serialisation FAILED: {e}")
    print("    ↳ This is the root cause of the 500 error on the DNA endpoint.")
    traceback.print_exc()
    print()

# ── 7. API response construction ─────────────────────────────────────────────
print("STEP 7 ▸ Building API response (Pydantic models) …")
try:
    from app.schemas.eagle_eye import (
        BehavioralDNAResponse, DNAWindowProfileResponse,
        DNASetupExampleResponse, DNASetupBarResponse,
    )
    if stored is not None or dna_dict is not None:
        src = dna_dict or stored
        wp = src.get("window_profiles", [])
        ex = src.get("setup_examples", [])
        print(f"  window_profiles: {len(wp)}, setup_examples: {len(ex)}")

        # Try building each setup example response
        for i, raw_ex in enumerate(ex):
            try:
                bars_ok = 0
                for bar in raw_ex.get("bars", []):
                    _b = {
                        "date": str(bar.get("date", "")),
                        **{k: (None if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else v)
                           for k in ("open","high","low","close","volume","rel_volume","rsi",
                                     "macd_line","macd_signal","macd_histogram","adx","plus_di","minus_di")
                           for v in [bar.get(k)]}
                    }
                    DNASetupBarResponse(**_b)
                    bars_ok += 1
                print(f"  ✓ Example {i}: {raw_ex.get('setup_date')} — {bars_ok} bars OK")
            except Exception as ex_err:
                print(f"  ✗ Example {i}: {raw_ex.get('setup_date')} FAILED — {ex_err}")
    else:
        print("  ⚠  Skipped (no DNA data)\n")
    print()
except Exception as e:
    print(f"  ✗ Pydantic model construction crashed: {e}")
    traceback.print_exc()
    print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("="*60)
if dna is None and stored is None:
    print("RESULT: No DNA data at all for this ticker.")
    print("  → The endpoint will return status='unavailable', not an error.")
    print("  → Frontend shows 'Insufficient Price History', not 'Failed to load'.")
    print("  → If you see 'Failed to load', the backend hasn't restarted yet.")
elif dna is not None or stored is not None:
    print("RESULT: DNA data exists — check steps above for any ✗ markers.")
    print("  → If all steps passed, restart the backend to apply code fixes.")
print("="*60, "\n")
