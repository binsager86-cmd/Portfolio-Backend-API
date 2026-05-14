"""Step 3 debug: confirm TP1 ATR-floor is being applied."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.rating_engine import (
    classify_stage, compute_confidence, compute_support_resistance,
    compute_entry_stop_targets,
)
from app.services.eagle_eye.store import load_ohlcv

ohlcv = load_ohlcv("KFH")
EVAL_START = pd.Timestamp("2024-01-01")
eval_days = ohlcv.index[ohlcv.index >= EVAL_START]
sampled = eval_days[::5][:5]

print("=== KFH — 5-day TP1 debug ===")
for T in sampled:
    hist = ohlcv.loc[ohlcv.index <= T]
    if len(hist) < 200:
        continue
    ind_df = compute_all_indicators(hist)
    if ind_df.empty:
        continue
    indicators_row = ind_df.iloc[-1].to_dict()
    stage = classify_stage(indicators_row)
    confidence = compute_confidence(indicators_row, stage, dna=None, regime="NEUTRAL")
    sr = compute_support_resistance(hist, indicators_row)
    targets = compute_entry_stop_targets(hist, indicators_row, sr, stage=stage)
    close = float(hist["close"].iloc[-1])
    atr = float(indicators_row.get("atr") or close * 0.02)
    tp1_pct = (targets["tp1"] / close - 1) * 100
    print(f"  summary: date={T.date()} stage={stage} conf={confidence:.1f} "
          f"close={close:.4f} atr={atr:.4f} tp1={targets['tp1']:.4f} "
          f"tp1_dist%={tp1_pct:.2f}% atr_pct={atr/close*100:.2f}%")
    print()
