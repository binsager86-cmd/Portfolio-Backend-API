from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import random

SEED = 2107
random.seed(SEED)


@dataclass
class Bar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    value: float


def _next_weekday(dt: datetime) -> datetime:
    d = dt + timedelta(days=1)
    while d.weekday() in (4, 5):  # Fri/Sat off for Kuwait
        d += timedelta(days=1)
    return d


def _mk_bar(date: datetime, close: float, volume: float, jitter: float = 0.01) -> Bar:
    o = close * (1 + random.uniform(-jitter, jitter))
    h = max(o, close) * (1 + random.uniform(0.0, jitter * 1.8))
    l = min(o, close) * (1 - random.uniform(0.0, jitter * 1.8))
    value = close * volume
    return Bar(date=date, open=o, high=h, low=l, close=close, volume=volume, value=value)


def _write_csv(path: Path, rows: list[Bar]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume", "value"])
        for r in rows:
            w.writerow([
                r.date.strftime("%d/%m/%Y"),
                round(r.open, 3),
                round(r.high, 3),
                round(r.low, 3),
                round(r.close, 3),
                round(r.volume, 0),
                round(r.value, 2),
            ])


def synthetic_tijara(start: datetime) -> list[Bar]:
    rows: list[Bar] = []
    d = start
    close = 90.0
    # 180-session base 75-105 with late accumulation and volume rise.
    for i in range(180):
        bias = -0.07 if i < 60 else (0.05 if i > 140 else 0.0)
        close = min(105.0, max(75.0, close + random.uniform(-1.2, 1.2) + bias))
        vol = 120000 + (i * 1200)
        rows.append(_mk_bar(d, close, vol, 0.008))
        d = _next_weekday(d)
    # Pre-breakout pressure: near highs with elevated volume.
    for _ in range(2):
        close = min(104.5, max(100.5, close + random.uniform(0.2, 0.8)))
        rows.append(_mk_bar(d, close, 900000, 0.010))
        d = _next_weekday(d)
    # Breakout day 3x volume.
    close = 108.5
    rows.append(_mk_bar(d, close, 2400000, 0.015))
    d = _next_weekday(d)
    # Markup to ~180 with EMA30-holding pullbacks.
    for i in range(80):
        growth = 0.9 if i % 12 == 0 else 1.8
        close = min(185.0, close + growth + random.uniform(-0.8, 0.8))
        if i in (25, 52):
            close -= 7.5
        rows.append(_mk_bar(d, close, 240000 + i * 2500, 0.012))
        d = _next_weekday(d)
    return rows


def synthetic_bpcc(start: datetime) -> list[Bar]:
    rows: list[Bar] = []
    d = start
    close = 760.0
    # Long decline.
    for _ in range(90):
        close = max(560.0, close - random.uniform(1.0, 3.2))
        rows.append(_mk_bar(d, close, 130000, 0.007))
        d = _next_weekday(d)
    # 60-session base near lows.
    for i in range(60):
        close = min(615.0, max(555.0, 585.0 + random.uniform(-18, 18) + (i - 30) * 0.05))
        rows.append(_mk_bar(d, close, 140000 + i * 400, 0.006))
        d = _next_weekday(d)
    for _ in range(2):
        close = min(613.0, max(606.0, close + random.uniform(0.8, 2.0)))
        rows.append(_mk_bar(d, close, 780000, 0.010))
        d = _next_weekday(d)
    # MA200 reclaim style breakout.
    close = 622.0
    rows.append(_mk_bar(d, close, 1800000, 0.012))
    d = _next_weekday(d)
    for _ in range(50):
        close = min(700.0, close + random.uniform(0.2, 2.4))
        rows.append(_mk_bar(d, close, 210000, 0.01))
        d = _next_weekday(d)
    return rows


def synthetic_zain(start: datetime) -> list[Bar]:
    rows: list[Bar] = []
    d = start
    close = 515.0
    # 100-session 496-534 range.
    for i in range(100):
        close = min(534.0, max(496.0, 515.0 + random.uniform(-10, 10) + (0.06 * (i - 50))))
        rows.append(_mk_bar(d, close, 170000 + i * 350, 0.007))
        d = _next_weekday(d)
    for _ in range(2):
        close = min(533.5, max(529.0, close + random.uniform(0.4, 1.2)))
        rows.append(_mk_bar(d, close, 760000, 0.010))
        d = _next_weekday(d)
    # Breakout.
    close = 538.0
    rows.append(_mk_bar(d, close, 1900000, 0.013))
    d = _next_weekday(d)
    # Stair-step markup with pullbacks that should hold EMA30.
    for i in range(70):
        close += random.uniform(0.4, 1.6)
        if i in (18, 36, 57):
            close -= 5.0
        rows.append(_mk_bar(d, close, 230000, 0.01))
        d = _next_weekday(d)
    return rows


def synthetic_sanam(start: datetime) -> list[Bar]:
    rows: list[Bar] = []
    d = start
    close = 205.0
    # Tight base under 216.
    for i in range(120):
        close = min(216.0, max(195.0, 205 + random.uniform(-6, 6) + (i - 60) * 0.04))
        rows.append(_mk_bar(d, close, 110000 + i * 250, 0.007))
        d = _next_weekday(d)
    for _ in range(2):
        close = min(215.5, max(213.0, close + random.uniform(0.3, 1.0)))
        rows.append(_mk_bar(d, close, 620000, 0.010))
        d = _next_weekday(d)
    close = 219.0
    rows.append(_mk_bar(d, close, 1500000, 0.012))
    d = _next_weekday(d)
    # Strong markup; RSI should pin >70 for long stretch.
    for i in range(65):
        close = min(335.0, close + random.uniform(1.0, 2.7))
        if i in (24, 43):
            close -= 4.0
        rows.append(_mk_bar(d, close, 200000 + i * 1200, 0.011))
        d = _next_weekday(d)
    return rows


def synthetic_mabanee(start: datetime) -> list[Bar]:
    rows: list[Bar] = []
    d = start
    close = 980.0
    # Markup into top.
    for i in range(90):
        close = min(1185.0, close + random.uniform(0.6, 3.2))
        vol = 220000 - i * 1200  # weakening participation into highs
        rows.append(_mk_bar(d, close, max(70000, vol), 0.009))
        d = _next_weekday(d)
    # Climax bar near top then rollover.
    close = 1160.0
    rows.append(_mk_bar(d, close, 2600000, 0.02))
    d = _next_weekday(d)
    # Breakdown -20%+
    for _ in range(80):
        close = max(920.0, close - random.uniform(1.5, 4.8))
        rows.append(_mk_bar(d, close, 240000, 0.012))
        d = _next_weekday(d)
    return rows


def main() -> None:
    out_dir = Path(__file__).parent
    start = datetime(2025, 8, 3)
    series = {
        "synthetic_tijara.csv": synthetic_tijara(start),
        "synthetic_bpcc.csv": synthetic_bpcc(start),
        "synthetic_zain.csv": synthetic_zain(start),
        "synthetic_sanam.csv": synthetic_sanam(start),
        "synthetic_mabanee.csv": synthetic_mabanee(start),
    }
    for name, rows in series.items():
        _write_csv(out_dir / name, rows)


if __name__ == "__main__":
    main()
