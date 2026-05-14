# Kuwait Stock Analysis Engine — Phase 1

> Multi-year forensic behavioral learning across Boursa Kuwait stocks.
> Foundation for the full 5-phase analysis engine.

## What This Is

Phase 1 of the comprehensive Kuwait Stock Analysis Engine. This phase studies
every stock's last 3 years of price action and extracts its **behavioral DNA** —
a fingerprint of how that specific stock has historically behaved before
significant moves.

The behavioral DNA captures:
- The stock's personality (slow_builder / volatile_burst / high_amplitude_trender / range_grinder / balanced_mover)
- Average consolidation period before major moves
- Average move duration and magnitude
- Per-threshold profiles (10%, 15%, 25%, 50%, 100%+ moves studied separately)
- The most reliable early-warning signals (with average lead time, reliability %, and false-positive rate)
- Fakeout signatures (signals that fire but don't lead to real moves)

This is the foundation Phase 2 (ML), Phase 3 (live rating), Phase 4 (React Native UI), and Phase 5 (backtesting) build on.

## Architecture

```
core/
  config.py        — all tunable parameters
  pipeline.py      — Phase 1 orchestrator

data/
  adapters.py      — DataAdapter interface + SyntheticAdapter
                     (TickerChartAdapter stub for your API)

indicators/
  engine.py        — 60+ technical indicators in pure numpy/pandas
                     (trend, momentum, volatility, volume/flow,
                      structure, statistical, institutional, regime)

forensics/
  move_detector.py    — finds every significant move at every threshold
  recorder.py         — captures indicator state at t-90/60/30/14/7/3/1/0
                        around each move; computes signal sequence
  dna_extractor.py    — aggregates events into per-stock behavioral DNA

tests/
  test_indicators.py — sanity tests for indicator math

demo_run_phase1.py   — end-to-end runner using SyntheticAdapter
```

## Running

```bash
pip install numpy pandas
python tests/test_indicators.py     # validate indicator math
python demo_run_phase1.py            # run full Phase 1 pipeline
```

Outputs land in `./output/`:
- `dna/<TICKER>_dna.json`           — behavioral DNA per stock
- `events/<TICKER>_events.csv`      — all historical move events
- `indicators/<TICKER>.csv`         — full indicator history
- `phase1_summary.json`             — top-level summary

## Wiring In Your Ticker Chart API

`data/adapters.py` has a stub `TickerChartAdapter` class. Once we inspect your
existing React Native code under the technical analysis page, we fill in:

```python
class TickerChartAdapter(DataAdapter):
    def list_stocks(self) -> List[StockMeta]: ...
    def get_ohlcv_daily(self, ticker, start, end) -> pd.DataFrame: ...
    def get_ohlcv_weekly(self, ticker, start, end) -> pd.DataFrame: ...
    def get_corporate_actions(self, ticker) -> pd.DataFrame: ...
    def get_market_index(self, index_ticker, start, end) -> pd.DataFrame: ...
```

Once that's done, `run_phase1(TickerChartAdapter(...))` runs the entire pipeline
against real Boursa Kuwait data with no other changes to engine code.

## What's Next

- **Phase 2** — ML pipeline (LightGBM + LSTM sequence model) trained on the
  forensic events from Phase 1
- **Phase 3** — Live rating engine producing the 8 outputs (entry, S/R, TP1/2/3,
  stop, sizing, stage, rating)
- **Phase 4** — React Native UI (scanner, stock detail, behavioral DNA viewer,
  trade plan card)
- **Phase 5** — Backtesting + paper trading + continuous learning

## Configuration

All parameters live in `core/config.py`. Key ones:
- `HISTORY_YEARS = 3` — change to extend the learning window
- `MOVE_THRESHOLDS_PCT = (10, 15, 25, 50, 100)` — what counts as a "move"
- `PRE_MOVE_LOOKBACK_DAYS = (90, 60, 30, 14, 7, 3, 1, 0)` — when to snapshot
- `LIQUIDITY_CAP_PCT_OF_DAILY_TURNOVER = 10` — never own > 10% of avg daily turnover
- `CIRCUIT_BREAKER_DRAWDOWN_PCT = 25` — halt new entries on 25% portfolio drawdown
- `CONFIRMATION_MODAL_THRESHOLD_PCT = 30` — confirm before sizing > 30%
