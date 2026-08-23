from __future__ import annotations

from pathlib import Path

ARCHIVE_ROOT = Path(r"F:\eagle_eye_archive")
SIMULATOR_ROOT = ARCHIVE_ROOT / "simulator"
LEDGER_PATH = SIMULATOR_ROOT / "ee_sim_ledger.db"
MANIFEST_PATH = ARCHIVE_ROOT / "MANIFEST.json"
RELEASE_ROOT = Path(__file__).resolve().parents[5] / "backend-api-main-release"
FORWARD_SURFACE_DB = RELEASE_ROOT / "artifacts" / "preview1a_prestart" / "review_final" / "forward_surface_gate_live_full.db"

INITIAL_CAPITAL_KWD = 100_000.0
POSITION_SIZE_FRACTION = 0.10
MAX_CONCURRENT_POSITIONS = 10
COMMISSION_RATE = 0.00325

BUY_PORTFOLIO = "BUY"
WATCHLIST_PORTFOLIO = "WATCHLIST"
PORTFOLIOS = (BUY_PORTFOLIO, WATCHLIST_PORTFOLIO)

ENTRY_REASONS = {
    "BASE_CONFIRMED_DIRECT",
    "MARKUP_PULLBACK_EMA_BAND",
    "MARKUP_FLAG_BREAKOUT",
    "MARKUP_CONFIRMED_DIRECT",
}
ENTRY_REASON_ALIASES = {
    "M1": "MARKUP_PULLBACK_EMA_BAND",
    "M2": "MARKUP_FLAG_BREAKOUT",
    "M3": "MARKUP_CONFIRMED_DIRECT",
}
EXIT_REASONS = {
    "EXIT_STRUCTURAL_EMA30_2C",
    "EXIT_AVOID_HARD",
    "EXITED_TIMESTOP_STAGNANT",
}

FROZEN_CODE = {
    "state_machine": {
        "path": ARCHIVE_ROOT / "freeze_v3" / "code" / "r16_3_candidate_state_machine.py",
        "sha256": "d16afb2ffa7faf80dfe2ad3d64034403589c7a21ed35b0fd09bd958954cf2eeb",
    },
    "harness": {
        "path": ARCHIVE_ROOT / "freeze_v3" / "code" / "r16_3_harness_v53.py",
        "sha256": "968625754efd1deb35259bc749ad583e2514e33efe46205186351a9692be1eee",
    },
}

FROZEN_VARIANT = "A"
RATIFIED_COMMISSION_AUTHORITY = "SIM-1 owner ratification: 0.325% per side"
