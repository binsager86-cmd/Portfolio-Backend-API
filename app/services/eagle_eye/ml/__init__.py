"""Eagle Eye ML pipeline package — Phase 1 (data layer) + Addendum A."""

from .db_tables import ensure_ml_tables, log_lifecycle, log_data_lineage
from .feature_store import FeatureStore
from .leakage_audit import LeakageAuditor, audit_feature_builder_module
from .data_pipeline import DataPipeline
from .corporate_events import CorporateEventFeatureBuilder, ingest_event
from .market_context import MarketContextBuilder
from .macro_features import MacroFeatureBuilder, write_data_gaps_report
from .eligibility_report import generate_eligibility_report, get_eligibility_summary_for_frontend
from .signal_logger import log_considered_signal, fill_realized_outcomes, SIGNAL_SKIP_REASONS
from .tier_resolver import resolve_model_for_ticker
from .trainer import EagleEyeMLTrainer

__all__ = [
    # Phase 1 — data layer
    "ensure_ml_tables",
    "log_lifecycle",
    "log_data_lineage",
    "FeatureStore",
    "LeakageAuditor",
    "audit_feature_builder_module",
    "DataPipeline",
    "CorporateEventFeatureBuilder",
    "ingest_event",
    "MarketContextBuilder",
    # Addendum A.3 — macro features
    "MacroFeatureBuilder",
    "write_data_gaps_report",
    # Addendum A.1 — eligibility coverage report
    "generate_eligibility_report",
    "get_eligibility_summary_for_frontend",
    # Addendum A.4 — considered-signal logger
    "log_considered_signal",
    "fill_realized_outcomes",
    "SIGNAL_SKIP_REASONS",
    # Phase 2 — training
    "EagleEyeMLTrainer",
    "resolve_model_for_ticker",
]

