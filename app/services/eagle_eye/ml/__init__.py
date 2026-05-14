"""Eagle Eye Phase 2A ML training pipeline package."""

from .tier_resolver import resolve_model_for_ticker
from .trainer import EagleEyeMLTrainer

__all__ = ["EagleEyeMLTrainer", "resolve_model_for_ticker"]
