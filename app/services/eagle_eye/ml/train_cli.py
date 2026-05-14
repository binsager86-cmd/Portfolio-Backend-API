from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from .model_store import get_logs_root, get_reports_root
from .trainer import EagleEyeMLTrainer


def _configure_logging(models_root: Path | str | None) -> Path:
    logs_root = get_logs_root(models_root)
    log_path = logs_root / f"{date.today().isoformat()}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return log_path


def _print_summary(summary: dict) -> None:
    per_stock = summary.get("per_stock", {})
    per_sector = summary.get("per_sector", {})
    global_tier = summary.get("global", {})

    print("\n=== Eagle Eye ML Training Summary ===")
    print(
        "Per-stock tier: "
        f"{per_stock.get('trained', 0)} models trained, "
        f"{per_stock.get('rejected', 0)} rejected (AUC < 0.55)"
    )
    print(
        "Per-sector tier: "
        f"{per_sector.get('trained', 0)} models trained"
    )
    print(
        "Global tier: "
        f"{global_tier.get('trained', 0)} model trained"
    )
    print(f"Mean AUC per-stock (accepted): {per_stock.get('mean_auc')}")
    print(f"Mean AUC per-sector (accepted): {per_sector.get('mean_auc')}")
    print(f"Mean AUC global: {global_tier.get('mean_auc')}")
    print(f"Runtime (sec): {summary.get('runtime_sec')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eagle Eye Phase 2A ML trainer")
    parser.add_argument(
        "--tier",
        choices=["per_stock", "per_sector", "global", "all"],
        default="all",
        help="Training tier scope",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="Override output directory for ml_models",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of event feature cache",
    )
    args = parser.parse_args()

    log_file = _configure_logging(args.models_root)
    logging.getLogger(__name__).info("Starting Eagle Eye ML train_cli (tier=%s)", args.tier)

    trainer = EagleEyeMLTrainer(models_root=args.models_root)
    summary = trainer.run(tier=args.tier, force_rebuild=args.force_rebuild)

    reports_root = get_reports_root(args.models_root)
    report_path = reports_root / date.today().isoformat() / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _print_summary(summary)
    print(f"Log file: {log_file}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
