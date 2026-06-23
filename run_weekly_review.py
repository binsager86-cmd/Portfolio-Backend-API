"""
Generate Eagle Eye weekly performance review.

Usage:
    python run_weekly_review.py                    # current week
    python run_weekly_review.py 2026-06-19         # specific week ending date
    python run_weekly_review.py 2026-06-19 --json  # JSON output only
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.services.eagle_eye.recommendation_tracker import (  # noqa: E402
    export_weekly_report,
    generate_weekly_review,
)


if __name__ == "__main__":
    week_end = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
        else None
    )
    json_only = "--json" in sys.argv

    if json_only:
        import json

        review = generate_weekly_review(week_end)
        print(json.dumps(review, indent=2, default=str))
    else:
        md = export_weekly_report(
            week_end,
            output_path="reports/weekly_review_latest.md",
        )
        print(md)
        print("\nSaved to reports/weekly_review_latest.md")
