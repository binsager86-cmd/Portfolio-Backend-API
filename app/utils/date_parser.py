"""
Date parsing utilities.

Handles the flexible date formats found in the legacy data:
  - YYYY-MM-DD (standard)
  - DD/MM/YYYY (European)
  - MM/DD/YYYY (US)
  - DD-Mon-YYYY (e.g., 15-Jan-2024)
  - Excel serial date numbers
"""

import re
import logging
from datetime import datetime, date
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Pre-compiled patterns
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EU_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
_DMY_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")


def parse_date(value: Union[str, int, float, datetime, date, None]) -> Optional[str]:
    """
    Parse a date value into ISO format (YYYY-MM-DD).

    Accepts strings, datetime objects, and Excel serial date numbers.
    Returns None if the value cannot be parsed.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None

    # Already a date/datetime object
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.isoformat()

    # Excel serial date (numeric)
    if isinstance(value, (int, float)):
        try:
            # Excel epoch: 1899-12-30
            if 1 < value < 200000:
                from datetime import timedelta
                return (datetime(1899, 12, 30) + timedelta(days=int(value))).strftime("%Y-%m-%d")
        except Exception:
            pass
        return None

    # String parsing
    s = str(value).strip()

    # ISO format: YYYY-MM-DD
    if _ISO_RE.match(s):
        return s

    # European or US slash format: DD/MM/YYYY or MM/DD/YYYY
    if _EU_RE.match(s):
        parts = s.split("/")
        day_or_month, month_or_day, year = int(parts[0]), int(parts[1]), int(parts[2])
        # Heuristic: if first part > 12, it must be DD/MM/YYYY
        if day_or_month > 12:
            return f"{year:04d}-{month_or_day:02d}-{day_or_month:02d}"
        elif month_or_day > 12:
            return f"{year:04d}-{day_or_month:02d}-{month_or_day:02d}"
        else:
            # Ambiguous — assume DD/MM/YYYY (European, matches Kuwait conventions)
            return f"{year:04d}-{month_or_day:02d}-{day_or_month:02d}"

    # Day-Month-Year: 15-Jan-2024
    if _DMY_RE.match(s):
        try:
            dt = datetime.strptime(s, "%d-%b-%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Fallback: try pandas/dateutil
    try:
        from dateutil.parser import parse as dateutil_parse
        dt = dateutil_parse(s, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    logger.warning("Could not parse date: %r", value)
    return None


def today_iso() -> str:
    """Return today's date in ISO format."""
    return date.today().isoformat()
