"""Full pipeline test for NBK PE movement."""
import sys
sys.path.insert(0, ".")

from app.services.tickerchart_service import fetch_pe_from_flatfiles
from app.services.quarter_movement.pe_module import QuarterlyPERatioMovementModule
from datetime import date

market = "KSE"
pe_price_divisor = 1000.0
today = date.today()

flatfiles_pe = fetch_pe_from_flatfiles("NBK", "KSE")
print("FlatFiles PE loaded:", len(flatfiles_pe), "days")

result = QuarterlyPERatioMovementModule().compute_from_pe_series(flatfiles_pe, today)
print("eps_coverage:", result["eps_coverage"])
print("ttm_eps (before implied EPS):", result["ttm_eps"])

last_pe_date = max(flatfiles_pe.keys())
last_pe_val = flatfiles_pe[last_pe_date]
simulated_close_fils = 910.0
implied_eps = (simulated_close_fils / pe_price_divisor) / last_pe_val
result["ttm_eps"] = round(implied_eps, 6)
print("Implied EPS (KWD):", result["ttm_eps"])
print("Last PE date:", last_pe_date, "PE:", round(last_pe_val, 2))

print("\n== PE Movement Table (with data) ==")
for yr in sorted(result["pe_movement_table"].keys()):
    for q, cell in result["pe_movement_table"][yr].items():
        if cell and not cell.get("insufficient_data"):
            hi = cell["highest_pe"]
            lo = cell["lowest_pe"]
            ip = cell["in_progress"]
            print(f"  {yr} {q}: High={hi}x  Low={lo}x  in_progress={ip}")
        elif cell and cell.get("in_progress"):
            print(f"  {yr} {q}: in-progress (no TC PE data yet)")

print("\n== PE Means ==")
for q, m in result["pe_movement_means"].items():
    if m.get("highest_pe_mean"):
        hi = m["highest_pe_mean"]
        lo = m["lowest_pe_mean"]
        n = m["sample_count"]
        print(f"  {q}: High mean={hi}x  Low mean={lo}x  samples={n}")
