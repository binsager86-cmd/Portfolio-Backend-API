"""Quick smoke test — try importing each new router."""
import sys, traceback

modules = [
    "app.api.v1.dividends",
    "app.api.v1.securities",
    "app.api.v1.backup",
    "app.api.v1.stocks",
    "app.api.v1.tracker",
]

for mod_name in modules:
    try:
        mod = __import__(mod_name, fromlist=["router"])
        router = getattr(mod, "router")
        paths = [r.path for r in router.routes]
        print(f"  OK  {mod_name}  ({len(paths)} routes): {paths}")
    except Exception:
        print(f"FAIL  {mod_name}")
        traceback.print_exc()
        print()

# Also test full v1 router
print("\n--- Full v1 router ---")
try:
    from app.api.v1 import v1_router
    all_paths = [r.path for r in v1_router.routes]
    print(f"Total v1 routes: {len(all_paths)}")
    for p in sorted(all_paths):
        print(f"  {p}")
except Exception:
    print("FAIL to import v1_router")
    traceback.print_exc()
