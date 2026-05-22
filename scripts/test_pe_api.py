"""Quick test: quarter-movement PE data via live API."""
import httpx

r = httpx.post(
    "http://127.0.0.1:8004/api/v1/auth/login",
    json={"username": "binsager.86@gmail.com", "password": "password123"},
)
print("Login:", r.status_code, r.text[:200])
token = r.json()["token"]
h = {"Authorization": f"Bearer {token}"}

stocks = httpx.get("http://127.0.0.1:8004/api/v1/analysis/stocks", headers=h).json()
nbk = next((s for s in stocks.get("stocks", []) if "NBK" in (s.get("symbol") or "")), None)
print("NBK:", nbk)

if nbk:
    qm = httpx.get(
        f"http://127.0.0.1:8004/api/v1/signals/quarter-movement/{nbk['id']}",
        headers=h,
        timeout=90,
    ).json()
    data = qm.get("data", {})
    print("eps_coverage:", data.get("eps_coverage"))
    print("ttm_eps:", data.get("ttm_eps"))
    print("ttm_eps_source:", data.get("ttm_eps_source"))
    pe_table = data.get("pe_movement_table", {})
    print("\n-- PE movement table --")
    for yr in sorted(pe_table.keys()):
        for q, cell in pe_table[yr].items():
            if cell and not cell.get("insufficient_data"):
                print(f"  {yr} {q}: hi={cell.get('highest_pe')} lo={cell.get('lowest_pe')} in_prog={cell.get('in_progress')}")
    print("\n-- PE means --")
    for q, m in (data.get("pe_movement_means") or {}).items():
        print(f"  {q}: hi_mean={m.get('highest_pe_mean')} lo_mean={m.get('lowest_pe_mean')}")
