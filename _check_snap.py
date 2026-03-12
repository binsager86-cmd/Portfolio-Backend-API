import requests, json
from app.core.security import create_access_token
TOKEN = create_access_token(1, "sager alsager")
r = requests.get("http://localhost:8002/api/v1/analytics/snapshots",
                  headers={"Authorization": f"Bearer {TOKEN}"})
data = r.json()
snapshots = data.get("data", {}).get("snapshots", [])
print(f"Total snapshots returned: {len(snapshots)}")
for s in snapshots[:5]:
    print(f"  {s['snapshot_date']}  val={s['portfolio_value']:>12.2f}  net_gain={s.get('net_gain')}  roi={s.get('roi_percent')}")
