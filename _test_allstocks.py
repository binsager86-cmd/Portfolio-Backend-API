import requests

# First get a token
t = requests.post(
    "http://127.0.0.1:8004/api/v1/auth/login",
    json={"username": "admin", "password": "admin123"},
    timeout=10,
)
print(f"Login: {t.status_code}")
token = t.json().get("access_token", "")

r = requests.get(
    "http://127.0.0.1:8004/api/v1/market/refresh",
    headers={"Authorization": f"Bearer {token}"},
    timeout=180,
)
d = r.json()
print(f"Status: {r.status_code}")
data = d.get("data", d)
print(f"\nTop Gainers ({len(data.get('top_gainers', []))}):")
for m in data.get("top_gainers", []):
    print(f"  {m['symbol']:12} {m.get('changePercent', '')}%")
print(f"\nTop Losers ({len(data.get('top_losers', []))}):")
for m in data.get("top_losers", []):
    print(f"  {m['symbol']:12} {m.get('changePercent', '')}%")
print(f"\nTop Value ({len(data.get('top_value', []))}):")
for m in data.get("top_value", []):
    print(f"  {m['symbol']:12} vol={m.get('volume', '')}")

b.close()
pw.stop()
