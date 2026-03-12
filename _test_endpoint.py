import asyncio
from app.api.v1.fundamental import list_stocks, _ensure_schema
from app.core.security import TokenData

async def test():
    _ensure_schema()
    # Simulate user_id=2
    user = TokenData(user_id=2, username="binsager")
    result = await list_stocks(search=None, current_user=user)
    print("Status:", result["status"])
    print("Count:", result["data"]["count"])
    for s in result["data"]["stocks"]:
        print(f"  id={s['id']} {s['symbol']} - {s['company_name']}")

asyncio.run(test())
