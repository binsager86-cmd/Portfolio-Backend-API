
import asyncio
from app.services import tickerchart_service as tc
from app.services.indicators_service import attach_indicators
from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal
from datetime import date, timedelta
from app.core.database import engine

async def main():
    print('Fetching data...')
    fetch_from = date.today() - timedelta(days=730)
    rows = await tc.fetch_ohlcv('NBK', 'KSE', from_d=fetch_from, to_d=None)
    print(f'Fetched {len(rows)} rows.')
    rows = attach_indicators(rows)
    print('Indicators attached. Generating signal...')
    signal = await generate_kuwait_signal(rows=rows, stock_code='NBK', segment='PREMIER')
    print('Signal generated!')
    print(signal['signal'])

asyncio.run(main())

