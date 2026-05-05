"""Test script for TickerChart Order Book integration.

Quick verification that order book fetching works with your existing
TickerChart credentials.
"""
import asyncio
import sys
from pathlib import Path

# Add backend-api to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.orderbook_client import OrderBookClient


async def test_order_book_fetch():
    """Test fetching order book for Kuwait stocks."""
    print("=" * 70)
    print("Testing TickerChart Order Book Integration")
    print("=" * 70)
    print()
    
    # Test symbols
    symbols = ["NBK", "KFH", "BOUBYAN"]
    
    # Initialize client
    print("Initializing OrderBookClient (market=KSE)...")
    client = OrderBookClient(market="KSE", timeout=10.0)
    await client.connect()
    print("✅ Client connected\n")
    
    for symbol in symbols:
        print(f"\n📊 Fetching order book for {symbol}...")
        print("-" * 70)
        
        try:
            snapshot = await client.get_order_book(symbol, depth=10)
            
            if snapshot is None:
                print(f"⚠️  Order book unavailable for {symbol}")
                print("    (Market depth may not be enabled for this symbol)")
                continue
            
            # Display results
            print(f"✅ Symbol: {snapshot['symbol']}")
            print(f"   Market: {snapshot['market']}")
            print(f"   Timestamp: {snapshot['timestamp']}")
            print(f"   Total Bid Volume: {snapshot['total_bid_volume']:,.0f}")
            print(f"   Total Ask Volume: {snapshot['total_ask_volume']:,.0f}")
            
            print("\n   Top 5 Bids:")
            for i, bid in enumerate(snapshot['bids'][:5], 1):
                print(f"      {i}. Price: {bid['price']:,.3f}  Volume: {bid['volume']:,.0f}")
            
            print("\n   Top 5 Asks:")
            for i, ask in enumerate(snapshot['asks'][:5], 1):
                print(f"      {i}. Price: {ask['price']:,.3f}  Volume: {ask['volume']:,.0f}")
            
            # Calculate imbalance
            total = snapshot['total_bid_volume'] + snapshot['total_ask_volume']
            if total > 0:
                imbalance = (snapshot['total_bid_volume'] - snapshot['total_ask_volume']) / total
                print(f"\n   📈 Imbalance Ratio: {imbalance:+.4f}")
                if imbalance > 0.30:
                    print("      → Strong BUYING pressure")
                elif imbalance < -0.30:
                    print("      → Strong SELLING pressure")
                else:
                    print("      → Balanced market")
            
            # Test historical spread
            print(f"\n   Fetching historical spread...")
            avg_spread = await client.get_historical_spread(symbol, lookback_days=20)
            print(f"   📊 Avg Spread (20d): {avg_spread:.4f} ({avg_spread*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    await client.close()
    print("\n" + "=" * 70)
    print("✅ Test Complete")
    print("=" * 70)


async def test_order_book_endpoints():
    """Test which TickerChart endpoint works for market depth."""
    print("\n" + "=" * 70)
    print("Testing TickerChart Market Depth Endpoints")
    print("=" * 70)
    
    from app.services import tickerchart_service
    
    test_symbol = "NBK"
    market = "KSE"
    
    print(f"\n🔍 Testing endpoints for {test_symbol}.{market}...")
    
    try:
        result = await tickerchart_service.fetch_order_book(
            base_symbol=test_symbol,
            market_abb=market,
            depth=10,
        )
        
        if result:
            print(f"\n✅ Successfully fetched order book!")
            print(f"   Bids: {len(result['bids'])} levels")
            print(f"   Asks: {len(result['asks'])} levels")
            print(f"   Total Bid Volume: {result['total_bid_volume']:,.0f}")
            print(f"   Total Ask Volume: {result['total_ask_volume']:,.0f}")
        else:
            print("\n⚠️  No order book data returned (may not be available)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting TickerChart Order Book Tests\n")
    
    # Test 1: Direct endpoint testing
    asyncio.run(test_order_book_endpoints())
    
    # Test 2: Client wrapper testing
    asyncio.run(test_order_book_fetch())
    
    print("\n✅ All tests complete!\n")
