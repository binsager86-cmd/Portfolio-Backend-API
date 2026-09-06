"""Owner-operated Gate 3A verification; opens KFH and reports states only."""

from __future__ import annotations

import asyncio

from .connector import KfhGate3AConnector


async def main() -> None:
    connector = KfhGate3AConnector()
    print("Opening an isolated, ephemeral KFH Chromium session.")
    print("Enter credentials and OTP only in the visible KFH-controlled page.")
    snapshot = await connector.connect()
    print(f"KFH state: {snapshot.state.value}")
    snapshot = await connector.wait_for_ready(timeout_seconds=300)
    print(f"KFH state: {snapshot.state.value}")
    if snapshot.state.value == "READY":
        await asyncio.to_thread(input, "Press Enter to perform allowlisted KFH logout and close: ")
        snapshot = await connector.logout()
        print(f"KFH state: {snapshot.state.value}")
    else:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(main())
