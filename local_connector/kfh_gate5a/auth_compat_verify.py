"""Owner verifier for Gate 5A authentication transparency; stops at READY."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import suppress

from local_connector.kfh_gate3a.connector import KfhGate3AConnector

from .browser import Gate5AAuthDiagnostics, Gate5APassiveBrowserRuntime
from .ui_debug import Gate5ATempUiDebugger

LOGIN_WAIT_SECONDS = 600
POST_AUTH_UI_WINDOW_SECONDS = 15
POST_AUTH_SCHEDULING_GRACE_SECONDS = 1


def _discard_statement_frame(_frame: str | bytes) -> None:
    return


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Owner-operated Gate 5A authentication compatibility verifier"
    )
    parser.add_argument(
        "--debug-temp",
        action="store_true",
        help="write sanitized temporary UI-signal diagnostics outside the repository",
    )
    return parser.parse_args()


def _print_temp_summary(
    debugger: Gate5ATempUiDebugger,
    final_state: str,
    diagnostics: Gate5AAuthDiagnostics,
) -> None:
    summary = debugger.summary(final_state, diagnostics)
    protocol = summary["authProtocol"]
    print("AUTH PROTOCOL:")
    print(f"5/101 = {'YES' if protocol['authResponseSeen'] else 'NO'}")
    print(f"authSts=1 = {'YES' if protocol['authStatusSuccess'] else 'NO'}")
    print("LOGIN UI:")
    print(f"inactive = {'YES' if summary['loginUiInactive'] else 'NO'}")
    print("CLOSED GATE3A UI:")
    print(f"count = {summary['closedGate3ASignalCount']}")
    print("CLOSED GATE3A LOGIN SIGNALS (FINAL SAMPLE):")
    for signal in summary["loginSignals"]:
        print(
            f"{signal['signal']} matched={signal['matched']} "
            f"visible={signal['visible']} matchCount={signal['matchCount']} "
            f"hasNonzeroBoundingBox={signal['hasNonzeroBoundingBox']} "
            f"ancestorVisible={signal['ancestorVisible']}"
        )
    print("TEMP CANDIDATES:")
    for name, visible in summary["tempCandidatesVisible"].items():
        print(f"{name} = {'YES' if visible else 'NO'}")
    page_state = summary["pageState"] or {}
    print("FRAME:")
    print(f"numberOfFrames = {page_state.get('numberOfFrames')}")
    print(
        "authenticatedMarkerFoundInChildFrame = "
        f"{'YES' if summary['authenticatedMarkerFoundInChildFrame'] else 'NO'}"
    )
    print("TIMING:")
    for sample in summary["timing"]:
        print(
            f"T+{sample['sampleOffsetSeconds']} = "
            f"authenticatedUiSignalCount={sample['authenticatedUiSignalCount']} "
            f"closedLoginUiActive={sample['closedLoginUiActive']} "
            f"wouldGate3AAuthenticate={sample['wouldGate3AAuthenticate']}"
        )
    print(f"FINAL STATE: {summary['finalState']}")
    print(f"ROOT CAUSE CATEGORY: {summary['rootCauseCategory']}")


async def _watch_owner_visual_marker(debugger: Gate5ATempUiDebugger) -> None:
    """Allow an optional Enter marker without affecting authentication state."""
    try:
        import msvcrt
    except ImportError:
        return
    print("OPTIONAL: press Enter after login is visually complete (marker only).")
    while not debugger.finalized:
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in {"\r", "\n"}:
                debugger.owner_visual_login_marker()
                return
        await asyncio.sleep(0.1)


async def main(*, debug_temp: bool = False) -> None:
    diagnostics = Gate5AAuthDiagnostics()
    debugger = Gate5ATempUiDebugger() if debug_temp else None
    if debugger:
        print(f"TEMP DEBUG FILE: {debugger.path}")
    runtime = Gate5APassiveBrowserRuntime(
        on_statement_request_frame=_discard_statement_frame,
        on_statement_response_frame=_discard_statement_frame,
        diagnostics=diagnostics,
        ui_debugger=debugger,
    )
    connector = KfhGate3AConnector(runtime)
    print("Opening the isolated, ephemeral KFH authentication verifier.")
    print("Enter credentials and OTP only in the visible KFH-controlled page.")
    wait_task: asyncio.Task | None = None
    debug_wait_task: asyncio.Task | None = None
    marker_task: asyncio.Task | None = None
    try:
        await connector.connect()
        # Owner credential/OTP entry is intentionally manual. Allow enough time
        # for that interaction and the authenticated UI to render without
        # changing any Gate 3A signal, threshold, or transition criterion.
        if debugger:
            wait_task = asyncio.create_task(
                connector.wait_for_ready(timeout_seconds=LOGIN_WAIT_SECONDS)
            )
            debug_wait_task = asyncio.create_task(
                debugger.wait(timeout_seconds=LOGIN_WAIT_SECONDS)
            )
            marker_task = asyncio.create_task(_watch_owner_visual_marker(debugger))
            done, _pending = await asyncio.wait(
                {wait_task, debug_wait_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if wait_task in done:
                snapshot = wait_task.result()
                if snapshot.state.value == "READY" and not debugger.completed.is_set():
                    await debugger.wait(
                        timeout_seconds=(
                            POST_AUTH_UI_WINDOW_SECONDS
                            + POST_AUTH_SCHEDULING_GRACE_SECONDS
                        )
                    )
            else:
                snapshot = connector.status()
        else:
            snapshot = await connector.wait_for_ready(
                timeout_seconds=LOGIN_WAIT_SECONDS
            )
        print(
            "GATE5A_AUTH_DIAGNOSTIC "
            + json.dumps(
                diagnostics.public_dict(snapshot.state.value), separators=(",", ":")
            )
        )
        if snapshot.state.value == "READY":
            print("GATE3A_AUTH_COMPATIBILITY = PASS")
            print("GATE3A_R1_LIVE = PASS")
            print("STATE = READY")
        else:
            print("GATE3A_AUTH_COMPATIBILITY = FAIL")
            print("GATE3A_R1_LIVE = FAIL")
            print(f"STATE = {snapshot.state.value}")
        if debugger:
            _print_temp_summary(debugger, snapshot.state.value, diagnostics)
    except Exception:
        if debugger:
            debugger.fail_safely()
        raise
    finally:
        for task in (wait_task, debug_wait_task, marker_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if debugger:
            await debugger.finalize(connector.status().state.value)
        await connector.close()
        if debugger:
            escaped_path = str(debugger.path).replace("'", "''")
            print(f"TEMP DEBUG FILE: {debugger.path}")
            print(f"DELETE WITH: Remove-Item -LiteralPath '{escaped_path}'")


if __name__ == "__main__":
    arguments = _arguments()
    asyncio.run(main(debug_temp=arguments.debug_temp))
