"""KFH two-phase staging, matching, atomicity, and idempotency tests."""

from __future__ import annotations

import json
import time
from decimal import Decimal

import pytest

from app.core.config import get_settings
from app.core.database import exec_sql, query_all, query_one


@pytest.fixture(autouse=True)
def enable_kfh_for_controlled_test_user(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "KFH_AUTO_SYNC_ENABLED", True)
    monkeypatch.setattr(settings, "KFH_AUTO_SYNC_TEST_USER_IDS", "1")


def _record(
    reference: str,
    transaction_type: str = "DEPOSIT",
    *,
    date: str = "2026-08-01",
    symbol: str | None = None,
    quantity: str | None = None,
    price: str | None = None,
    amount: str = "10.000",
    status: str = "ready",
    settlement_date: str | None = None,
) -> dict:
    return {
        "broker_transaction_ref": reference,
        "transaction_date": date,
        "transaction_type": transaction_type,
        "settlement_date": settlement_date,
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "amount": amount,
        "interpretation_status": status,
        "description": f"test {reference}",
        "raw_payload": {"trnsRef": reference},
    }


def test_live_settlement_date_still_matches_legacy_xlsx_transaction(
    test_client, auth_headers
):
    suffix = str(time.time_ns())
    symbol = f"X{suffix[-7:]}"
    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-08-05', 'Buy', 10, 12.5, 0, 0, 0, 1.25,
                   NULL, 'portfolio', 'KFH_XLSX_IMPORT', 0, ?)""",
        (symbol, int(time.time())),
    )

    batch = _preview(
        test_client,
        auth_headers,
        f"cross-source-{suffix}",
        [
            _record(
                f"LIVE-{suffix}",
                "BUY",
                date="2026-08-05",
                settlement_date="2026-08-07",
                symbol=symbol,
                quantity="10",
                price="1.25",
                amount="12.5",
            )
        ],
    )

    assert batch["counts"]["NEW"] == 0
    assert batch["counts"]["MATCHED_EXISTING"] == 1
    assert batch["items"][0]["match_type"] == "FINGERPRINT_EXACT"


def test_committed_transaction_and_deposit_carry_kfh_reference_in_notes(
    test_client, auth_headers
):
    """The KFH broker reference must land in the Saham `notes` field - it's
    the only field the transaction/deposit edit forms actually surface,
    while `reference` is an internal linking column the user never sees."""
    suffix = str(time.time_ns())
    symbol = f"N{suffix[-7:]}"
    txn_ref = f"TXN-REF-{suffix}"
    deposit_ref = f"DEP-REF-{suffix}"
    batch = _preview(
        test_client,
        auth_headers,
        f"notes-{suffix}",
        [
            _record(
                txn_ref,
                "BUY",
                symbol=symbol,
                quantity="10",
                price="1",
                amount="10",
            ),
            _record(deposit_ref, "DEPOSIT", amount="100"),
        ],
    )
    response = _confirm(test_client, auth_headers, batch)
    assert response.status_code == 200, response.text

    txn_notes = query_one(
        "SELECT notes FROM transactions WHERE user_id = 1 AND reference = ?",
        (txn_ref,),
    )["notes"]
    assert txn_ref in txn_notes

    deposit_notes = query_one(
        "SELECT notes FROM cash_deposits WHERE user_id = 1 AND amount = 100.0 "
        "ORDER BY id DESC LIMIT 1"
    )["notes"]
    assert deposit_ref in deposit_notes


def _summary(**overrides) -> dict:
    summary = {
        "currency": "KWD",
        "open_balance": "100.000",
        "close_balance": "100.000",
        "total_deposit": "0.000",
        "total_withdrawal": "0.000",
        "total_buy": "0.000",
        "total_sell": "0.000",
        "total_other": "0.000",
        "vat_amount": "0.000",
    }
    summary.update(overrides)
    return summary


def _preview(
    test_client,
    auth_headers,
    account: str,
    records: list[dict],
    *,
    statement_summary: dict | None = None,
    unsettled_records: list[dict] | None = None,
    scope: dict | None = None,
) -> dict:
    payload = {
        "broker_account": account,
        "scope": scope or {
            "mode": "SMART_INCREMENTAL",
            "mergePolicy": "ADD_MISSING_ONLY",
        },
        "records": records,
        "unsettled_records": unsettled_records or [],
    }
    if statement_summary is not None:
        payload["statement_summary"] = statement_summary
    response = test_client.post(
        "/api/v1/kfh-sync/batches/preview",
        headers=auth_headers,
        json=payload,
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


def _confirm(
    test_client,
    auth_headers,
    batch: dict,
    selected: list[int] | None = None,
    *,
    update_cash_balance: bool = False,
):
    if selected is None:
        selected = [item["id"] for item in batch["items"] if item["classification"] == "NEW"]
    return test_client.post(
        f"/api/v1/kfh-sync/batches/{batch['batch_id']}/confirm",
        headers=auth_headers,
        json={
            "selected_item_ids": selected,
            "update_cash_balance": update_cash_balance,
        },
    )


def test_preview_has_five_states_and_does_not_change_financial_data(test_client, auth_headers):
    suffix = str(time.time_ns())
    account = f"classification-{suffix}"
    matched_ref = f"MATCH-{suffix}"
    conflict_ref = f"CONFLICT-{suffix}"
    duplicate_ref = f"DUPLICATE-{suffix}"
    new_ref = f"NEW-{suffix}"
    unsupported_ref = f"UNSUPPORTED-{suffix}"
    matched_symbol = f"M{suffix[-7:]}"
    conflict_symbol = f"C{suffix[-7:]}"

    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-08-02', 'Buy', 10, 10, 0, 0, 0, 1,
                   NULL, 'portfolio', 'MANUAL', 0, ?)""",
        (matched_symbol, int(time.time())),
    )
    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-08-03', 'Buy', 10, 10, 0, 0, 0, 1,
                   ?, 'portfolio', 'MANUAL', 0, ?)""",
        (conflict_symbol, conflict_ref, int(time.time())),
    )

    first = _preview(test_client, auth_headers, account, [_record(duplicate_ref, amount="25")])
    first_commit = _confirm(test_client, auth_headers, first)
    assert first_commit.status_code == 200

    before_transactions = query_one("SELECT COUNT(*) AS n FROM transactions")["n"]
    before_deposits = query_one("SELECT COUNT(*) AS n FROM cash_deposits")["n"]
    batch = _preview(
        test_client,
        auth_headers,
        account,
        [
            _record(new_ref, amount="50"),
            _record(duplicate_ref, amount="25"),
            _record(
                matched_ref,
                "BUY",
                date="2026-08-02",
                symbol=matched_symbol.lower(),
                quantity="10.0000000",
                price="1.0",
                amount="10.000",
            ),
            _record(
                conflict_ref,
                "BUY",
                date="2026-08-03",
                symbol=conflict_symbol,
                quantity="20",
                price="1",
                amount="20",
            ),
            _record(unsupported_ref, "STOCK_TRANSFER_FEE", amount="1", status="unsupported"),
        ],
    )

    assert batch["counts"] == {
        "NEW": 1,
        "EXACT_DUPLICATE": 1,
        "MATCHED_EXISTING": 1,
        "CONFLICT": 1,
        "UNSUPPORTED": 1,
        "ALREADY_IN_SAHAM": 2,
        "NEEDS_REVIEW": 2,
        "ERRORS": 0,
    }
    assert batch["financial_data_changed"] is False
    assert query_one("SELECT COUNT(*) AS n FROM transactions")["n"] == before_transactions
    assert query_one("SELECT COUNT(*) AS n FROM cash_deposits")["n"] == before_deposits

    selected_defaults = {
        item["classification"]: item["selected_default"] for item in batch["items"]
    }
    assert selected_defaults["NEW"] == 1
    assert all(
        item["selected_default"] == 0
        for item in batch["items"]
        if item["classification"] != "NEW"
    )
    matched_item = next(item for item in batch["items"] if item["classification"] == "MATCHED_EXISTING")
    assert matched_item["match_type"] == "FINGERPRINT_EXACT"
    assert matched_item["confidence"] == 1.0
    manual_after_preview = query_one(
        "SELECT shares, purchase_cost, price_override FROM transactions WHERE id = ?",
        (matched_item["matched_record_id"],),
    )
    assert tuple(manual_after_preview[field] for field in ("shares", "purchase_cost", "price_override")) == (
        10.0,
        10.0,
        1.0,
    )

    committed = _confirm(test_client, auth_headers, batch)
    assert committed.status_code == 200, committed.text
    result = committed.json()["data"]
    assert result["added"] == 1
    assert result["already_existed"] == 2
    assert result["requires_review"] == 2
    assert result["conflicts"] == 1
    assert result["unsupported"] == 1
    assert result["duplicates_added"] == 0
    assert result["existing_transactions_modified"] == 0


def test_non_new_items_cannot_be_confirmed(test_client, auth_headers):
    suffix = str(time.time_ns())
    batch = _preview(
        test_client,
        auth_headers,
        f"unsupported-{suffix}",
        [_record(f"UNSUPPORTED-{suffix}", "UNKNOWN", status="unsupported")],
    )
    response = _confirm(test_client, auth_headers, batch, [batch["items"][0]["id"]])
    assert response.status_code == 400
    assert "Only NEW" in response.text


def test_confirm_is_atomic_on_unexpected_failure(test_client, auth_headers, monkeypatch):
    from app.services import kfh_sync_service

    suffix = str(time.time_ns())
    references = [f"ATOMIC-A-{suffix}", f"ATOMIC-B-{suffix}"]
    batch = _preview(
        test_client,
        auth_headers,
        f"atomic-{suffix}",
        [
            _record(
                references[0],
                "BUY",
                symbol=f"A{suffix[-6:]}",
                quantity="10",
                price="1",
                amount="10",
            ),
            _record(
                references[1],
                "BUY",
                symbol=f"B{suffix[-6:]}",
                quantity="10",
                price="1",
                amount="10",
            ),
        ],
    )
    original = kfh_sync_service._insert_transaction
    calls = 0

    def fail_second(user_id, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated database failure")
        return original(user_id, payload)

    monkeypatch.setattr(kfh_sync_service, "_insert_transaction", fail_second)
    with pytest.raises(RuntimeError, match="simulated database failure"):
        _confirm(test_client, auth_headers, batch)

    count = query_one(
        "SELECT COUNT(*) AS n FROM transactions WHERE reference IN (?, ?)",
        tuple(references),
    )["n"]
    assert count == 0
    stored = query_one(
        "SELECT status FROM broker_import_batches WHERE id = ?",
        (batch["batch_id"],),
    )
    assert stored["status"] == "PREVIEW_READY"


def test_five_unchanged_syncs_equal_one_sync(test_client, auth_headers):
    suffix = str(time.time_ns())
    account = f"idempotency-{suffix}"
    reference = f"IDEMPOTENT-{suffix}"
    record = _record(
        reference,
        "BUY",
        date="2026-08-20",
        symbol=f"I{suffix[-7:]}",
        quantity="12",
        price="1.250000",
        amount="15.000000",
    )

    results = []
    for _ in range(5):
        batch = _preview(test_client, auth_headers, account, [record])
        response = _confirm(test_client, auth_headers, batch)
        assert response.status_code == 200, response.text
        results.append(response.json()["data"])

    assert results[0]["added"] == 1
    assert all(result["added"] == 0 for result in results[1:])
    transaction_count = query_one(
        "SELECT COUNT(*) AS n FROM transactions WHERE user_id = 1 AND reference = ?",
        (reference,),
    )["n"]
    raw_count = query_one(
        "SELECT COUNT(*) AS n FROM broker_raw_transactions WHERE user_id = 1 AND broker_transaction_ref = ?",
        (reference,),
    )["n"]
    link_count = query_one(
        """SELECT COUNT(*) AS n FROM broker_transaction_links l
           JOIN broker_raw_transactions r ON r.id = l.broker_raw_transaction_id
           WHERE r.user_id = 1 AND r.broker_transaction_ref = ?""",
        (reference,),
    )["n"]
    assert (transaction_count, raw_count, link_count) == (1, 1, 1)


def test_confirm_retry_returns_same_result_without_writes(test_client, auth_headers):
    suffix = str(time.time_ns())
    batch = _preview(
        test_client,
        auth_headers,
        f"retry-{suffix}",
        [_record(f"RETRY-{suffix}", amount="33")],
    )
    selected = [batch["items"][0]["id"]]
    first = _confirm(test_client, auth_headers, batch, selected)
    second = _confirm(test_client, auth_headers, batch, selected)
    assert first.status_code == second.status_code == 200
    assert first.json()["data"] == second.json()["data"]


def test_same_broker_reference_with_changed_material_fields_is_conflict(test_client, auth_headers):
    suffix = str(time.time_ns())
    account = f"raw-conflict-{suffix}"
    reference = f"RAW-CONFLICT-{suffix}"
    first = _preview(test_client, auth_headers, account, [_record(reference, amount="10")])
    assert first["counts"]["NEW"] == 1

    changed = _preview(test_client, auth_headers, account, [_record(reference, amount="11")])
    assert changed["counts"]["CONFLICT"] == 1
    assert changed["items"][0]["reason"] == (
        "Same KFH reference was retrieved with different material fields"
    )
    assert changed["items"][0]["saham_value"]["amount"] == "10.000000"
    assert changed["items"][0]["kfh_value"]["amount"] == "11.000000"
    stored = query_one(
        """SELECT COUNT(*) AS n, MIN(amount) AS amount
           FROM broker_raw_transactions WHERE user_id = 1 AND broker_transaction_ref = ?""",
        (reference,),
    )
    assert stored["n"] == 1
    assert Decimal(str(stored["amount"])) == Decimal("10.000000")


def test_transaction_reference_is_scoped_to_kfh_account(test_client, auth_headers):
    suffix = str(time.time_ns())
    reference = f"ACCOUNT-SCOPED-{suffix}"
    record = _record(reference, amount="17")

    first = _preview(test_client, auth_headers, f"account-a-{suffix}", [record])
    second = _preview(test_client, auth_headers, f"account-b-{suffix}", [record])
    assert first["counts"]["NEW"] == second["counts"]["NEW"] == 1
    assert _confirm(test_client, auth_headers, first).status_code == 200
    assert _confirm(test_client, auth_headers, second).status_code == 200

    raw_count = query_one(
        """SELECT COUNT(*) AS n FROM broker_raw_transactions
           WHERE user_id = 1 AND broker_transaction_ref = ?""",
        (reference,),
    )["n"]
    assert raw_count == 2


def test_statement_totals_use_decimal_tolerance(test_client, auth_headers):
    suffix = str(time.time_ns())
    batch = _preview(
        test_client,
        auth_headers,
        f"reconciled-{suffix}",
        [_record(f"DEPOSIT-{suffix}", amount="10.0004")],
        statement_summary=_summary(
            close_balance="110.000",
            total_deposit="10.000",
        ),
    )

    assert batch["status"] == "PREVIEW_READY"
    assert batch["reconciliation"]["status"] == "PASSED"
    assert batch["reconciliation"]["tolerance"] == "0.001"
    comparison = batch["reconciliation"]["comparisons"]["total_deposit"]
    assert comparison == {
        "reported": "10.000",
        "parsed": "10.000",
        "difference": "0.000",
        "matches": True,
    }


def test_cash_dividend_reconciles_with_kfh_total_deposit(test_client, auth_headers):
    """KFH reports positive corporate-action cash inside totDeposit."""
    suffix = str(time.time_ns())
    batch = _preview(
        test_client,
        auth_headers,
        f"dividend-total-{suffix}",
        [
            _record(f"DEPOSIT-{suffix}", amount="1000.000"),
            _record(
                f"DIVIDEND-{suffix}",
                "CASH_DIVIDEND",
                symbol="KFH",
                amount="418.620",
            ),
        ],
        statement_summary=_summary(
            total_deposit="1418.620",
            total_other="0.000",
        ),
    )

    assert batch["status"] == "PREVIEW_READY"
    assert batch["counts"]["ERRORS"] == 0
    assert batch["reconciliation"]["status"] == "PASSED"
    assert batch["reconciliation"]["comparisons"]["total_deposit"]["matches"] is True
    assert batch["reconciliation"]["comparisons"]["total_other"]["matches"] is True


def test_manual_buy_matches_despite_price_rounding_drift(test_client, auth_headers):
    """A manually entered BUY has no price_override; Saham derives price as
    purchase_cost / shares, which will not exactly equal KFH's displayed price
    once brokerage fees are folded into the total. The trade must still be
    recognised as already-in-Saham using date/type/symbol/quantity/amount."""
    suffix = str(time.time_ns())
    symbol = f"D{suffix[-7:]}"
    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-07-06', 'Buy', 283287, 32628.872, 0, 0, 0, NULL,
                   '', 'portfolio', 'MANUAL', 0, ?)""",
        (symbol, int(time.time())),
    )

    batch = _preview(
        test_client,
        auth_headers,
        f"price-drift-{suffix}",
        [
            _record(
                f"REF-{suffix}",
                "BUY",
                date="2026-07-06",
                symbol=symbol,
                quantity="283287",
                price="0.115",
                amount="32628.872",
            )
        ],
    )

    assert batch["counts"]["NEW"] == 0
    assert batch["counts"]["MATCHED_EXISTING"] == 1
    matched = next(item for item in batch["items"] if item["classification"] == "MATCHED_EXISTING")
    assert matched["match_type"] == "FINGERPRINT_EXACT"


def test_manual_sell_with_zero_price_override_matches(test_client, auth_headers):
    """Legacy manual rows sometimes stored price_override as a literal 0
    instead of NULL. A real trade can never execute at zero price, so a
    stored 0 must be treated as unset rather than a required price match."""
    suffix = str(time.time_ns())
    symbol = f"Z{suffix[-7:]}"
    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-07-27', 'Sell', 72260, 0, 19993.004, 0, 0, 0.0,
                   NULL, 'portfolio', 'MANUAL', 0, ?)""",
        (symbol, int(time.time())),
    )

    batch = _preview(
        test_client,
        auth_headers,
        f"zero-price-override-{suffix}",
        [
            _record(
                f"REF-{suffix}",
                "SELL",
                date="2026-07-27",
                symbol=symbol,
                quantity="72260",
                price="0.277",
                amount="19993.004",
            )
        ],
    )

    assert batch["counts"]["NEW"] == 0
    assert batch["counts"]["MATCHED_EXISTING"] == 1


def test_amount_within_kwd_tolerance_still_matches(test_client, auth_headers):
    """A 0.001 KWD rounding difference between Saham and KFH must still be
    treated as the same trade, consistent with the statement reconciliation
    tolerance, rather than silently duplicated as NEW."""
    suffix = str(time.time_ns())
    symbol = f"T{suffix[-7:]}"
    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-05-13', 'Sell', 129870, 0, 20743.030, 0, 0, NULL,
                   '', 'portfolio', 'MANUAL', 0, ?)""",
        (symbol, int(time.time())),
    )

    batch = _preview(
        test_client,
        auth_headers,
        f"tolerance-match-{suffix}",
        [
            _record(
                f"REF-{suffix}",
                "SELL",
                date="2026-05-13",
                symbol=symbol,
                quantity="129870",
                price="0.160",
                amount="20743.031",
            )
        ],
    )

    assert batch["counts"]["NEW"] == 0
    assert batch["counts"]["MATCHED_EXISTING"] == 1
    matched = next(item for item in batch["items"] if item["classification"] == "MATCHED_EXISTING")
    assert matched["match_type"] == "MATCHED_WITHIN_TOLERANCE"


def test_same_identity_different_amount_is_conflict_not_new(test_client, auth_headers):
    """Same date/type/symbol/quantity but an amount outside tolerance is a
    genuine discrepancy - it must be surfaced for review, not silently
    imported a second time as a duplicate NEW row."""
    suffix = str(time.time_ns())
    symbol = f"C{suffix[-7:]}"
    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, price_override,
            reference, category, source, is_deleted, created_at)
           VALUES (1, 'KFH', ?, '2026-06-08', 'Buy', 10260, 3765.420, 0, 0, 0, NULL,
                   '', 'portfolio', 'MANUAL', 0, ?)""",
        (symbol, int(time.time())),
    )

    batch = _preview(
        test_client,
        auth_headers,
        f"real-discrepancy-{suffix}",
        [
            _record(
                f"REF-{suffix}",
                "BUY",
                date="2026-06-08",
                symbol=symbol,
                quantity="10260",
                price="0.367",
                amount="3769.685",
            )
        ],
    )

    assert batch["counts"]["NEW"] == 0
    assert batch["counts"]["CONFLICT"] == 1


def test_unsupported_row_reconciles_into_total_other(test_client, auth_headers):
    """A row type KFH doesn't break out (transfer fee, custody charge, etc.)
    must reconcile against totOther instead of being dropped from every
    bucket, which previously made any statement with a nonzero totOther
    unconditionally fail reconciliation."""
    suffix = str(time.time_ns())
    batch = _preview(
        test_client,
        auth_headers,
        f"other-bucket-{suffix}",
        [
            _record(f"DEPOSIT-{suffix}", amount="500.000"),
            _record(
                f"FEE-{suffix}",
                "STOCK_TRANSFER_FEE",
                amount="2.500",
                status="unsupported",
            ),
        ],
        statement_summary=_summary(
            total_deposit="500.000",
            total_other="2.500",
        ),
    )

    assert batch["status"] == "PREVIEW_READY"
    assert batch["reconciliation"]["status"] == "PASSED"
    assert batch["reconciliation"]["comparisons"]["total_other"]["matches"] is True


def test_history_outside_requested_window_does_not_break_reconciliation(
    test_client, auth_headers
):
    """KFH's reported statement totals describe only the requested window.
    `records` legitimately carries older history too (smart incremental sync
    looks further back for duplicate/matched detection, and the live
    connector has been observed returning rows beyond the requested range
    regardless). Older rows must never be summed into the totals check."""
    suffix = str(time.time_ns())
    batch = _preview(
        test_client,
        auth_headers,
        f"window-{suffix}",
        [
            _record(
                f"IN-WINDOW-{suffix}",
                "BUY",
                date="2026-08-20",
                symbol="NBK",
                quantity="10",
                price="1",
                amount="10",
            ),
            _record(
                f"OUT-OF-WINDOW-{suffix}",
                "BUY",
                date="2026-04-07",
                symbol="HUMANSOFT",
                quantity="500",
                price="2",
                amount="1000",
            ),
        ],
        statement_summary=_summary(total_buy="10.000"),
        scope={
            "mode": "CUSTOM_RANGE",
            "fromDate": "2026-08-05",
            "toDate": "2026-09-05",
            "mergePolicy": "ADD_MISSING_ONLY",
        },
    )

    assert batch["status"] == "PREVIEW_READY"
    assert batch["reconciliation"]["status"] == "PASSED"
    assert batch["reconciliation"]["comparisons"]["total_buy"] == {
        "reported": "10.000",
        "parsed": "10.000",
        "difference": "0.000",
        "matches": True,
    }
    # The old row is still returned for classification/matching - it's only
    # excluded from the totals sum, not dropped from the batch.
    assert batch["counts"]["NEW"] == 2


def test_reconciliation_failure_blocks_confirmation(test_client, auth_headers):
    # A short (<=35 day) requested window is where live evidence shows KFH's
    # reported summary reliably describes the same period as the fetched
    # rows, so a mismatch here stays a hard, blocking error.
    suffix = str(time.time_ns())
    reference = f"BAD-TOTAL-{suffix}"
    batch = _preview(
        test_client,
        auth_headers,
        f"bad-total-{suffix}",
        [_record(reference, "BUY", date="2026-08-20", symbol="NBK", quantity="10", price="1", amount="10")],
        statement_summary=_summary(total_buy="12.000"),
        scope={
            "mode": "CUSTOM_RANGE",
            "fromDate": "2026-08-05",
            "toDate": "2026-09-05",
            "mergePolicy": "ADD_MISSING_ONLY",
        },
    )

    assert batch["status"] == "RECONCILIATION_FAILED"
    assert batch["reconciliation"]["status"] == "SYNC_RECONCILIATION_FAILED"
    assert batch["counts"]["ERRORS"] == 1
    response = _confirm(test_client, auth_headers, batch)
    assert response.status_code == 400
    assert "SYNC_RECONCILIATION_FAILED" in response.text
    assert query_one(
        "SELECT COUNT(*) AS n FROM transactions WHERE user_id = 1 AND reference = ?",
        (reference,),
    )["n"] == 0


def test_wide_window_mismatch_is_advisory_not_blocking(test_client, auth_headers):
    """Real KFH accounts show the reported statement summary only reliably
    covers a short recent period - a multi-month or open-ended fetch (smart
    incremental, full history) can genuinely span more than that summary
    describes even when every parsed row is correct. That mismatch must still
    be visible in full, but must not trap the owner out of saving real,
    correctly matched transactions."""
    suffix = str(time.time_ns())
    reference = f"WIDE-BAD-TOTAL-{suffix}"
    batch = _preview(
        test_client,
        auth_headers,
        f"wide-bad-total-{suffix}",
        [_record(reference, "BUY", date="2026-04-20", symbol="NBK", quantity="10", price="1", amount="10")],
        statement_summary=_summary(total_buy="12.000"),
        scope={
            "mode": "CUSTOM_RANGE",
            "fromDate": "2026-03-05",
            "toDate": "2026-09-05",
            "mergePolicy": "ADD_MISSING_ONLY",
        },
    )

    assert batch["status"] == "PREVIEW_READY"
    assert batch["reconciliation"]["status"] == "ADVISORY_MISMATCH"
    assert batch["reconciliation"]["comparisons"]["total_buy"]["matches"] is False
    assert batch["counts"]["ERRORS"] == 0

    response = _confirm(test_client, auth_headers, batch)
    assert response.status_code == 200, response.text


def test_reported_cash_update_is_opt_in_and_uses_manual_override(test_client, auth_headers):
    suffix = str(time.time_ns())
    existing_cash = query_one(
        "SELECT id FROM portfolio_cash WHERE user_id = 1 AND portfolio = 'KFH'"
    )
    if existing_cash:
        exec_sql(
            """UPDATE portfolio_cash
               SET balance = 77.000, currency = 'KWD', manual_override = 1, last_updated = ?
               WHERE id = ?""",
            (int(time.time()), existing_cash["id"]),
        )
    else:
        exec_sql(
            """INSERT INTO portfolio_cash
               (user_id, portfolio, balance, currency, last_updated, manual_override)
               VALUES (1, 'KFH', 77.000, 'KWD', ?, 1)""",
            (int(time.time()),),
        )

    no_update = _preview(
        test_client,
        auth_headers,
        f"cash-off-{suffix}",
        [],
        statement_summary=_summary(close_balance="125.500"),
    )
    assert no_update["cash_summary"] == {
        "currency": "KWD",
        "reported_opening_cash": "100.000",
        "reported_closing_cash": "125.500",
        "current_saham_cash": "77.000",
        "difference": "48.500",
        "manual_override": True,
        "update_selected_default": False,
    }
    unchanged = _confirm(test_client, auth_headers, no_update)
    assert unchanged.status_code == 200
    assert query_one(
        "SELECT balance, manual_override FROM portfolio_cash WHERE user_id = 1 AND portfolio = 'KFH'"
    )["balance"] == 77.0

    opted_in = _preview(
        test_client,
        auth_headers,
        f"cash-on-{suffix}",
        [],
        statement_summary=_summary(close_balance="125.500"),
    )
    updated = _confirm(
        test_client,
        auth_headers,
        opted_in,
        update_cash_balance=True,
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["cash_balance_updated"] is True
    cash = query_one(
        "SELECT balance, manual_override FROM portfolio_cash WHERE user_id = 1 AND portfolio = 'KFH'"
    )
    assert cash["balance"] == 125.5
    assert cash["manual_override"] == 1


def test_backend_schema_contains_no_kfh_password_column(test_client):
    broker_tables = (
        "broker_connections",
        "broker_import_batches",
        "broker_raw_transactions",
        "broker_unsettled_transactions",
        "broker_import_items",
        "broker_transaction_links",
    )
    forbidden = ("password", "session_token", "plaintext_token", "access_token", "refresh_token")
    for table in broker_tables:
        columns = query_all(f"PRAGMA table_info('{table}')")
        names = {str(column["name"]).lower() for column in columns}
        assert not names.intersection(forbidden), (table, names.intersection(forbidden))


def test_raw_broker_payload_is_redacted_before_persistence(test_client, auth_headers):
    suffix = str(time.time_ns())
    record = _record(f"REDACT-{suffix}")
    record["raw_payload"] = {
        "trnsRef": record["broker_transaction_ref"],
        "username": "private-user",
        "password": "private-password",
        "nested": {
            "sesnId": "private-session",
            "ssoToken": "private-token",
            "secAccNum": "private-account",
        },
    }
    _preview(test_client, auth_headers, f"redact-{suffix}", [record])
    stored = query_one(
        """SELECT raw_payload FROM broker_raw_transactions
           WHERE user_id = 1 AND broker_transaction_ref = ?""",
        (record["broker_transaction_ref"],),
    )
    payload = json.loads(stored["raw_payload"])
    assert payload == {
        "trnsRef": record["broker_transaction_ref"],
        "username": "[REDACTED]",
        "password": "[REDACTED]",
        "nested": {
            "sesnId": "[REDACTED]",
            "ssoToken": "[REDACTED]",
            "secAccNum": "[REDACTED]",
        },
    }


def test_connection_batch_raw_and_unsettled_audit_lifecycle(test_client, auth_headers):
    suffix = str(time.time_ns())
    account = f"KFH-SEC-{suffix[-8:]}3658"
    reference = f"AUDIT-{suffix}"
    record = _record(
        reference,
        "BUY",
        date="2026-08-31",
        symbol=f"AU{suffix[-6:]}",
        quantity="4",
        price="2.500",
        amount="10.000",
    )
    record.update(
        {
            "raw_transaction_type": "Buy|شراء",
            "raw_description": "Captured KFH purchase",
            "raw_date": "20260831",
            "parser_version": "kfh-normalizer-v1",
            "adapter_version": "kfh-cashlog-adapter-v1",
        }
    )
    unsettled = {
        "broker_transaction_ref": f"UNSETTLED-{suffix}",
        "raw_transaction_type": "Pending Buy",
        "raw_description": "Not finalized",
        "raw_date": "20260901",
        "raw_payload": {
            "trnsRef": f"UNSETTLED-{suffix}",
            "sesnId": "must-not-persist",
        },
        "parser_version": "kfh-normalizer-v1",
        "adapter_version": "kfh-cashlog-adapter-v1",
    }
    before = query_one(
        "SELECT COUNT(*) AS n FROM transactions WHERE user_id = 1 AND reference = ?",
        (reference,),
    )["n"]

    batch = _preview(
        test_client,
        auth_headers,
        account,
        [record],
        unsettled_records=[unsettled],
        scope={
            "mode": "CUSTOM_RANGE",
            "fromDate": "2026-08-01",
            "toDate": "2026-09-01",
            "mergePolicy": "ADD_MISSING_ONLY",
        },
    )
    assert batch["status"] == "PREVIEW_READY"
    assert batch["mode"] == "CUSTOM_RANGE"
    assert batch["requested_from"] == "2026-08-01"
    assert batch["requested_to"] == "2026-09-01"
    assert batch["fetched_count"] == 1
    assert batch["unsettled_count"] == 1
    assert query_one(
        "SELECT COUNT(*) AS n FROM transactions WHERE user_id = 1 AND reference = ?",
        (reference,),
    )["n"] == before

    raw = query_one(
        """SELECT sync_batch_id, raw_transaction_type, raw_description, raw_date,
                  raw_hash, parser_version, adapter_version
           FROM broker_raw_transactions
           WHERE user_id = 1 AND broker_transaction_ref = ?""",
        (reference,),
    )
    assert raw["sync_batch_id"] == batch["batch_id"]
    assert raw["raw_transaction_type"] == "Buy|شراء"
    assert raw["raw_description"] == "Captured KFH purchase"
    assert raw["raw_date"] == "20260831"
    assert len(raw["raw_hash"]) == 64
    assert raw["parser_version"] == "kfh-normalizer-v1"
    assert raw["adapter_version"] == "kfh-cashlog-adapter-v1"

    pending = query_one(
        """SELECT status, raw_payload, raw_hash, parser_version, adapter_version
           FROM broker_unsettled_transactions
           WHERE user_id = 1 AND sync_batch_id = ?""",
        (batch["batch_id"],),
    )
    assert pending["status"] == "UNSETTLED"
    assert json.loads(pending["raw_payload"])["sesnId"] == "[REDACTED]"
    assert len(pending["raw_hash"]) == 64

    connection_response = test_client.get(
        "/api/v1/kfh-sync/connection",
        headers=auth_headers,
    )
    assert connection_response.status_code == 200
    connection = connection_response.json()["data"]
    assert set(connection) == {
        "id",
        "broker",
        "account_label",
        "auth_mode",
        "status",
        "created_at",
        "last_connected_at",
        "last_successful_sync",
        "last_sync_status",
    }
    assert connection["broker"] == "KFH"
    assert connection["account_label"] == "KFH ••••3658"
    assert connection["auth_mode"] == "LOCAL_BROWSER_SESSION"
    assert connection["status"] == "PREVIEW_READY"

    committed = _confirm(test_client, auth_headers, batch)
    assert committed.status_code == 200, committed.text
    assert committed.json()["data"]["status"] == "COMMITTED"
    transaction = query_one(
        "SELECT source FROM transactions WHERE user_id = 1 AND reference = ?",
        (reference,),
    )
    assert transaction["source"] == "KFH_LIVE_SYNC"
    stored_batch = query_one(
        """SELECT status, started_at, fetched_at, confirmed_at, committed_at,
                  new_count, duplicate_count, matched_count, conflict_count,
                  unsupported_count
           FROM broker_import_batches WHERE id = ?""",
        (batch["batch_id"],),
    )
    assert stored_batch["status"] == "COMMITTED"
    assert all(
        stored_batch[field] is not None
        for field in ("started_at", "fetched_at", "confirmed_at", "committed_at")
    )
    assert stored_batch["new_count"] == 1

    connection = test_client.get(
        "/api/v1/kfh-sync/connection",
        headers=auth_headers,
    ).json()["data"]
    assert connection["status"] == "READY"
    assert connection["last_sync_status"] == "COMMITTED"
    assert connection["last_successful_sync"]

    disconnected = test_client.delete(
        f"/api/v1/kfh-sync/connection/{connection['id']}",
        headers=auth_headers,
    )
    assert disconnected.status_code == 200
    assert disconnected.json()["data"]["status"] == "DISCONNECTED"


def test_distinct_references_with_identical_material_fields_are_both_added(
    test_client, auth_headers
):
    suffix = str(time.time_ns())
    records = [
        _record(f"SAME-DAY-A-{suffix}", amount="12.345"),
        _record(f"SAME-DAY-B-{suffix}", amount="12.345"),
    ]
    batch = _preview(test_client, auth_headers, f"same-day-{suffix}", records)
    assert batch["counts"]["NEW"] == 2
    committed = _confirm(test_client, auth_headers, batch)
    assert committed.status_code == 200
    assert committed.json()["data"]["added"] == 2


def test_all_existing_preview_commits_nothing(test_client, auth_headers):
    suffix = str(time.time_ns())
    account = f"all-existing-{suffix}"
    records = [_record(f"EXISTING-A-{suffix}"), _record(f"EXISTING-B-{suffix}")]
    first = _preview(test_client, auth_headers, account, records)
    assert _confirm(test_client, auth_headers, first).json()["data"]["added"] == 2

    second = _preview(test_client, auth_headers, account, records)
    assert second["counts"]["NEW"] == 0
    assert second["counts"]["ALREADY_IN_SAHAM"] == 2
    result = _confirm(test_client, auth_headers, second, [])
    assert result.status_code == 200
    assert result.json()["data"]["added"] == 0


def test_feature_gate_and_merge_policy_fail_closed(test_client, auth_headers, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "KFH_AUTO_SYNC_ENABLED", False)
    monkeypatch.setattr(settings, "KFH_LOCAL_TEST_ENABLED", False)
    disabled = test_client.get("/api/v1/kfh-sync/connection", headers=auth_headers)
    assert disabled.status_code == 404

    monkeypatch.setattr(settings, "KFH_LOCAL_TEST_ENABLED", True)
    local_headers = {**auth_headers, "Origin": "http://localhost:8081"}
    local_manual = test_client.get(
        "/api/v1/kfh-sync/connection", headers=local_headers
    )
    assert local_manual.status_code == 200
    foreign_origin = test_client.get(
        "/api/v1/kfh-sync/connection",
        headers={**auth_headers, "Origin": "http://foreign.example"},
    )
    assert foreign_origin.status_code == 404
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    production = test_client.get(
        "/api/v1/kfh-sync/connection", headers=local_headers
    )
    assert production.status_code == 404
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")

    monkeypatch.setattr(settings, "KFH_AUTO_SYNC_ENABLED", True)
    rejected = test_client.post(
        "/api/v1/kfh-sync/batches/preview",
        headers=auth_headers,
        json={
            "broker_account": "policy-test",
            "scope": {"mode": "SMART_INCREMENTAL", "mergePolicy": "OVERWRITE"},
            "records": [],
        },
    )
    assert rejected.status_code == 422


def test_supported_trade_cash_and_corporate_action_records_commit_together(
    test_client, auth_headers
):
    suffix = str(time.time_ns())
    symbol = f"T{suffix[-7:]}"
    records = [
        _record(
            f"BUY-{suffix}",
            "BUY",
            symbol=symbol,
            quantity="2",
            price="1.250",
            amount="2.500",
        ),
        _record(
            f"SELL-{suffix}",
            "SELL",
            symbol=symbol,
            quantity="2",
            price="1.300",
            amount="2.600",
        ),
        _record(f"DIVIDEND-{suffix}", "CASH_DIVIDEND", symbol=symbol, amount="0.125"),
        _record(f"DEPOSIT-{suffix}", "DEPOSIT", amount="50.000"),
        _record(f"CORPORATE-{suffix}", "CASH_DIVIDEND", symbol=symbol, amount="0.250"),
    ]
    records[-1]["description"] = "Deposit by Corporate Action"
    records[0]["transaction_timestamp"] = "2026-08-01T09:08:38+03:00"

    batch = _preview(test_client, auth_headers, f"supported-types-{suffix}", records)
    assert batch["counts"]["NEW"] == 5
    committed = _confirm(test_client, auth_headers, batch)
    assert committed.status_code == 200, committed.text
    assert committed.json()["data"]["added"] == 5
    stored = query_one(
        "SELECT transaction_timestamp FROM broker_raw_transactions WHERE broker_transaction_ref = ?",
        (f"BUY-{suffix}",),
    )
    assert stored["transaction_timestamp"] == "2026-08-01T09:08:38+03:00"
