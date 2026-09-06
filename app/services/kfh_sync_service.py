"""Two-phase KFH import staging, matching, linkage, and atomic commit."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any

from app.core.database import (
    exec_sql,
    exec_sql_returning_id,
    query_all,
    query_one,
    safe_json_dumps,
    transaction,
)
from app.core.exceptions import BadRequestError, NotFoundError
from app.schemas.kfh_sync import KfhBrokerRecordInput, KfhPreviewRequest
from app.services.portfolio_service import PortfolioService

BROKER = "KFH"
PORTFOLIO = "KFH"
KWD_TOLERANCE = Decimal("0.001")
# KFH's own reported statement summary has been observed (live, real-account
# evidence) to reliably describe only a short/recent window. A one-month
# custom fetch reconciled exactly; multi-month fetches consistently showed a
# large, non-blocking-safe gap even though the underlying transaction rows
# were individually correct. Past this many days, the reported summary can no
# longer be trusted as a hard gate - the mismatch is still surfaced in full,
# it just stops blocking confirmation.
RECONCILIATION_STRICT_MAX_WINDOW_DAYS = 35
SUPPORTED_TYPES = {"BUY", "SELL", "CASH_DIVIDEND", "DEPOSIT", "WITHDRAWAL"}
CLASSIFICATIONS = ("NEW", "EXACT_DUPLICATE", "MATCHED_EXISTING", "CONFLICT", "UNSUPPORTED")
CONNECTION_STATES = (
    "DISCONNECTED",
    "CONNECTING",
    "LOGIN_REQUIRED",
    "AUTHENTICATING",
    "OTP_REQUIRED",
    "AUTHENTICATED",
    "READY",
    "SYNCING",
    "PREVIEW_READY",
    "REAUTH_REQUIRED",
    "SESSION_EXPIRED",
    "PARTIAL_SYNC",
    "RECONCILIATION_REQUIRED",
    "ERROR",
)
SECRET_PAYLOAD_KEYS = {
    "password",
    "kfhpassword",
    "brokerpassword",
    "username",
    "kfhusername",
    "otp",
    "onetimepassword",
    "sesnid",
    "sessionid",
    "ssotoken",
    "applicationtoken",
    "token",
    "authtoken",
    "accesstoken",
    "refreshtoken",
    "sessiontoken",
    "authenticationstate",
    "brokeraccount",
    "brokeraccountid",
    "secaccnum",
}
logger = logging.getLogger("app.kfh_sync")
SAFE_OPERATIONAL_FIELDS = {
    "user_id",
    "batch_id",
    "record_count",
    "unsettled_count",
    "selected_count",
    "new_count",
    "existing_count",
    "review_count",
    "added_count",
    "error_code",
}


def _log_kfh_event(event: str, **fields: Any) -> None:
    safe_fields = {key: value for key, value in fields.items() if key in SAFE_OPERATIONAL_FIELDS}
    logger.info("%s %s", event, safe_json_dumps(safe_fields))


def _row_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in list(row.keys())}


def _redact_broker_secrets(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_broker_secrets(item) for item in value]
    if not isinstance(value, dict):
        return value
    redacted = {}
    for key, child in value.items():
        normalized_key = "".join(character for character in key.lower() if character.isalnum())
        redacted[key] = (
            "[REDACTED]"
            if normalized_key in SECRET_PAYLOAD_KEYS
            else _redact_broker_secrets(child)
        )
    return redacted


def _normalize_type(value: str) -> str:
    normalized = "_".join(value.strip().upper().replace("-", " ").split())
    aliases = {
        "DIVIDEND": "CASH_DIVIDEND",
        "DIVIDEND_ONLY": "CASH_DIVIDEND",
        "CASH_DIVIDENDS": "CASH_DIVIDEND",
        "WITHDRAW": "WITHDRAWAL",
    }
    return aliases.get(normalized, normalized)


def _decimal_text(value: Decimal | float | int | str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        decimal = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError) as exc:
        raise BadRequestError(f"Invalid KFH decimal value: {value}") from exc
    if not decimal.is_finite():
        raise BadRequestError("KFH numeric values must be finite")
    return format(decimal, "f")


def _positive_decimal_text(value: Decimal | float | int | str | None) -> str | None:
    text = _decimal_text(value)
    if text is None:
        return None
    return _decimal_text(abs(Decimal(text)))


def _money_text(value: Decimal | float | int | str) -> str:
    """Canonical KWD display/storage value with three fractional digits."""
    return format(Decimal(str(value)).quantize(KWD_TOLERANCE, rounding=ROUND_HALF_EVEN), "f")


def _money(value: Decimal | float | int | str | None) -> Decimal:
    return Decimal(str(value or 0))


def _to_legacy_portfolio_number(value: Decimal) -> float:
    """Explicit boundary for the pre-existing float-based cash recalculation API."""
    return float(value)


def _account_key(broker_account: str) -> str:
    normalized = broker_account.strip().upper()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _account_label(broker_account: str) -> str:
    normalized = broker_account.strip()
    return f"KFH ••••{normalized[-4:]}" if len(normalized) > 4 else "KFH account"


def _canonical_record(record: KfhBrokerRecordInput) -> dict[str, Any]:
    transaction_type = _normalize_type(record.transaction_type)
    record_kind = "deposit" if transaction_type in {"DEPOSIT", "WITHDRAWAL"} else "transaction"
    return {
        "record_kind": record_kind,
        "transaction_date": record.transaction_date,
        "transaction_timestamp": record.transaction_timestamp,
        "settlement_date": record.settlement_date,
        "transaction_type": transaction_type,
        "symbol": record.symbol.strip().upper() if record.symbol and record.symbol.strip() else None,
        "quantity": _positive_decimal_text(record.quantity),
        "price": _positive_decimal_text(record.price),
        "amount": _positive_decimal_text(record.amount),
        "fees": _positive_decimal_text(record.fees),
        "description": record.description.strip() if record.description else None,
        "interpretation_status": record.interpretation_status,
        "interpretation_error": record.interpretation_error,
    }


def secondary_fingerprint(account_key: str, canonical: dict[str, Any]) -> str:
    """Hash normalized, reliable accounting fields rather than display text."""
    identity = {
        "broker": BROKER,
        "brokerAccount": account_key,
        "transactionDate": canonical["transaction_date"],
        "transactionType": canonical["transaction_type"],
        "symbol": canonical["symbol"],
        "quantity": canonical["quantity"],
        "price": canonical["price"],
        "amount": canonical["amount"],
    }
    if canonical.get("settlement_date"):
        identity["settlementDate"] = canonical["settlement_date"]
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cross_source_fingerprint(account_key: str, canonical: dict[str, Any]) -> str:
    """Identity shared by XLSX/manual rows and live rows.

    Settlement date is deliberately omitted because the legacy XLSX/manual
    destination does not preserve it. The durable raw-record fingerprint above
    still includes settlement date when KFH supplies it.
    """
    identity = {
        "broker": BROKER,
        "brokerAccount": account_key,
        "transactionDate": canonical["transaction_date"],
        "transactionType": canonical["transaction_type"],
        "symbol": canonical["symbol"],
        "quantity": canonical["quantity"],
        "price": canonical["price"],
        "amount": canonical["amount"],
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _unsupported_reason(canonical: dict[str, Any]) -> str | None:
    if canonical["interpretation_status"] != "ready":
        return canonical.get("interpretation_error") or "KFH transaction needs review"
    transaction_type = canonical["transaction_type"]
    if transaction_type not in SUPPORTED_TYPES:
        return f"Unsupported KFH transaction type: {transaction_type}"
    if canonical["amount"] is None:
        return "KFH amount is missing"
    if transaction_type in {"BUY", "SELL"}:
        missing = [
            name
            for name in ("symbol", "quantity", "price")
            if canonical[name] is None
        ]
        if missing:
            return f"KFH trade is missing: {', '.join(missing)}"
    return None


def _get_or_create_connection(user_id: int, broker_account: str, now: int) -> tuple[int, str]:
    account_key = _account_key(broker_account)
    existing = query_one(
        """SELECT id FROM broker_connections
           WHERE user_id = ? AND broker = ? AND broker_account_key = ?""",
        (user_id, BROKER, account_key),
    )
    if existing:
        exec_sql(
            """UPDATE broker_connections
               SET auth_mode = 'LOCAL_BROWSER_SESSION', status = 'SYNCING',
                   last_connected_at = ?, last_sync_status = 'FETCHING', updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (now, now, int(existing["id"]), user_id),
        )
        return int(existing["id"]), account_key
    connection_id = exec_sql_returning_id(
        """INSERT INTO broker_connections
           (user_id, broker, broker_account_key, account_label, auth_mode, status,
            last_connected_at, last_sync_status, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'LOCAL_BROWSER_SESSION', 'SYNCING', ?, 'FETCHING', ?, ?)""",
        (user_id, BROKER, account_key, _account_label(broker_account), now, now, now),
    )
    return connection_id, account_key


def _raw_material_equal(existing: Any, canonical: dict[str, Any]) -> bool:
    text_fields = (
        "record_kind",
        "transaction_date",
        "transaction_timestamp",
        "settlement_date",
        "transaction_type",
        "symbol",
    )
    if not all(existing[field] == canonical[field] for field in text_fields):
        return False
    for field in ("quantity", "price", "amount", "fees"):
        left = existing[field]
        right = canonical[field]
        if left is None or right is None:
            if left is not None or right is not None:
                return False
        elif Decimal(str(left)) != Decimal(str(right)):
            return False
    return True


def _amounts_within_tolerance(
    left: str | None, right: str | None, tolerance: Decimal = KWD_TOLERANCE
) -> bool:
    if left is None or right is None:
        return left == right
    return abs(Decimal(left) - Decimal(right)) <= tolerance


def _existing_core_identity_equal(existing: dict[str, Any], canonical: dict[str, Any]) -> bool:
    """True when a Saham record and a KFH row describe the same trade slot.

    Quantity, date, type and symbol are the reliable identity fields. Price is
    deliberately excluded: Saham stores it as amount / shares (or a manual
    override), which routinely disagrees with KFH's displayed price once fees
    are folded into the total or a manual entry rounds differently - matching
    on it produces false negatives, not false positives.
    """
    identity = ("record_kind", "transaction_date", "transaction_type", "symbol", "quantity")
    if not all(existing.get(field) == canonical.get(field) for field in identity):
        return False
    return not (
        canonical.get("settlement_date")
        and existing.get("settlement_date")
        and existing["settlement_date"] != canonical["settlement_date"]
    )


def _canonical_existing_transaction(row: Any) -> dict[str, Any] | None:
    transaction_type = _normalize_type(str(row["txn_type"] or ""))
    if transaction_type not in {"BUY", "SELL", "CASH_DIVIDEND"}:
        return None
    shares = _positive_decimal_text(row["shares"])
    if transaction_type == "BUY":
        amount = _positive_decimal_text(row["purchase_cost"])
    elif transaction_type == "SELL":
        amount = _positive_decimal_text(row["sell_value"])
    else:
        amount = _positive_decimal_text(row["cash_dividend"])
    price = _positive_decimal_text(row["price_override"])
    if price == _decimal_text(0):
        # A stored literal 0 means "never set" for legacy manual rows, not a
        # real trade executed at zero price - fall through to deriving it.
        price = None
    if price is None and shares not in (None, _decimal_text(0)) and amount is not None:
        price = _decimal_text(Decimal(amount) / Decimal(shares))
    return {
        "record_kind": "transaction",
        "transaction_date": row["txn_date"],
        "settlement_date": None,
        "transaction_type": transaction_type,
        "symbol": str(row["stock_symbol"] or "").strip().upper() or None,
        "quantity": shares,
        "price": price,
        "amount": amount,
        "fees": _positive_decimal_text(row["fees"]),
    }


def _canonical_existing_deposit(row: Any) -> dict[str, Any]:
    return {
        "record_kind": "deposit",
        "transaction_date": row["deposit_date"],
        "settlement_date": None,
        "transaction_type": _normalize_type(str(row["source"] or "DEPOSIT")),
        "symbol": None,
        "quantity": None,
        "price": None,
        "amount": _positive_decimal_text(row["amount"]),
        "fees": None,
    }


def _destination_is_available(user_id: int, raw_id: int, kind: str, record_id: int) -> bool:
    linked = query_one(
        """SELECT broker_raw_transaction_id FROM broker_transaction_links
           WHERE user_id = ? AND saham_record_kind = ? AND saham_record_id = ?
             AND broker_raw_transaction_id <> ?""",
        (user_id, kind, record_id, raw_id),
    )
    return linked is None


def _existing_records(user_id: int) -> list[tuple[str, int, str | None, dict[str, Any]]]:
    records: list[tuple[str, int, str | None, dict[str, Any]]] = []
    transactions = query_all(
        """SELECT id, txn_date, txn_type, stock_symbol, shares, purchase_cost,
                  sell_value, cash_dividend, fees, price_override, reference
           FROM transactions
           WHERE user_id = ? AND COALESCE(NULLIF(TRIM(portfolio), ''), 'KFH') = 'KFH'
             AND COALESCE(category, 'portfolio') = 'portfolio'
             AND COALESCE(is_deleted, 0) = 0""",
        (user_id,),
    )
    for row in transactions:
        canonical = _canonical_existing_transaction(row)
        if canonical:
            records.append(("transaction", int(row["id"]), row["reference"], canonical))

    deposits = query_all(
        """SELECT id, deposit_date, amount, source
           FROM cash_deposits
           WHERE user_id = ? AND portfolio = 'KFH' AND COALESCE(is_deleted, 0) = 0""",
        (user_id,),
    )
    for row in deposits:
        records.append(("deposit", int(row["id"]), None, _canonical_existing_deposit(row)))
    return records


def _find_existing_match(
    user_id: int,
    raw_id: int,
    broker_reference: str,
    account_key: str,
    canonical: dict[str, Any],
) -> dict[str, Any] | None:
    records = _existing_records(user_id)

    for kind, record_id, reference, existing in records:
        if reference != broker_reference:
            continue
        if not _destination_is_available(user_id, raw_id, kind, record_id):
            return {
                "classification": "CONFLICT",
                "reason": "KFH reference is already linked to a different broker record",
            }
        if _existing_core_identity_equal(existing, canonical) and _amounts_within_tolerance(
            existing.get("amount"), canonical.get("amount")
        ):
            return {
                "classification": "MATCHED_EXISTING",
                "kind": kind,
                "record_id": record_id,
                "match_type": "BROKER_REFERENCE_EXACT",
                "confidence": 1.0,
                "reason": "KFH reference matches an existing Saham transaction",
            }
        return {
            "classification": "CONFLICT",
            "kind": kind,
            "record_id": record_id,
            "reason": "Same KFH reference exists with different material fields",
            "saham_value": existing,
            "kfh_value": canonical,
        }

    for kind, record_id, _reference, existing in records:
        if not _destination_is_available(user_id, raw_id, kind, record_id):
            continue
        if not _existing_core_identity_equal(existing, canonical):
            continue
        if _amounts_within_tolerance(existing.get("amount"), canonical.get("amount")):
            exact = existing.get("amount") == canonical.get("amount")
            return {
                "classification": "MATCHED_EXISTING",
                "kind": kind,
                "record_id": record_id,
                "match_type": "FINGERPRINT_EXACT" if exact else "MATCHED_WITHIN_TOLERANCE",
                "confidence": 1.0 if exact else 0.98,
                "reason": (
                    "Canonical fingerprint matches an existing Saham transaction"
                    if exact
                    else f"Matches an existing Saham transaction within the KFH ±{KWD_TOLERANCE} KWD reconciliation tolerance"
                ),
            }
        return {
            "classification": "CONFLICT",
            "kind": kind,
            "record_id": record_id,
            "reason": "Same trade (date, type, symbol, quantity) exists in Saham with a different amount",
            "saham_value": existing,
            "kfh_value": canonical,
        }
    return None


def _insert_link(
    user_id: int,
    raw_id: int,
    kind: str,
    record_id: int,
    match_type: str,
    confidence: float,
    now: int,
) -> None:
    existing = query_one(
        "SELECT id FROM broker_transaction_links WHERE broker_raw_transaction_id = ?",
        (raw_id,),
    )
    if existing:
        return
    exec_sql(
        """INSERT INTO broker_transaction_links
           (user_id, broker_raw_transaction_id, saham_record_kind, saham_record_id,
            match_type, confidence, linked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, raw_id, kind, record_id, match_type, confidence, now),
    )


def _stage_record(
    user_id: int,
    connection_id: int,
    batch_id: int,
    account_key: str,
    record: KfhBrokerRecordInput,
    now: int,
) -> tuple[int, dict[str, Any]]:
    canonical = _canonical_record(record)
    fingerprint = secondary_fingerprint(account_key, canonical)
    raw_payload = _redact_broker_secrets(record.raw_payload or {})
    serialized_raw = json.dumps(
        raw_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    raw_hash = hashlib.sha256(serialized_raw.encode("utf-8")).hexdigest()
    raw_transaction_type = record.raw_transaction_type or record.transaction_type
    raw_description = record.raw_description or record.description
    raw_date = record.raw_date or record.transaction_date
    existing_raw = query_one(
        """SELECT * FROM broker_raw_transactions
           WHERE user_id = ? AND broker_connection_id = ? AND broker_transaction_ref = ?""",
        (user_id, connection_id, record.broker_transaction_ref),
    )

    if existing_raw:
        raw_id = int(existing_raw["id"])
        exec_sql(
            """UPDATE broker_raw_transactions
               SET sync_batch_id = COALESCE(sync_batch_id, ?),
                   raw_transaction_type = COALESCE(raw_transaction_type, ?),
                   raw_description = COALESCE(raw_description, ?),
                   raw_date = COALESCE(raw_date, ?),
                   transaction_timestamp = COALESCE(transaction_timestamp, ?),
                   raw_hash = COALESCE(raw_hash, ?),
                   parser_version = COALESCE(parser_version, ?),
                   adapter_version = COALESCE(adapter_version, ?),
                   updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (
                batch_id,
                raw_transaction_type,
                raw_description,
                raw_date,
                canonical["transaction_timestamp"],
                raw_hash,
                record.parser_version,
                record.adapter_version,
                now,
                raw_id,
                user_id,
            ),
        )
        if existing_raw["secondary_fingerprint"] != fingerprint or not _raw_material_equal(existing_raw, canonical):
            return raw_id, {
                "classification": "CONFLICT",
                "reason": "Same KFH reference was retrieved with different material fields",
                "saham_value": json.loads(existing_raw["canonical_payload"]),
                "kfh_value": canonical,
            }
    else:
        raw_id = exec_sql_returning_id(
            """INSERT INTO broker_raw_transactions
               (user_id, broker_connection_id, sync_batch_id, broker_transaction_ref,
                secondary_fingerprint, record_kind, transaction_date, settlement_date,
                transaction_timestamp, transaction_type, symbol, quantity, price, amount, fees,
                canonical_payload, raw_payload, raw_transaction_type, raw_description,
                raw_date, raw_hash, parser_version, adapter_version, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                connection_id,
                batch_id,
                record.broker_transaction_ref,
                fingerprint,
                canonical["record_kind"],
                canonical["transaction_date"],
                canonical["settlement_date"],
                canonical["transaction_timestamp"],
                canonical["transaction_type"],
                canonical["symbol"],
                canonical["quantity"],
                canonical["price"],
                canonical["amount"],
                canonical["fees"],
                safe_json_dumps(canonical),
                serialized_raw,
                raw_transaction_type,
                raw_description,
                raw_date,
                raw_hash,
                record.parser_version,
                record.adapter_version,
                now,
                now,
            ),
        )

    unsupported = _unsupported_reason(canonical)
    if unsupported:
        return raw_id, {"classification": "UNSUPPORTED", "reason": unsupported}

    current_raw = query_one(
        "SELECT committed_at FROM broker_raw_transactions WHERE id = ? AND user_id = ?",
        (raw_id, user_id),
    )
    if current_raw and current_raw["committed_at"] is not None:
        return raw_id, {
            "classification": "EXACT_DUPLICATE",
            "reason": "Already imported through a previous KFH sync",
        }

    link = query_one(
        """SELECT saham_record_kind, saham_record_id, match_type, confidence
           FROM broker_transaction_links WHERE broker_raw_transaction_id = ?""",
        (raw_id,),
    )
    if link:
        return raw_id, {
            "classification": "MATCHED_EXISTING",
            "kind": link["saham_record_kind"],
            "record_id": int(link["saham_record_id"]),
            "match_type": link["match_type"],
            "confidence": float(link["confidence"]),
            "reason": "This KFH transaction is already linked to Saham",
        }

    match = _find_existing_match(
        user_id,
        raw_id,
        record.broker_transaction_ref,
        account_key,
        canonical,
    )
    if match:
        if match["classification"] == "MATCHED_EXISTING":
            _insert_link(
                user_id,
                raw_id,
                match["kind"],
                match["record_id"],
                match["match_type"],
                match["confidence"],
                now,
            )
        return raw_id, match

    return raw_id, {
        "classification": "NEW",
        "reason": "No matching Saham transaction exists",
    }


def _counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {classification: 0 for classification in CLASSIFICATIONS}
    for item in items:
        counts[item["classification"]] += 1
    counts["ALREADY_IN_SAHAM"] = counts["EXACT_DUPLICATE"] + counts["MATCHED_EXISTING"]
    counts["NEEDS_REVIEW"] = counts["CONFLICT"] + counts["UNSUPPORTED"]
    counts["ERRORS"] = 0
    return counts


def _batch_items(batch_id: int) -> list[dict[str, Any]]:
    rows = query_all(
        """SELECT i.id, i.classification, i.reason, i.matched_record_kind,
                  i.matched_record_id, i.match_type, i.confidence, i.selected_default,
                  i.saham_value_json, i.kfh_value_json,
                  r.broker_transaction_ref, r.transaction_date, r.settlement_date,
                  r.transaction_type, r.symbol, r.quantity, r.price, r.amount,
                  r.fees, r.record_kind
           FROM broker_import_items i
           JOIN broker_raw_transactions r ON r.id = i.broker_raw_transaction_id
           WHERE i.batch_id = ? ORDER BY i.id""",
        (batch_id,),
    )
    items = []
    for row in rows:
        item = _row_dict(row)
        item["saham_value"] = (
            json.loads(item.pop("saham_value_json"))
            if item["saham_value_json"]
            else None
        )
        item["kfh_value"] = (
            json.loads(item.pop("kfh_value_json"))
            if item["kfh_value_json"]
            else None
        )
        for field in ("quantity", "price", "amount", "fees"):
            item[field] = _decimal_text(item[field])
        items.append(item)
    return items


def _reconcile_statement(request: KfhPreviewRequest) -> dict[str, Any]:
    """Compare KFH evidence with independently parsed totals using Decimal only."""
    if request.statement_summary is None:
        return {
            "status": "NOT_PROVIDED",
            "message": None,
            "currency": "KWD",
            "tolerance": _money_text(KWD_TOLERANCE),
            "comparisons": {},
        }

    parsed = {
        "total_deposit": Decimal(0),
        "total_withdrawal": Decimal(0),
        "total_buy": Decimal(0),
        "total_sell": Decimal(0),
        "total_other": Decimal(0),
        "vat_amount": Decimal(0),
    }
    type_to_total = {
        "DEPOSIT": "total_deposit",
        "WITHDRAWAL": "total_withdrawal",
        "BUY": "total_buy",
        "SELL": "total_sell",
        # Real KFH cash-statement evidence places positive corporate-action
        # cash dividends in totDeposit, not totOther.  The row remains a
        # CASH_DIVIDEND for Saham transaction semantics; this mapping is only
        # for reconciling KFH's statement summary buckets.
        "CASH_DIVIDEND": "total_deposit",
    }
    # KFH's reported summary totals describe only the requested window, but
    # `records` legitimately carries extra history outside it too (smart
    # incremental sync deliberately looks further back for duplicate/matched
    # detection, and the live connector has been observed returning rows
    # beyond the requested range regardless). Summing everything against a
    # narrower reported total makes reconciliation fail on real, correctly
    # imported history that was never part of what's being reported here.
    scope_from = request.scope.fromDate
    scope_to = request.scope.toDate
    for record in request.records:
        canonical = _canonical_record(record)
        transaction_date = canonical["transaction_date"]
        if scope_from and transaction_date < scope_from:
            continue
        if scope_to and transaction_date > scope_to:
            continue
        # Any type KFH doesn't break out into its own bucket (transfer fees,
        # custody charges, corporate actions, etc.) lands in totOther on the
        # real statement, so unmapped types must reconcile there too rather
        # than being silently dropped from every total.
        total_name = type_to_total.get(canonical["transaction_type"], "total_other")
        if canonical["amount"] is not None:
            parsed[total_name] += abs(_money(canonical["amount"]))
        if canonical["fees"] is not None:
            parsed["vat_amount"] += abs(_money(canonical["fees"]))

    summary = request.statement_summary
    reported = {
        "total_deposit": abs(summary.total_deposit),
        "total_withdrawal": abs(summary.total_withdrawal),
        "total_buy": abs(summary.total_buy),
        "total_sell": abs(summary.total_sell),
        "total_other": abs(summary.total_other),
        "vat_amount": abs(summary.vat_amount),
    }
    comparisons: dict[str, dict[str, Any]] = {}
    failed = False
    for name, reported_value in reported.items():
        difference = parsed[name] - reported_value
        matches = abs(difference) <= KWD_TOLERANCE
        failed = failed or not matches
        comparisons[name] = {
            "reported": _money_text(reported_value),
            "parsed": _money_text(parsed[name]),
            "difference": _money_text(difference),
            "matches": matches,
        }

    window_days = _scope_window_days(request.scope)
    strict = window_days is not None and window_days <= RECONCILIATION_STRICT_MAX_WINDOW_DAYS
    if failed and not strict:
        status = "ADVISORY_MISMATCH"
        window_label = f"{window_days}-day" if window_days is not None else "open-ended"
        message = (
            f"KFH totals and parsed transaction totals do not match for this {window_label} "
            "statement window. KFH's reported summary reliably covers only a short recent "
            "period, so a mismatch this wide does not block saving - review the breakdown below."
        )
    else:
        status = "SYNC_RECONCILIATION_FAILED" if failed else "PASSED"
        message = (
            "KFH totals and parsed transaction totals do not match. Review required."
            if failed
            else "KFH totals match parsed transaction totals."
        )
    return {
        "status": status,
        "message": message,
        "currency": summary.currency,
        "tolerance": _money_text(KWD_TOLERANCE),
        "comparisons": comparisons,
    }


def _scope_window_days(scope: Any) -> int | None:
    if not scope.fromDate or not scope.toDate:
        return None
    return (date.fromisoformat(scope.toDate) - date.fromisoformat(scope.fromDate)).days


def _cash_summary(user_id: int, request: KfhPreviewRequest) -> dict[str, Any] | None:
    if request.statement_summary is None:
        return None
    row = query_one(
        "SELECT balance, manual_override FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
        (user_id, PORTFOLIO),
    )
    current = _money(row["balance"] if row else 0)
    reported = request.statement_summary.close_balance
    return {
        "currency": request.statement_summary.currency,
        "reported_opening_cash": _money_text(request.statement_summary.open_balance),
        "reported_closing_cash": _money_text(reported),
        "current_saham_cash": _money_text(current),
        "difference": _money_text(reported - current),
        "manual_override": bool(row["manual_override"]) if row else False,
        "update_selected_default": False,
    }


def create_preview(user_id: int, request: KfhPreviewRequest) -> dict[str, Any]:
    now = int(time.time())
    _log_kfh_event(
        "KFH_SYNC_STARTED",
        user_id=user_id,
        record_count=len(request.records),
        unsettled_count=len(request.unsettled_records),
    )
    with transaction():
        connection_id, account_key = _get_or_create_connection(user_id, request.broker_account, now)
        scope = request.scope.model_dump()
        mode = request.scope.mode
        requested_from = request.scope.fromDate
        requested_to = request.scope.toDate
        summary = request.statement_summary
        batch_id = exec_sql_returning_id(
            """INSERT INTO broker_import_batches
               (user_id, broker_connection_id, status, mode, requested_from,
                requested_to, started_at, fetched_at, scope_json, fetched_count,
                unsettled_count, counts_json, kfh_open_balance, kfh_close_balance,
                kfh_total_buy, kfh_total_sell, kfh_total_deposit,
                kfh_total_withdrawal, created_at)
               VALUES (?, ?, 'FETCHED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                connection_id,
                mode,
                requested_from,
                requested_to,
                now,
                now,
                safe_json_dumps(scope),
                len(request.records),
                len(request.unsettled_records),
                safe_json_dumps({}),
                _money_text(summary.open_balance) if summary else None,
                _money_text(summary.close_balance) if summary else None,
                _money_text(summary.total_buy) if summary else None,
                _money_text(summary.total_sell) if summary else None,
                _money_text(summary.total_deposit) if summary else None,
                _money_text(summary.total_withdrawal) if summary else None,
                now,
            ),
        )
        staged: list[tuple[int, dict[str, Any]]] = []
        staged_index_by_raw_id: dict[int, int] = {}
        for record in request.records:
            raw_id, result = _stage_record(
                user_id,
                connection_id,
                batch_id,
                account_key,
                record,
                now,
            )
            previous_index = staged_index_by_raw_id.get(raw_id)
            if previous_index is not None:
                if result["classification"] == "CONFLICT":
                    staged[previous_index] = (raw_id, result)
                continue
            staged_index_by_raw_id[raw_id] = len(staged)
            staged.append((raw_id, result))

        item_results = [result for _, result in staged]
        counts = _counts(item_results)
        for unsettled in request.unsettled_records:
            redacted_payload = _redact_broker_secrets(unsettled.raw_payload)
            serialized_payload = json.dumps(
                redacted_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            raw_hash = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
            exec_sql(
                """INSERT INTO broker_unsettled_transactions
                   (user_id, sync_batch_id, broker_connection_id,
                    broker_transaction_ref, raw_transaction_type, raw_description,
                    raw_date, raw_payload, raw_hash, parser_version, adapter_version,
                    status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'UNSETTLED', ?)""",
                (
                    user_id,
                    batch_id,
                    connection_id,
                    unsettled.broker_transaction_ref,
                    unsettled.raw_transaction_type,
                    unsettled.raw_description,
                    unsettled.raw_date,
                    serialized_payload,
                    raw_hash,
                    unsettled.parser_version,
                    unsettled.adapter_version,
                    now,
                ),
            )
        reconciliation = _reconcile_statement(request)
        cash_summary = _cash_summary(user_id, request)
        batch_status = (
            "RECONCILIATION_FAILED"
            if reconciliation["status"] == "SYNC_RECONCILIATION_FAILED"
            else "PREVIEW_READY"
        )
        if batch_status == "RECONCILIATION_FAILED":
            counts["ERRORS"] += 1
        exec_sql(
            """UPDATE broker_import_batches
               SET status = ?, counts_json = ?, reconciliation_json = ?,
                   cash_summary_json = ?, new_count = ?, duplicate_count = ?,
                   matched_count = ?, conflict_count = ?, unsupported_count = ?
               WHERE id = ? AND user_id = ?""",
            (
                batch_status,
                safe_json_dumps(counts),
                safe_json_dumps(reconciliation),
                safe_json_dumps(cash_summary) if cash_summary is not None else None,
                counts["NEW"],
                counts["EXACT_DUPLICATE"],
                counts["MATCHED_EXISTING"],
                counts["CONFLICT"],
                counts["UNSUPPORTED"],
                batch_id,
                user_id,
            ),
        )
        connection_status = (
            "RECONCILIATION_REQUIRED"
            if batch_status == "RECONCILIATION_FAILED"
            else "PREVIEW_READY"
        )
        exec_sql(
            """UPDATE broker_connections
               SET status = ?, last_sync_status = ?, updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (connection_status, batch_status, now, connection_id, user_id),
        )
        for raw_id, result in staged:
            exec_sql(
                """INSERT INTO broker_import_items
                   (batch_id, broker_raw_transaction_id, classification, reason,
                    matched_record_kind, matched_record_id, match_type, confidence,
                    saham_value_json, kfh_value_json, selected_default, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id,
                    raw_id,
                    result["classification"],
                    result.get("reason"),
                    result.get("kind"),
                    result.get("record_id"),
                    result.get("match_type"),
                    result.get("confidence"),
                    (
                        safe_json_dumps(result["saham_value"])
                        if result.get("saham_value") is not None
                        else None
                    ),
                    (
                        safe_json_dumps(result["kfh_value"])
                        if result.get("kfh_value") is not None
                        else None
                    ),
                    1 if result["classification"] == "NEW" else 0,
                    now,
                ),
            )

    if batch_status == "RECONCILIATION_FAILED":
        _log_kfh_event("KFH_RECONCILIATION_FAILED", batch_id=batch_id, error_code=batch_status)
    _log_kfh_event(
        "KFH_PREVIEW_READY",
        batch_id=batch_id,
        new_count=counts["NEW"],
        existing_count=counts["ALREADY_IN_SAHAM"],
        review_count=counts["NEEDS_REVIEW"],
    )
    return {
        "batch_id": batch_id,
        "status": batch_status,
        "fetched_count": len(request.records),
        "unsettled_count": len(request.unsettled_records),
        "mode": mode,
        "requested_from": requested_from,
        "requested_to": requested_to,
        "counts": counts,
        "items": _batch_items(batch_id),
        "reconciliation": reconciliation,
        "cash_summary": cash_summary,
        "financial_data_changed": False,
    }


def get_preview(user_id: int, batch_id: int) -> dict[str, Any]:
    batch = query_one(
        "SELECT * FROM broker_import_batches WHERE id = ? AND user_id = ?",
        (batch_id, user_id),
    )
    if not batch:
        raise NotFoundError("KFH import batch", batch_id)
    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "fetched_count": int(batch["fetched_count"]),
        "unsettled_count": int(batch["unsettled_count"] or 0),
        "mode": batch["mode"],
        "requested_from": batch["requested_from"],
        "requested_to": batch["requested_to"],
        "counts": json.loads(batch["counts_json"]),
        "items": _batch_items(batch_id),
        "reconciliation": (
            json.loads(batch["reconciliation_json"])
            if batch["reconciliation_json"]
            else None
        ),
        "cash_summary": (
            json.loads(batch["cash_summary_json"])
            if batch["cash_summary_json"]
            else None
        ),
        "financial_data_changed": batch["status"] == "COMMITTED",
    }


def get_connection(user_id: int) -> dict[str, Any] | None:
    row = query_one(
        """SELECT id, broker, account_label, auth_mode, status, created_at,
                  last_connected_at, last_successful_sync, last_sync_status
           FROM broker_connections
           WHERE user_id = ? AND broker = 'KFH'
           ORDER BY COALESCE(updated_at, created_at) DESC, id DESC LIMIT 1""",
        (user_id,),
    )
    return _row_dict(row) if row else None


def disconnect_connection(user_id: int, connection_id: int) -> dict[str, Any]:
    existing = query_one(
        """SELECT id FROM broker_connections
           WHERE id = ? AND user_id = ? AND broker = 'KFH'""",
        (connection_id, user_id),
    )
    if not existing:
        raise NotFoundError("KFH broker connection", connection_id)
    now = int(time.time())
    exec_sql(
        """UPDATE broker_connections
           SET status = 'DISCONNECTED', last_sync_status = 'CANCELLED', updated_at = ?
           WHERE id = ? AND user_id = ?""",
        (now, connection_id, user_id),
    )
    return get_connection(user_id)


def _canonical_from_raw(row: Any) -> dict[str, Any]:
    return json.loads(row["canonical_payload"])


def _transaction_payload(canonical: dict[str, Any], reference: str, now: int) -> dict[str, Any]:
    transaction_type = canonical["transaction_type"]
    amount = Decimal(canonical["amount"] or "0")
    zero = Decimal("0")
    payload = {
        "portfolio": PORTFOLIO,
        "stock_symbol": canonical["symbol"] or "DIVIDEND",
        "txn_date": canonical["transaction_date"],
        "txn_type": {
            "BUY": "Buy",
            "SELL": "Sell",
            "CASH_DIVIDEND": "DIVIDEND_ONLY",
        }[transaction_type],
        "shares": Decimal(canonical["quantity"] or "0"),
        "purchase_cost": amount if transaction_type == "BUY" else zero,
        "sell_value": amount if transaction_type == "SELL" else zero,
        "bonus_shares": zero,
        "cash_dividend": amount if transaction_type == "CASH_DIVIDEND" else zero,
        "reinvested_dividend": zero,
        "fees": Decimal(canonical["fees"] or "0"),
        "price_override": Decimal(canonical["price"]) if canonical["price"] is not None else None,
        "planned_cum_shares": None,
        "broker": "KFH Trade",
        "reference": reference,
        "notes": (
            f"[KFH Trade sync] KFH ref: {reference} — {canonical['description']}"
            if canonical.get("description")
            else f"[KFH Trade sync] KFH ref: {reference}"
        ),
        "created_at": now,
    }
    return payload


def _insert_transaction(user_id: int, payload: dict[str, Any]) -> int:
    # The legacy transactions table is still float-backed in the application
    # layer.  Keep all KFH arithmetic as Decimal and cross that boundary only
    # when binding the final, confirmed row.
    def legacy_number(value: Decimal | None) -> float | None:
        return _to_legacy_portfolio_number(value) if value is not None else None

    transaction_id = exec_sql_returning_id(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, bonus_shares, cash_dividend,
            reinvested_dividend, fees, price_override, planned_cum_shares,
            broker, reference, notes, category, fx_rate_at_txn, source,
            is_deleted, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   'portfolio', 1.0, 'KFH_LIVE_SYNC', 0, ?)""",
        (
            user_id,
            payload["portfolio"],
            payload["stock_symbol"],
            payload["txn_date"],
            payload["txn_type"],
            legacy_number(payload["shares"]),
            legacy_number(payload["purchase_cost"]),
            legacy_number(payload["sell_value"]),
            legacy_number(payload["bonus_shares"]),
            legacy_number(payload["cash_dividend"]),
            legacy_number(payload["reinvested_dividend"]),
            legacy_number(payload["fees"]),
            legacy_number(payload["price_override"]),
            payload["planned_cum_shares"],
            payload["broker"],
            payload["reference"],
            payload["notes"],
            payload["created_at"],
        ),
    )

    if payload["txn_type"] in {"Buy", "Sell"}:
        symbol = payload["stock_symbol"].strip().upper()
        stock = query_one(
            "SELECT id FROM stocks WHERE user_id = ? AND UPPER(TRIM(symbol)) = ?",
            (user_id, symbol),
        )
        if not stock:
            from app.services.price_service import _yahoo_symbol

            exec_sql(
                """INSERT INTO stocks
                   (user_id, symbol, name, portfolio, currency, current_price,
                    yf_ticker, price_source, created_at)
                   VALUES (?, ?, ?, 'KFH', 'KWD', 0.0, ?, 'AUTO', ?)""",
                (user_id, symbol, symbol, _yahoo_symbol(symbol, "KWD"), payload["created_at"]),
            )
    return transaction_id


def _insert_deposit(user_id: int, canonical: dict[str, Any], reference: str, now: int) -> int:
    source = canonical["transaction_type"].lower()
    return exec_sql_returning_id(
        """INSERT INTO cash_deposits
           (user_id, portfolio, deposit_date, amount, currency, bank_name,
            source, notes, description, comments, include_in_analysis,
            fx_rate_at_deposit, is_deleted, created_at)
           VALUES (?, 'KFH', ?, ?, 'KWD', 'KFH', ?, ?, ?, NULL, 1, 1.0, 0, ?)""",
        (
            user_id,
            canonical["transaction_date"],
            _to_legacy_portfolio_number(Decimal(canonical["amount"])),
            source,
            f"[KFH Trade sync] KFH ref: {reference}",
            canonical.get("description"),
            now,
        ),
    )


def confirm_batch(
    user_id: int,
    batch_id: int,
    selected_item_ids: list[int],
    *,
    update_cash_balance: bool = False,
) -> dict[str, Any]:
    now = int(time.time())
    _log_kfh_event("KFH_SYNC_CONFIRMED", batch_id=batch_id, selected_count=len(selected_item_ids))
    with transaction() as conn:
        batch = query_one(
            "SELECT * FROM broker_import_batches WHERE id = ? AND user_id = ?",
            (batch_id, user_id),
        )
        if not batch:
            raise NotFoundError("KFH import batch", batch_id)
        if batch["status"] == "COMMITTED":
            return json.loads(batch["commit_result_json"])
        if batch["status"] == "RECONCILIATION_FAILED":
            raise BadRequestError(
                "SYNC_RECONCILIATION_FAILED: KFH totals and parsed transaction totals "
                "do not match. Review required."
            )
        if batch["status"] != "PREVIEW_READY":
            raise BadRequestError("KFH batch is not awaiting confirmation")
        cash_summary = (
            json.loads(batch["cash_summary_json"])
            if batch["cash_summary_json"]
            else None
        )
        if update_cash_balance and cash_summary is None:
            raise BadRequestError("KFH did not report a closing cash balance for this batch")

        selected_ids = {int(item_id) for item_id in selected_item_ids}
        items = query_all(
            """SELECT i.*, r.*, i.id AS import_item_id, r.id AS raw_id
               FROM broker_import_items i
               JOIN broker_raw_transactions r ON r.id = i.broker_raw_transaction_id
               WHERE i.batch_id = ?""",
            (batch_id,),
        )
        item_ids = {int(item["import_item_id"]) for item in items}
        if not selected_ids.issubset(item_ids):
            raise BadRequestError("Selected KFH item does not belong to this batch")
        invalid = [
            item
            for item in items
            if int(item["import_item_id"]) in selected_ids and item["classification"] != "NEW"
        ]
        if invalid:
            raise BadRequestError("Only NEW KFH transactions can be selected for confirmation")

        connection = query_one(
            "SELECT broker_account_key FROM broker_connections WHERE id = ? AND user_id = ?",
            (batch["broker_connection_id"], user_id),
        )
        if not connection:
            raise BadRequestError("KFH broker connection no longer exists")
        account_key = connection["broker_account_key"]

        pending_inserts: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        raced_already = 0
        for item in items:
            if int(item["import_item_id"]) not in selected_ids:
                continue
            raw_id = int(item["raw_id"])
            canonical = _canonical_from_raw(item)
            current = query_one(
                "SELECT committed_at FROM broker_raw_transactions WHERE id = ? AND user_id = ?",
                (raw_id, user_id),
            )
            link = query_one(
                "SELECT id FROM broker_transaction_links WHERE broker_raw_transaction_id = ?",
                (raw_id,),
            )
            if (current and current["committed_at"] is not None) or link:
                raced_already += 1
                continue
            match = _find_existing_match(
                user_id,
                raw_id,
                item["broker_transaction_ref"],
                account_key,
                canonical,
            )
            if match and match["classification"] == "CONFLICT":
                raise BadRequestError(
                    f"KFH transaction {item['broker_transaction_ref']} changed after preview; review required"
                )
            if match and match["classification"] == "MATCHED_EXISTING":
                _insert_link(
                    user_id,
                    raw_id,
                    match["kind"],
                    match["record_id"],
                    match["match_type"],
                    match["confidence"],
                    now,
                )
                raced_already += 1
                continue
            pending_inserts.append((item, canonical, _transaction_payload(canonical, item["broker_transaction_ref"], now) if canonical["record_kind"] == "transaction" else {}))

        transaction_payloads = [payload for _, canonical, payload in pending_inserts if canonical["record_kind"] == "transaction"]
        if transaction_payloads:
            from app.api.v1.portfolio import validate_position_mutation

            validate_position_mutation(user_id, additions=transaction_payloads)

        cash_delta = Decimal(0)
        added = 0
        for item, canonical, payload in pending_inserts:
            raw_id = int(item["raw_id"])
            if canonical["record_kind"] == "transaction":
                record_id = _insert_transaction(user_id, payload)
                amount = _money(canonical["amount"])
                fees = _money(canonical["fees"])
                if canonical["transaction_type"] == "BUY":
                    cash_delta -= amount + fees
                elif canonical["transaction_type"] == "SELL":
                    cash_delta += amount - fees
                else:
                    cash_delta += amount - fees
                record_kind = "transaction"
            else:
                record_id = _insert_deposit(user_id, canonical, item["broker_transaction_ref"], now)
                amount = _money(canonical["amount"])
                cash_delta += -amount if canonical["transaction_type"] == "WITHDRAWAL" else amount
                record_kind = "deposit"
            _insert_link(
                user_id,
                raw_id,
                record_kind,
                record_id,
                "BROKER_REFERENCE_EXACT",
                1.0,
                now,
            )
            exec_sql(
                "UPDATE broker_raw_transactions SET committed_at = ?, updated_at = ? WHERE id = ?",
                (now, now, raw_id),
            )
            added += 1

        if added:
            PortfolioService(user_id).recalc_portfolio_cash(
                deposit_delta=_to_legacy_portfolio_number(cash_delta),
                delta_portfolio=PORTFOLIO,
                conn=conn,
            )

        if update_cash_balance:
            reported_cash = _money(cash_summary["reported_closing_cash"])
            existing_cash = query_one(
                "SELECT id FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
                (user_id, PORTFOLIO),
            )
            if existing_cash:
                exec_sql(
                    """UPDATE portfolio_cash
                       SET balance = ?, currency = 'KWD', manual_override = 1, last_updated = ?
                       WHERE user_id = ? AND portfolio = ?""",
                    (reported_cash, now, user_id, PORTFOLIO),
                )
            else:
                exec_sql(
                    """INSERT INTO portfolio_cash
                       (user_id, portfolio, balance, currency, last_updated, manual_override)
                       VALUES (?, ?, ?, 'KWD', ?, 1)""",
                    (user_id, PORTFOLIO, reported_cash, now),
                )

        current_cash_row = query_one(
            "SELECT balance FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
            (user_id, PORTFOLIO),
        )
        current_cash = _money(current_cash_row["balance"] if current_cash_row else 0)

        counts = json.loads(batch["counts_json"])
        result = {
            "batch_id": batch_id,
            "status": "COMMITTED",
            "added": added,
            "already_existed": counts["ALREADY_IN_SAHAM"] + raced_already,
            "requires_review": counts["NEEDS_REVIEW"],
            "conflicts": counts["CONFLICT"],
            "unsupported": counts["UNSUPPORTED"],
            "duplicates_added": 0,
            "existing_transactions_modified": 0,
            "cash_balance_updated": update_cash_balance,
            "current_saham_cash": _money_text(current_cash),
            "reported_closing_cash": (
                cash_summary["reported_closing_cash"] if cash_summary else None
            ),
            "message": (
                f"{added} transactions added; "
                f"{counts['ALREADY_IN_SAHAM'] + raced_already} already existed; "
                f"{counts['NEEDS_REVIEW']} requires review"
                + ("; Saham cash balance updated" if update_cash_balance else "")
            ),
        }
        exec_sql(
            """UPDATE broker_import_batches
               SET status = 'COMMITTED', confirmed_at = ?, committed_at = ?,
                   commit_result_json = ?
               WHERE id = ? AND user_id = ?""",
            (now, now, safe_json_dumps(result), batch_id, user_id),
        )
        exec_sql(
            """UPDATE broker_connections
               SET status = 'READY', last_successful_sync = ?,
                   last_sync_status = 'COMMITTED', updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), now, batch["broker_connection_id"], user_id),
        )
    _log_kfh_event("KFH_SYNC_COMMITTED", batch_id=batch_id, added_count=result["added"])
    return result
