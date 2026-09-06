"""Request contracts for the two-phase, read-only KFH synchronization flow."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

KFH_PARSER_VERSION = "kfh-normalizer-v1"
KFH_LIVE_ADAPTER_VERSION = "kfh-cashlog-adapter-v2"


class KfhBrokerRecordInput(BaseModel):
    broker_transaction_ref: str = Field(..., min_length=1, max_length=200)
    transaction_date: str = Field(..., max_length=10)
    transaction_timestamp: str | None = Field(None, max_length=30)
    settlement_date: str | None = Field(None, max_length=10)
    transaction_type: str = Field(..., min_length=1, max_length=50)
    symbol: str | None = Field(None, max_length=50)
    quantity: Decimal | None = None
    price: Decimal | None = None
    amount: Decimal | None = None
    fees: Decimal | None = None
    description: str | None = Field(None, max_length=1000)
    interpretation_status: Literal["ready", "unsupported", "error"] = "ready"
    interpretation_error: str | None = Field(None, max_length=1000)
    raw_payload: dict[str, Any] | None = None
    raw_transaction_type: str | None = Field(None, max_length=100)
    raw_description: str | None = Field(None, max_length=1000)
    raw_date: str | None = Field(None, max_length=100)
    parser_version: str = Field(KFH_PARSER_VERSION, min_length=1, max_length=100)
    adapter_version: str = Field(KFH_LIVE_ADAPTER_VERSION, min_length=1, max_length=100)

    @field_validator("transaction_date", "settlement_date")
    @classmethod
    def validate_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        compact = value.replace("-", "")
        if len(compact) == 8 and compact.isdigit():
            normalized = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        else:
            normalized = value
        date.fromisoformat(normalized)
        return normalized

    @field_validator("broker_transaction_ref", "transaction_type")
    @classmethod
    def trim_required(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed

    @field_validator("transaction_timestamp")
    @classmethod
    def validate_market_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() != timedelta(hours=3):
            raise ValueError("KFH transaction timestamp must use Kuwait offset +03:00")
        if value != parsed.isoformat(timespec="seconds"):
            raise ValueError("KFH transaction timestamp must be an ISO timestamp with seconds")
        return parsed.astimezone(timezone(timedelta(hours=3))).isoformat(timespec="seconds")


class KfhUnsettledRecordInput(BaseModel):
    broker_transaction_ref: str | None = Field(None, max_length=200)
    raw_transaction_type: str | None = Field(None, max_length=100)
    raw_description: str | None = Field(None, max_length=1000)
    raw_date: str | None = Field(None, max_length=100)
    raw_payload: dict[str, Any]
    parser_version: str = Field(KFH_PARSER_VERSION, min_length=1, max_length=100)
    adapter_version: str = Field(KFH_LIVE_ADAPTER_VERSION, min_length=1, max_length=100)


class KfhSyncOptions(BaseModel):
    mode: Literal[
        "SMART_INCREMENTAL",
        "LAST_30_DAYS",
        "LAST_90_DAYS",
        "CUSTOM_RANGE",
        "FULL_RECONCILIATION",
    ]
    # camelCase matches the frontend's KfhSyncOptions wire contract exactly -
    # deliberate, not an oversight.
    fromDate: str | None = None  # noqa: N815
    toDate: str | None = None  # noqa: N815
    mergePolicy: Literal["ADD_MISSING_ONLY"] = "ADD_MISSING_ONLY"  # noqa: N815

    @field_validator("fromDate", "toDate")
    @classmethod
    def validate_scope_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return date.fromisoformat(value).isoformat()


class KfhPreviewRequest(BaseModel):
    broker_account: str = Field(..., min_length=1, max_length=200)
    scope: KfhSyncOptions
    records: list[KfhBrokerRecordInput] = Field(..., max_length=5000)
    unsettled_records: list[KfhUnsettledRecordInput] = Field(default_factory=list, max_length=5000)
    statement_summary: "KfhStatementSummaryInput | None" = None

    @field_validator("broker_account")
    @classmethod
    def trim_account(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("broker_account must not be blank")
        return trimmed


class KfhConfirmRequest(BaseModel):
    selected_item_ids: list[int] = Field(default_factory=list, max_length=5000)
    update_cash_balance: bool = False


class KfhStatementSummaryInput(BaseModel):
    """Accounting evidence returned by KFH, represented as exact decimals."""

    currency: str = Field("KWD", min_length=3, max_length=3)
    open_balance: Decimal
    close_balance: Decimal
    total_deposit: Decimal
    total_withdrawal: Decimal
    total_buy: Decimal
    total_sell: Decimal
    total_other: Decimal
    vat_amount: Decimal

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        currency = value.strip().upper()
        if currency != "KWD":
            raise ValueError("KFH cash-statement reconciliation currently supports KWD only")
        return currency


KfhPreviewRequest.model_rebuild()
