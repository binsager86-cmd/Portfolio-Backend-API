"""Durable broker-record identity, staging, and linkage models."""

from decimal import Decimal

from sqlalchemy import Float, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BrokerConnection(Base):
    __tablename__ = "broker_connections"
    __table_args__ = (
        UniqueConstraint("user_id", "broker", "broker_account_key", name="uq_broker_connection_account"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    broker: Mapped[str] = mapped_column(String(50), nullable=False)
    broker_account_key: Mapped[str] = mapped_column(String(64), nullable=False)
    account_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    auth_mode: Mapped[str] = mapped_column(
        String(40), nullable=False, default="LOCAL_BROWSER_SESSION"
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="DISCONNECTED")
    last_connected_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_successful_sync: Mapped[str | None] = mapped_column(String(30), nullable=True)
    last_sync_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BrokerRawTransaction(Base):
    __tablename__ = "broker_raw_transactions"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "broker_connection_id",
            "broker_transaction_ref",
            name="uq_broker_record_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    broker_connection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_connections.id"), nullable=False
    )
    sync_batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    broker_transaction_ref: Mapped[str] = mapped_column(String(200), nullable=False)
    secondary_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    record_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    transaction_date: Mapped[str] = mapped_column(String(10), nullable=False)
    transaction_timestamp: Mapped[str | None] = mapped_column(String(30), nullable=True)
    settlement_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    transaction_type: Mapped[str] = mapped_column(String(30), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(50), nullable=True)
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    amount: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    fees: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    canonical_payload: Mapped[str] = mapped_column(Text, nullable=False)
    raw_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_transaction_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_date: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(100), nullable=False)
    committed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BrokerImportBatch(Base):
    __tablename__ = "broker_import_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    broker_connection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_connections.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    mode: Mapped[str | None] = mapped_column(String(30), nullable=True)
    requested_from: Mapped[str | None] = mapped_column(String(10), nullable=True)
    requested_to: Mapped[str | None] = mapped_column(String(10), nullable=True)
    started_at: Mapped[int] = mapped_column(Integer, nullable=False)
    fetched_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    committed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scope_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unsettled_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    new_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conflict_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unsupported_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kfh_open_balance: Mapped[Decimal | None] = mapped_column(Numeric(38, 3), nullable=True)
    kfh_close_balance: Mapped[Decimal | None] = mapped_column(Numeric(38, 3), nullable=True)
    kfh_total_buy: Mapped[Decimal | None] = mapped_column(Numeric(38, 3), nullable=True)
    kfh_total_sell: Mapped[Decimal | None] = mapped_column(Numeric(38, 3), nullable=True)
    kfh_total_deposit: Mapped[Decimal | None] = mapped_column(Numeric(38, 3), nullable=True)
    kfh_total_withdrawal: Mapped[Decimal | None] = mapped_column(Numeric(38, 3), nullable=True)
    counts_json: Mapped[str] = mapped_column(Text, nullable=False)
    reconciliation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    cash_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    commit_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    confirmed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BrokerImportItem(Base):
    __tablename__ = "broker_import_items"
    __table_args__ = (
        UniqueConstraint("batch_id", "broker_raw_transaction_id", name="uq_broker_batch_record"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_import_batches.id"), nullable=False
    )
    broker_raw_transaction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_raw_transactions.id"), nullable=False
    )
    classification: Mapped[str] = mapped_column(String(30), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_record_kind: Mapped[str | None] = mapped_column(String(20), nullable=True)
    matched_record_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    match_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    saham_value_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    kfh_value_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    selected_default: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class BrokerTransactionLink(Base):
    __tablename__ = "broker_transaction_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    broker_raw_transaction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_raw_transactions.id"), nullable=False, unique=True
    )
    saham_record_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    saham_record_id: Mapped[int] = mapped_column(Integer, nullable=False)
    match_type: Mapped[str] = mapped_column(String(30), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    linked_at: Mapped[int] = mapped_column(Integer, nullable=False)


class BrokerUnsettledTransaction(Base):
    __tablename__ = "broker_unsettled_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    sync_batch_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_import_batches.id"), nullable=False
    )
    broker_connection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("broker_connections.id"), nullable=False
    )
    broker_transaction_ref: Mapped[str | None] = mapped_column(String(200), nullable=True)
    raw_transaction_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_date: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_payload: Mapped[str] = mapped_column(Text, nullable=False)
    raw_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="UNSETTLED")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
