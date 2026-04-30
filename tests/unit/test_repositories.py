"""
Unit tests for BaseRepository, UserRepository, and CashDepositRepository.

Tests run against an in-memory SQLite database to stay fast and isolated.
[B-2] coverage for the repository pattern.
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Import all models so their metadata is registered before create_all
import app.models.user          # noqa: F401
import app.models.portfolio     # noqa: F401
import app.models.cash          # noqa: F401
import app.models.news          # noqa: F401
import app.models.audit         # noqa: F401
import app.models.security      # noqa: F401
import app.models.snapshot      # noqa: F401

from app.core.database import Base
from app.core.repositories.base import BaseRepository
from app.core.repositories.user import UserRepository
from app.core.repositories.cash import CashDepositRepository
from app.models.cash import CashDeposit
from app.models.user import User


# ── In-memory DB fixture ─────────────────────────────────────────────

@pytest.fixture
def db_session():
    """Isolated in-memory SQLite session — torn down after each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_wal(conn, _rec):
        conn.execute("PRAGMA foreign_keys=ON;")

    # Create all tables so eager-loaded relationships don't fail
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


# ── Seed helpers ─────────────────────────────────────────────────────

def _make_user(session, username: str = "testuser") -> User:
    user = User(
        username=username,
        password_hash="$2b$12$fakehash",
        name="Test User",
        is_admin=0,
        created_at=int(time.time()),
    )
    session.add(user)
    session.flush()
    return user


def _make_deposit(session, user_id: int, amount: float = 100.0, is_deleted: int = 0) -> CashDeposit:
    deposit = CashDeposit(
        user_id=user_id,
        portfolio="KFH",
        deposit_date="2025-01-15",
        amount=amount,
        currency="KWD",
        source="deposit",
        is_deleted=is_deleted,
        created_at=int(time.time()),
    )
    session.add(deposit)
    session.flush()
    return deposit


# ── BaseRepository tests ─────────────────────────────────────────────

class TestBaseRepository:
    def test_add_and_get(self, db_session):
        user = _make_user(db_session)
        repo: BaseRepository[User] = BaseRepository(db_session, User)

        fetched = repo.get(user.id)
        assert fetched is not None
        assert fetched.id == user.id
        assert fetched.username == "testuser"

    def test_get_missing_returns_none(self, db_session):
        repo: BaseRepository[User] = BaseRepository(db_session, User)
        assert repo.get(99999) is None

    def test_filter_by(self, db_session):
        _make_user(db_session, "alice")
        _make_user(db_session, "bob")
        repo: BaseRepository[User] = BaseRepository(db_session, User)

        results = repo.filter_by(username="alice")
        assert len(results) == 1
        assert results[0].username == "alice"

    def test_filter_by_no_match(self, db_session):
        _make_user(db_session, "charlie")
        repo: BaseRepository[User] = BaseRepository(db_session, User)
        assert repo.filter_by(username="nobody") == []

    def test_select_with_criteria(self, db_session):
        _make_user(db_session, "dan")
        _make_user(db_session, "eve")
        repo: BaseRepository[User] = BaseRepository(db_session, User)

        results = repo.select(User.username == "dan")
        assert len(results) == 1
        assert results[0].username == "dan"

    def test_bulk_insert(self, db_session):
        user = _make_user(db_session)
        repo: BaseRepository[CashDeposit] = BaseRepository(db_session, CashDeposit)

        repo.bulk_insert([
            {"user_id": user.id, "portfolio": "KFH", "deposit_date": "2025-02-01",
             "amount": 500.0, "currency": "KWD", "is_deleted": 0,
             "created_at": int(time.time())},
            {"user_id": user.id, "portfolio": "KFH", "deposit_date": "2025-03-01",
             "amount": 750.0, "currency": "KWD", "is_deleted": 0,
             "created_at": int(time.time())},
        ])
        db_session.commit()

        all_deposits = repo.all()
        assert len(all_deposits) == 2

    def test_delete(self, db_session):
        user = _make_user(db_session)
        repo: BaseRepository[User] = BaseRepository(db_session, User)

        repo.delete(user)
        db_session.commit()
        assert repo.get(user.id) is None

    def test_refresh(self, db_session):
        user = _make_user(db_session)
        repo: BaseRepository[User] = BaseRepository(db_session, User)

        # Mutate directly, then refresh
        user.name = "Changed"
        refreshed = repo.refresh(user)
        assert refreshed is user  # same object


# ── UserRepository tests ─────────────────────────────────────────────

class TestUserRepository:
    def test_get_by_username_found(self, db_session):
        _make_user(db_session, "frank")
        repo = UserRepository(db_session)

        user = repo.get_by_username("frank")
        assert user is not None
        assert user.username == "frank"

    def test_get_by_username_not_found(self, db_session):
        repo = UserRepository(db_session)
        assert repo.get_by_username("ghost") is None

    def test_get_by_username_case_sensitive(self, db_session):
        _make_user(db_session, "Grace")
        repo = UserRepository(db_session)
        # Exact case must match
        assert repo.get_by_username("grace") is None
        assert repo.get_by_username("Grace") is not None


# ── CashDepositRepository tests ──────────────────────────────────────

class TestCashDepositRepository:
    def test_get_active_found(self, db_session):
        user = _make_user(db_session)
        deposit = _make_deposit(db_session, user.id, amount=200.0, is_deleted=0)
        db_session.commit()

        repo = CashDepositRepository(db_session)
        found = repo.get_active(deposit.id, user.id)
        assert found is not None
        assert found.amount == 200.0

    def test_get_active_soft_deleted_returns_none(self, db_session):
        user = _make_user(db_session)
        deposit = _make_deposit(db_session, user.id, is_deleted=1)
        db_session.commit()

        repo = CashDepositRepository(db_session)
        assert repo.get_active(deposit.id, user.id) is None

    def test_get_active_wrong_user_returns_none(self, db_session):
        user = _make_user(db_session, "henry")
        deposit = _make_deposit(db_session, user.id)
        db_session.commit()

        repo = CashDepositRepository(db_session)
        assert repo.get_active(deposit.id, user_id=99999) is None

    def test_get_deleted_found(self, db_session):
        user = _make_user(db_session)
        deposit = _make_deposit(db_session, user.id, is_deleted=1)
        db_session.commit()

        repo = CashDepositRepository(db_session)
        found = repo.get_deleted(deposit.id, user.id)
        assert found is not None
        assert found.is_deleted == 1

    def test_get_deleted_active_deposit_returns_none(self, db_session):
        user = _make_user(db_session)
        deposit = _make_deposit(db_session, user.id, is_deleted=0)
        db_session.commit()

        repo = CashDepositRepository(db_session)
        assert repo.get_deleted(deposit.id, user.id) is None

    def test_get_any_finds_both_states(self, db_session):
        user = _make_user(db_session)
        active = _make_deposit(db_session, user.id, amount=50.0, is_deleted=0)
        deleted = _make_deposit(db_session, user.id, amount=75.0, is_deleted=1)
        db_session.commit()

        repo = CashDepositRepository(db_session)
        assert repo.get_any(active.id, user.id) is not None
        assert repo.get_any(deleted.id, user.id) is not None

    def test_soft_delete_via_orm(self, db_session):
        user = _make_user(db_session)
        deposit = _make_deposit(db_session, user.id)
        db_session.commit()

        # Simulate the route's soft-delete pattern
        deposit.is_deleted = 1
        deposit.deleted_at = int(time.time())
        db_session.commit()

        repo = CashDepositRepository(db_session)
        assert repo.get_active(deposit.id, user.id) is None
        assert repo.get_deleted(deposit.id, user.id) is not None

    def test_restore_via_orm(self, db_session):
        user = _make_user(db_session)
        deposit = _make_deposit(db_session, user.id, is_deleted=1)
        db_session.commit()

        deposit.is_deleted = 0
        deposit.deleted_at = None
        db_session.commit()

        repo = CashDepositRepository(db_session)
        assert repo.get_active(deposit.id, user.id) is not None
        assert repo.get_deleted(deposit.id, user.id) is None
