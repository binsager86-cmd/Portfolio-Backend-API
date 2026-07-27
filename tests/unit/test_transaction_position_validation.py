import pytest

from app.api.v1.portfolio import validate_position_mutation
from app.core.exceptions import BadRequestError
from tests.helpers import create_transaction, get_test_db


pytestmark = pytest.mark.usefixtures("_init_test_db")


def _clear_symbol(symbol: str) -> None:
    conn = get_test_db()
    try:
        conn.execute(
            "DELETE FROM transactions WHERE user_id = 1 AND UPPER(TRIM(stock_symbol)) = ?",
            (symbol.upper(),),
        )
        conn.commit()
    finally:
        conn.close()


def _proposed_sell(symbol: str, shares: float) -> dict:
    return {
        "portfolio": "KFH",
        "stock_symbol": symbol,
        "txn_date": "2024-03-01",
        "txn_type": "Sell",
        "shares": shares,
        "sell_value": shares * 0.5,
    }


def test_sell_allows_existing_buy_with_legacy_casing() -> None:
    symbol = "CASESELL"
    _clear_symbol(symbol)
    create_transaction(
        stock_symbol=symbol,
        txn_type=" buy ",
        shares=100,
        purchase_cost=1000,
    )

    validate_position_mutation(1, additions=[_proposed_sell(symbol, 100)])


def test_sell_allows_existing_bonus_shares_label() -> None:
    symbol = "BONUSSELL"
    _clear_symbol(symbol)
    create_transaction(
        stock_symbol=symbol,
        txn_type="Buy",
        shares=50,
        purchase_cost=500,
    )
    create_transaction(
        stock_symbol=symbol,
        txn_date="2024-02-01",
        txn_type="Bonus Shares",
        shares=25,
        purchase_cost=0,
    )

    validate_position_mutation(1, additions=[_proposed_sell(symbol, 75)])


def test_sell_allows_canonical_dividend_only_bonus_shares() -> None:
    symbol = "DIVBONUSSELL"
    _clear_symbol(symbol)
    create_transaction(
        stock_symbol=symbol,
        txn_type="Buy",
        shares=50,
        purchase_cost=500,
    )
    create_transaction(
        stock_symbol=symbol,
        txn_date="2024-02-01",
        txn_type="DIVIDEND_ONLY",
        shares=0,
        purchase_cost=0,
        bonus_shares=25,
    )

    validate_position_mutation(1, additions=[_proposed_sell(symbol, 75)])


def test_cash_dividend_only_transaction_does_not_require_shares() -> None:
    symbol = "CASHDIVONLY"
    _clear_symbol(symbol)

    validate_position_mutation(
        1,
        additions=[
            {
                "portfolio": "KFH",
                "stock_symbol": symbol,
                "txn_date": "2024-03-01",
                "txn_type": "DIVIDEND_ONLY",
                "shares": 0,
                "cash_dividend": 10,
            }
        ],
    )


def test_position_validation_still_blocks_real_oversell() -> None:
    symbol = "TRUEOVERSELL"
    _clear_symbol(symbol)
    create_transaction(
        stock_symbol=symbol,
        txn_type="Buy",
        shares=50,
        purchase_cost=500,
    )

    with pytest.raises(BadRequestError, match="Transaction would oversell TRUEOVERSELL in KFH"):
        validate_position_mutation(1, additions=[_proposed_sell(symbol, 51)])