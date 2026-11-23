"""Portfolio input and arithmetic contracts, independent of prompt wording."""

from decimal import Decimal
from hashlib import sha256

import pytest
from pydantic import ValidationError

from tradingagents.portfolio import PortfolioContext
from tradingagents.portfolio_sizing import calculate_sizing


@pytest.fixture
def inputs():
    return {
        "cash": 9000,
        "currency": "USD",
        "open_orders": [],
        "positions": [
            {
                "ticker": "AAPL",
                "quantity": 10,
                "mark_price": 100,
                "target_weight": 0.5,
                "max_weight": 0.6,
                "risk_budget": 300,
                "stop_price": 90,
                "horizon_days": 30,
                "leverage": 1,
            }
        ],
    }


@pytest.mark.unit
def test_exact_sizing_uses_mark_not_average_entry(inputs):
    inputs["positions"][0]["average_price"] = 500
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "aapl")
    assert (
        result.current_exposure,
        result.current_weight,
        result.target_gap,
        result.stop_quantity,
        result.buy_capacity,
        result.additional_quantity,
    ) == (Decimal(1000), Decimal(".1"), Decimal(4000), Decimal(30), Decimal(50), Decimal(20))
    assert result.unknown_reasons == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    "patch, capacity, additional",
    [
        ({"limits": {"max_weight": 0.2}}, 10, 10),
        ({"limits": {"cash_reserve": 8500}}, 5, 5),
        ({"cash": 500, "portfolio_value": 10000}, 5, 5),
        ({"cash": 500, "portfolio_value": 10000, "limits": {"max_leverage": 2}}, 5, 5),
        (
            {"open_orders": [{"ticker": "AAPL", "side": "buy", "quantity": 7, "limit_price": 100}]},
            43,
            13,
        ),
        (
            {
                "open_orders": [
                    {"ticker": "AAPL", "side": "sell", "quantity": 7, "limit_price": 100}
                ]
            },
            50,
            20,
        ),
        (
            {
                "open_orders": [
                    {"ticker": "MSFT", "side": "buy", "quantity": 88, "limit_price": 100}
                ]
            },
            2,
            2,
        ),
    ],
)
def test_capacity_respects_limits_and_pending_commitments(inputs, patch, capacity, additional):
    inputs.update(patch)
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert (result.buy_capacity, result.additional_quantity) == (
        Decimal(capacity),
        Decimal(additional),
    )


@pytest.mark.unit
def test_leverage_requires_both_instrument_and_portfolio_permission(inputs):
    inputs.update(cash=500, portfolio_value=10000, limits={"max_leverage": 3})
    inputs["positions"][0]["leverage"] = 2
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.buy_capacity == Decimal(10)


@pytest.mark.unit
def test_pending_sells_cannot_fund_buys_or_net_gross_capacity(inputs):
    inputs["open_orders"] = [
        {"ticker": "MSFT", "side": "sell", "quantity": 89, "limit_price": 100},
        {"ticker": "AAPL", "side": "buy", "quantity": 2, "limit_price": 100},
    ]
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.buy_capacity == 0
    assert result.additional_quantity == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, value",
    [
        ("mark_price", 0),
        ("mark_price", float("inf")),
        ("target_weight", 1.1),
        ("max_weight", -0.1),
        ("risk_budget", -1),
        ("stop_price", 0),
        ("horizon_days", 1.5),
        ("horizon_days", True),
        ("leverage", 0.5),
    ],
)
def test_invalid_instrument_risk_fields_rejected(inputs, field, value):
    inputs["positions"][0][field] = value
    with pytest.raises(ValidationError):
        PortfolioContext.model_validate(inputs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "patch",
    [
        {"portfolio_value": 0},
        {"limits": {"max_weight": 2}},
        {"limits": {"max_leverage": float("nan")}},
        {"limits": {"cash_reserve": -1}},
        {"open_orders": [{"ticker": "AAPL", "quantity": 1, "side": "hold"}]},
        {"open_orders": [{"ticker": "AAPL", "quantity": -1, "side": "buy"}]},
        {"open_orders": [{"ticker": "AAPL", "quantity": 1, "side": "buy", "limit_price": 0}]},
    ],
)
def test_invalid_portfolio_limits_and_orders_rejected(inputs, patch):
    inputs.update(patch)
    with pytest.raises(ValidationError):
        PortfolioContext.model_validate(inputs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field", ["mark_price", "target_weight", "risk_budget", "stop_price", "max_weight"]
)
def test_incomplete_instrument_inputs_do_not_invent_quantity(inputs, field):
    del inputs["positions"][0][field]
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.additional_quantity is None
    assert result.unknown_reasons


@pytest.mark.unit
@pytest.mark.parametrize(
    "patch",
    [
        {"cash": None},
        {"open_orders": None},
        {"open_orders": [{"ticker": "MSFT", "side": "buy", "quantity": 1}]},
        {"positions": [{"ticker": "AAPL", "quantity": 5, "average_price": 100}]},
    ],
)
def test_incomplete_account_inputs_leave_capacity_unknown(inputs, patch):
    inputs.update(patch)
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.buy_capacity is None
    assert result.additional_quantity is None
    assert result.unknown_reasons


@pytest.mark.unit
@pytest.mark.parametrize("quantity, stop", [(10, 100), (10, 110), (-10, 90)])
def test_invalid_long_stop_or_short_book_never_proposes_addition(inputs, quantity, stop):
    inputs["positions"][0].update(quantity=quantity, stop_price=stop)
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.additional_quantity is None
    assert result.unknown_reasons


@pytest.mark.unit
def test_alias_holdings_and_pending_orders_aggregate(inputs):
    inputs["positions"][0]["ticker"] = "BTCUSDT"
    inputs["positions"].append({**inputs["positions"][0], "ticker": "BTC-USD", "quantity": 5})
    inputs["open_orders"] = [{"ticker": "BTCUSDT", "side": "buy", "quantity": 2}]
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "BTC-USD")
    assert result.current_exposure == Decimal(1500)
    assert result.additional_quantity == Decimal(13)


@pytest.mark.unit
def test_legacy_fingerprint_preserves_checkpoint_bytes():
    raw = '{"cash":null,"currency":null,"positions":[{"ticker":"AAPL","quantity":5.0,"average_price":null}]}'
    assert (
        PortfolioContext.model_validate_json(raw).fingerprint()
        == sha256(raw.encode()).hexdigest()[:12]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, value",
    [
        ("mark_price", 101),
        ("target_weight", 0.4),
        ("max_weight", 0.5),
        ("risk_budget", 200),
        ("stop_price", 80),
        ("horizon_days", 60),
        ("leverage", 2),
    ],
)
def test_risk_input_changes_invalidate_fingerprint(inputs, field, value):
    original = PortfolioContext.model_validate(inputs).fingerprint()
    inputs["positions"][0][field] = value
    assert PortfolioContext.model_validate(inputs).fingerprint() != original


@pytest.mark.unit
@pytest.mark.parametrize(
    "patch",
    [
        {"portfolio_value": 12000},
        {"limits": {"cash_reserve": 100}},
        {"open_orders": [{"ticker": "AAPL", "side": "buy", "quantity": 1}]},
    ],
)
def test_account_input_changes_invalidate_fingerprint(inputs, patch):
    original = PortfolioContext.model_validate(inputs).fingerprint()
    inputs.update(patch)
    assert PortfolioContext.model_validate(inputs).fingerprint() != original


@pytest.mark.unit
def test_boundary_models_are_frozen(inputs):
    book = PortfolioContext.model_validate(inputs)
    with pytest.raises(ValidationError, match="frozen"):
        book.positions[0].mark_price = 99
    with pytest.raises(ValidationError, match="frozen"):
        book.cash = 0


@pytest.mark.unit
@pytest.mark.parametrize("target, budget, expected", [(0.15, 300, 5), (0.5, 50, 0), (0, 300, 0)])
def test_target_and_total_stop_budget_bound_additions(inputs, target, budget, expected):
    inputs["positions"][0].update(target_weight=target, risk_budget=budget)
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.additional_quantity == Decimal(expected)


@pytest.mark.unit
def test_conflicting_alias_marks_leave_exposure_unknown(inputs):
    inputs["positions"].append({**inputs["positions"][0], "mark_price": 200})
    result = calculate_sizing(PortfolioContext.model_validate(inputs), "AAPL")
    assert result.current_exposure is None
    assert result.additional_quantity is None
    assert result.unknown_reasons
