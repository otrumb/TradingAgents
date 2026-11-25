from decimal import Decimal

import pytest

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.portfolio import PortfolioContext, Position
from tradingagents.spot_decision import (
    MarketBias,
    SpotAction,
    calculate_spot_allocation,
    calculate_spot_decision,
    render_spot_decision,
)


@pytest.mark.parametrize("bias", [PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT])
@pytest.mark.parametrize("positions", [(), (Position(ticker="BTCUSDT", quantity=0),)])
def test_flat_or_absent_btc_never_sells(bias: MarketBias, positions: tuple[Position, ...]) -> None:
    book = PortfolioContext(positions=positions)

    result = calculate_spot_decision(book, "BTC-USD", bias)

    assert (result.action, result.quantity, result.market_bias) == (
        SpotAction.NO_TRADE,
        Decimal(0),
        bias,
    )


def test_sell_exits_all_held_eth_aliases_without_sizing_inputs() -> None:
    book = PortfolioContext(
        positions=(
            Position(ticker="ETHUSDT", quantity=0.1),
            Position(ticker="ETH-USD", quantity=0.2),
            Position(ticker="BTC-USD", quantity=4),
        )
    )
    before = book.model_dump_json()

    result = calculate_spot_decision(book, "ETH-USD", PortfolioRating.SELL)

    assert (result.action, result.quantity) == (SpotAction.SELL, Decimal("0.3"))
    assert book.model_dump_json() == before


@pytest.mark.parametrize("target, expected", [(0.2, "1"), (0.4, "0"), (0.6, "0"), (0, "2")])
def test_underweight_sells_only_excess_over_target(target: float, expected: str) -> None:
    book = PortfolioContext(
        cash=600,
        positions=(Position(ticker="ETHUSDT", quantity=2, mark_price=200, target_weight=target),),
    )

    result = calculate_spot_decision(book, "ETH-USD", PortfolioRating.UNDERWEIGHT)

    assert (result.action, result.quantity) == (
        SpotAction.SELL if Decimal(expected) > 0 else SpotAction.NO_TRADE,
        Decimal(expected),
    )


@pytest.mark.parametrize(
    "position, cash",
    [
        (Position(ticker="ETH-USD", quantity=2, mark_price=200), 600),
        (Position(ticker="ETH-USD", quantity=2, target_weight=0.2), 600),
        (Position(ticker="ETH-USD", quantity=2, mark_price=200, target_weight=0.2), None),
    ],
)
def test_underweight_needs_target_mark_and_equity(position: Position, cash: float | None) -> None:
    book = PortfolioContext(cash=cash, positions=(position,))

    result = calculate_spot_decision(book, "ETH-USD", PortfolioRating.UNDERWEIGHT)

    assert (result.action, result.quantity) == (SpotAction.NO_TRADE, Decimal(0))


@pytest.mark.parametrize("bias", [*PortfolioRating, "REVIEW"])
def test_negative_holdings_never_produce_spot_trade(bias: MarketBias) -> None:
    book = PortfolioContext(
        cash=1000,
        positions=(Position(ticker="ETHUSDT", quantity=-1, mark_price=200, target_weight=0),),
    )

    result = calculate_spot_decision(book, "ETH-USD", bias)

    assert (result.action, result.quantity) == (SpotAction.NO_TRADE, Decimal(0))


@pytest.mark.parametrize("bias", [PortfolioRating.BUY, PortfolioRating.OVERWEIGHT])
@pytest.mark.parametrize("cash, orders, expected", [(9000, (), 20), (0, (), 0), (9000, None, 0)])
def test_positive_bias_uses_existing_sizing_quantity(
    bias: MarketBias, cash: float, orders: tuple[()] | None, expected: int
) -> None:
    book = PortfolioContext(
        cash=cash,
        open_orders=orders,
        positions=(
            Position(
                ticker="ETHUSDT",
                quantity=10,
                mark_price=100,
                target_weight=0.5,
                max_weight=0.6,
                risk_budget=300,
                stop_price=90,
            ),
        ),
    )

    result = calculate_spot_decision(book, "ETH-USD", bias)

    assert (result.action, result.quantity) == (
        SpotAction.BUY if expected > 0 else SpotAction.NO_TRADE,
        Decimal(expected),
    )


@pytest.mark.parametrize("bias", [PortfolioRating.HOLD, "REVIEW"])
def test_hold_and_review_leave_held_assets_unchanged(bias: MarketBias) -> None:
    book = PortfolioContext(positions=(Position(ticker="ETH-USD", quantity=2),))

    result = calculate_spot_decision(book, "ETH-USD", bias)

    assert (result.action, result.quantity, result.market_bias) == (
        SpotAction.NO_TRADE,
        Decimal(0),
        bias,
    )


@pytest.mark.parametrize("bias", [*PortfolioRating, "REVIEW"])
def test_missing_portfolio_keeps_holdings_unknown_and_never_trades(bias: MarketBias) -> None:
    result = calculate_spot_decision(None, "BTCUSDT", bias)

    assert (result.action, result.quantity) == (SpotAction.NO_TRADE, Decimal(0))
    assert result.allocation.held_quantity is None
    assert result.allocation.additional_quantity is None


def test_allocation_normalizes_alias_and_preserves_flat_context() -> None:
    result = calculate_spot_allocation(PortfolioContext(), "BTCUSDT")

    assert (result.ticker, result.held_quantity) == ("BTC-USD", Decimal(0))


def test_markdown_renders_structured_bias_separately_from_spot_action() -> None:
    decision = calculate_spot_decision(None, "BTCUSDT", PortfolioRating.SELL)

    result = render_spot_decision(decision)

    assert result.splitlines() == [
        "### Spot decision: BTC-USD",
        "- Market bias: Sell",
        "- Held quantity: unknown",
        "- Spot action: NO TRADE",
        "- Quantity: 0",
    ]
