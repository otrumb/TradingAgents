"""Deterministic advisory spot actions, separate from the market rating."""

from decimal import Decimal
from enum import StrEnum
from typing import Literal, TypeAlias, assert_never

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.dataflows.crypto_models import FrozenModel
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.portfolio import PortfolioContext
from tradingagents.portfolio_sizing import calculate_sizing

MarketBias: TypeAlias = PortfolioRating | Literal["REVIEW"]


class SpotAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO TRADE"


class SpotAllocation(FrozenModel):
    """Known holdings and existing sizing evidence; None means unavailable."""

    ticker: str
    held_quantity: Decimal | None
    current_exposure: Decimal | None = None
    current_weight: Decimal | None = None
    target_gap: Decimal | None = None
    stop_quantity: Decimal | None = None
    buy_capacity: Decimal | None = None
    additional_quantity: Decimal | None = None
    unknown_reasons: tuple[str, ...] = ()


class SpotDecision(FrozenModel):
    allocation: SpotAllocation
    market_bias: MarketBias
    action: SpotAction
    quantity: Decimal


def calculate_spot_allocation(book: PortfolioContext | None, ticker: str) -> SpotAllocation:
    """Aggregate instrument aliases without treating missing context as flat."""
    symbol = normalize_symbol(ticker)
    if book is None:
        return SpotAllocation(ticker=symbol, held_quantity=None)
    sizing = calculate_sizing(book, symbol)
    return SpotAllocation(
        ticker=symbol,
        held_quantity=sum(
            (Decimal(str(p.quantity)) for p in book.positions_in(symbol)), Decimal(0)
        ),
        current_exposure=sizing.current_exposure,
        current_weight=sizing.current_weight,
        target_gap=sizing.target_gap,
        stop_quantity=sizing.stop_quantity,
        buy_capacity=sizing.buy_capacity,
        additional_quantity=sizing.additional_quantity,
        unknown_reasons=sizing.unknown_reasons,
    )


def calculate_spot_decision(
    book: PortfolioContext | None, ticker: str, market_bias: MarketBias
) -> SpotDecision:
    """Convert typed bias to a funded addition, long reduction, or no trade."""
    return derive_spot_action(calculate_spot_allocation(book, ticker), market_bias)


def derive_spot_action(allocation: SpotAllocation, market_bias: MarketBias) -> SpotDecision:
    """Derive action from run-start allocation, independent of LLM prose or futures."""
    held = allocation.held_quantity
    action = SpotAction.NO_TRADE
    quantity = Decimal(0)
    if held is not None and held >= 0:
        match market_bias:
            case PortfolioRating.SELL:
                if held > 0:
                    action, quantity = SpotAction.SELL, held
            case PortfolioRating.UNDERWEIGHT:
                exposure, gap = allocation.current_exposure, allocation.target_gap
                if held > 0 and exposure is not None and exposure > 0 and gap is not None:
                    excess = min(held, max(Decimal(0), -gap * held / exposure))
                    if excess > 0:
                        action, quantity = SpotAction.SELL, excess
            case PortfolioRating.BUY | PortfolioRating.OVERWEIGHT:
                additional = allocation.additional_quantity
                if additional is not None and additional > 0:
                    action, quantity = SpotAction.BUY, additional
            case PortfolioRating.HOLD | "REVIEW":
                pass
            case unreachable:
                assert_never(unreachable)
    return SpotDecision(
        allocation=allocation, market_bias=market_bias, action=action, quantity=quantity
    )


def render_spot_decision(decision: SpotDecision) -> str:
    """Render compact Markdown from structured fields only."""
    held = decision.allocation.held_quantity
    held_text = "unknown" if held is None else format(held, "f")
    return (
        f"### Spot decision: {decision.allocation.ticker}\n"
        f"- Market bias: {decision.market_bias}\n"
        f"- Held quantity: {held_text}\n"
        f"- Spot action: {decision.action}\n"
        f"- Quantity: {decision.quantity:f}"
    )
