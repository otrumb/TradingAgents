"""Immutable, current-run portfolio valuation from public Spot evidence."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Final

from tradingagents.dataflows.crypto_models import Asset, FrozenModel, SourceFailure
from tradingagents.dataflows.crypto_spot_models import IntradayPrice, SpotPriceContext
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.portfolio import PortfolioContext

SUPPORTED: Final[tuple[Asset, ...]] = ("BTC", "ETH", "SOL")


class RuntimeValuation(FrozenModel):
    portfolio: PortfolioContext
    marks: tuple[IntradayPrice, ...] = ()
    failures: tuple[SourceFailure, ...] = ()
    value_source: str


def enrich_portfolio(
    book: PortfolioContext, evidence: tuple[SpotPriceContext, ...]
) -> RuntimeValuation:
    """Apply trusted intraday marks only; never substitute entry or daily prices."""
    prices = {str(item.asset): item.intraday for item in evidence if item.intraday is not None}
    positions = []
    used: set[Asset] = set()
    failures = []
    for position in book.positions:
        if position.mark_price is not None or position.quantity == 0:
            positions.append(position)
            continue
        asset = crypto_base(position.ticker)
        price = prices.get(asset) if asset is not None else None
        if price is not None and book.currency in ("USD", "USDT"):
            positions.append(position.model_copy(update={"mark_price": float(price.price)}))
            used.add(price.asset)
        else:
            positions.append(position)
            failure = next(
                (item.intraday_failure for item in evidence if item.asset == asset), None
            )
            failures.append(
                SourceFailure(
                    kind="unavailable",
                    detail=f"{position.ticker}: "
                    + (
                        failure.model_dump_json()
                        if failure is not None
                        else "no eligible current Spot mark in portfolio currency"
                    ),
                )
            )
    value = book.portfolio_value
    source = "caller" if value is not None else "unavailable"
    if (
        value is None
        and book.cash is not None
        and all(p.quantity == 0 or p.mark_price is not None for p in positions)
    ):
        total = Decimal(str(book.cash)) + sum(
            (
                Decimal(str(p.quantity)) * Decimal(str(p.mark_price))
                for p in positions
                if p.mark_price is not None
            ),
            Decimal(0),
        )
        if total > 0:
            value = float(total)
            source = "cash plus signed marked holdings; pending orders excluded"
    return RuntimeValuation(
        portfolio=book.model_copy(update={"positions": tuple(positions), "portfolio_value": value}),
        marks=tuple(prices[asset] for asset in SUPPORTED if asset in used),
        failures=tuple(failures),
        value_source=source,
    )


def render_runtime_portfolio(
    book: PortfolioContext,
    ticker: str,
    trade_date: str,
    *,
    evidence: tuple[SpotPriceContext, ...] = (),
) -> str:
    """Render and size from supplied current-run evidence without collecting."""
    current = date.fromisoformat(trade_date) == datetime.now(UTC).date()
    result = enrich_portfolio(book, evidence if current else ())
    return render_runtime_valuation(result, ticker, current)


def render_runtime_valuation(result: RuntimeValuation, ticker: str, current: bool) -> str:
    note = result.model_dump_json(exclude={"portfolio"})
    gating = "current UTC date" if current else "current Spot fetch disabled for non-current date"
    quote = "USDT Spot quote; USD uses 1:1 USDT/USD assumption; no FX conversion"
    value = result.portfolio.portfolio_value
    value_line = f"\n- Portfolio value: {value:,.8g}" if value is not None else ""
    return (
        result.portfolio.render(ticker)
        + value_line
        + f"\nRuntime valuation ({gating}; {quote}): {note}"
    )
