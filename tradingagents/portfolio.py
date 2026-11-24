"""The caller's book, as the decision agents see it.

Optional input to a run: what is held, at what average price, and how much cash
is free. Without it the agents cannot tell adding to a full position from
opening a new one. Three states are distinct and must stay so: a position, a
flat book, and no context at all, since treating "not provided" as "flat" would
invent a fact about the caller's account.

Broker-neutral by construction: quantities are generic units and the currency is
whatever label the caller passes, so nothing here implies a venue or an
execution path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tradingagents.dataflows.symbols import normalize_symbol


class Position(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    ticker: str = Field(description="Instrument symbol, e.g. AAPL")
    quantity: float = Field(description="Signed units held; negative is short")
    average_price: float | None = Field(default=None, description="Average entry price per unit")
    mark_price: float | None = Field(default=None, gt=0)
    target_weight: float | None = Field(
        default=None, ge=0, le=1, description="Fraction of portfolio value"
    )
    max_weight: float | None = Field(default=None, ge=0, le=1)
    risk_budget: float | None = Field(
        default=None, ge=0, description="Total position stop risk in portfolio currency"
    )
    stop_price: float | None = Field(default=None, gt=0)
    horizon_days: int | None = Field(default=None, gt=0, strict=True)
    leverage: float | None = Field(default=None, ge=1)


class PortfolioLimits(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True, extra="forbid")

    max_weight: float | None = Field(default=None, ge=0, le=1)
    max_leverage: float | None = Field(default=None, ge=1)
    cash_reserve: float | None = Field(default=None, ge=0)


class OpenOrder(BaseModel):
    """Remaining unfilled units; pending sells never free capacity before filling."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)


class PortfolioContext(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    cash: float | None = Field(default=None, description="Free cash available")
    currency: str | None = Field(default=None, description="Currency label for cash and prices")
    positions: tuple[Position, ...] = ()
    portfolio_value: float | None = Field(
        default=None,
        gt=0,
        description="Equity in portfolio currency; otherwise derived from cash and marked holdings",
    )
    limits: PortfolioLimits | None = None
    open_orders: tuple[OpenOrder, ...] | None = None

    def positions_in(self, ticker: str) -> list[Position]:
        """The holdings of the instrument the run prices for ``ticker``, under any
        broker spelling (BTCUSDT, 700.HK). Holdings keep the spelling they were
        given: a symbol can also name another instrument (WTI the stock)."""
        symbol = normalize_symbol(ticker)
        return [p for p in self.positions if normalize_symbol(p.ticker) == symbol]

    def position_in(self, ticker: str) -> Position | None:
        held = self.positions_in(ticker)
        return held[0] if held else None

    def render(self, ticker: str, *, trade_date: str | None = None) -> str:
        """The portfolio block for the decision agents, led by the analyzed instrument."""
        if trade_date is not None:
            from tradingagents.portfolio_runtime import render_runtime_portfolio

            return render_runtime_portfolio(self, ticker, trade_date)
        symbol = normalize_symbol(ticker)
        held = self.positions_in(symbol)
        lines = []
        for position in held:
            price = (
                f", average price {position.average_price:,.2f}"
                if position.average_price is not None
                else ""
            )
            spelled = position.ticker.strip().upper()
            listed = f" (listed as {position.ticker})" if spelled != symbol else ""
            lines.append(
                f"- Current position in {symbol}: {position.quantity:,.4g} units{price}{listed}"
            )
        if not held:
            lines.append(f"- No current position in {symbol}")
        if self.cash is not None:
            lines.append(
                f"- Cash available: {self.cash:,.2f}{' ' + self.currency if self.currency else ''}"
            )
        others = [p for p in self.positions if all(p is not h for h in held)]
        if others:
            lines.append(
                "- Other positions: "
                + ", ".join(f"{p.ticker.upper()} {p.quantity:,.4g}" for p in others)
            )
        from tradingagents.portfolio_sizing import calculate_sizing

        lines.extend(calculate_sizing(self, ticker).render())
        for position in held:
            inputs = position.model_dump(
                exclude_none=True, exclude={"ticker", "quantity", "average_price"}
            )
            if inputs:
                lines.append(
                    f"- Sizing inputs (weights are fractions; risk budget in portfolio currency): {json.dumps(inputs)}"
                )
        if self.limits is not None:
            lines.append(f"- Portfolio limits: {self.limits.model_dump_json(exclude_none=True)}")
        if self.open_orders:
            lines.append(
                "- Pending orders reserve capacity; sells do not release cash or exposure until filled."
            )
        return "Portfolio at the analysis date:\n" + "\n".join(lines)

    def fingerprint(self) -> str:
        """Stable digest of the book, so a changed one cannot resume a stale run."""
        # Keep the original JSON bytes when new fields are omitted, including
        # legacy null cash/currency/average_price values.
        new_fields = Position.model_fields.keys() - {"ticker", "quantity", "average_price"}
        excluded = {
            "positions": {
                i: {name for name in new_fields if getattr(p, name) is None}
                for i, p in enumerate(self.positions)
            },
            **{
                name: True
                for name in ("portfolio_value", "limits", "open_orders")
                if getattr(self, name) is None
            },
        }
        return hashlib.sha256(self.model_dump_json(exclude=excluded).encode()).hexdigest()[:12]


def load_portfolio(path: str | Path) -> PortfolioContext:
    """Read a portfolio JSON file, failing before the run rather than mid-graph."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PortfolioContext.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"portfolio file {path} is not usable: {exc}") from exc
