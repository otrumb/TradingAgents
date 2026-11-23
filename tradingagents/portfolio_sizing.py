"""Pure, advisory sizing in portfolio currency. No fills, FX, or contract multipliers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, assert_never

from tradingagents.dataflows.symbols import normalize_symbol

if TYPE_CHECKING:
    from tradingagents.portfolio import PortfolioContext


@dataclass(frozen=True, slots=True)
class SizingGuidance:
    current_exposure: Decimal | None
    current_weight: Decimal | None
    target_gap: Decimal | None
    stop_quantity: Decimal | None
    buy_capacity: Decimal | None
    additional_quantity: Decimal | None
    unknown_reasons: tuple[str, ...]

    def render(self) -> list[str]:
        values = (
            ("Current exposure", self.current_exposure),
            ("Current weight (fraction)", self.current_weight),
            ("Target gap (currency, before pending orders)", self.target_gap),
            ("Stop-distance quantity (total position)", self.stop_quantity),
            ("Additional buy capacity (units)", self.buy_capacity),
            ("Additional long quantity (advisory only)", self.additional_quantity),
        )
        lines = [f"- {label}: {value:.8g}" for label, value in values if value is not None]
        if self.unknown_reasons:
            lines.append("- Sizing unknown: " + "; ".join(self.unknown_reasons))
        return lines


def calculate_sizing(book: PortfolioContext, ticker: str) -> SizingGuidance:
    """Size long additions conservatively; never assume pending sells have filled.

    All marks and limit prices must use portfolio currency and linear unit value.
    Missing open_orders means no pending-order information, not an empty order book.
    Stop budget covers existing units plus pending buys plus proposed additions.
    """
    held = book.positions_in(ticker)
    reasons: list[str] = []
    quantity = sum((Decimal(str(p.quantity)) for p in held), Decimal(0))
    marks = {p.mark_price for p in held if p.mark_price is not None}
    price = Decimal(str(next(iter(marks)))) if len(marks) == 1 else None
    if price is None:
        reasons.append("missing or conflicting mark_price for analyzed instrument")
    exposure = quantity * price if price is not None else (Decimal(0) if not held else None)
    marked = all(p.mark_price is not None or p.quantity == 0 for p in book.positions)
    gross = (
        sum(
            (
                abs(Decimal(str(p.quantity))) * Decimal(str(p.mark_price))
                for p in book.positions
                if p.mark_price is not None
            ),
            Decimal(0),
        )
        if marked
        else None
    )
    equity = Decimal(str(book.portfolio_value)) if book.portfolio_value is not None else None
    if equity is None and book.cash is not None and marked:
        equity = Decimal(str(book.cash)) + sum(
            (
                Decimal(str(p.quantity)) * Decimal(str(p.mark_price))
                for p in book.positions
                if p.mark_price is not None
            ),
            Decimal(0),
        )
    if equity is None or equity <= 0:
        equity = None
        reasons.append(
            "missing positive portfolio value (cash and all marks required to derive it)"
        )
    weight = exposure / equity if exposure is not None and equity is not None else None
    settings = {
        (p.target_weight, p.max_weight, p.risk_budget, p.stop_price, p.leverage) for p in held
    }
    if len(settings) != 1:
        reasons.append("missing or conflicting instrument sizing inputs")
        return SizingGuidance(exposure, weight, None, None, None, None, tuple(reasons))
    target, maximum, budget, stop, leverage = next(iter(settings))
    gap = (
        Decimal(str(target)) * equity - exposure
        if target is not None and equity is not None and exposure is not None
        else None
    )
    if target is None:
        reasons.append("missing target_weight")
    stop_quantity = None
    if price is not None and budget is not None and stop is not None and Decimal(str(stop)) < price:
        stop_quantity = Decimal(str(budget)) / (price - Decimal(str(stop)))
    else:
        reasons.append("long stop sizing requires risk_budget and 0 < stop_price < mark_price")
    if quantity < 0:
        reasons.append("short holdings require separate cover guidance")
    limits = book.limits
    maxima = [
        Decimal(str(v)) for v in (maximum, limits.max_weight if limits else None) if v is not None
    ]
    if not maxima:
        reasons.append("missing max_weight")
    if book.cash is None:
        reasons.append("missing cash")
    if gross is None:
        reasons.append("missing marks for portfolio gross exposure")
    if book.open_orders is None:
        reasons.append("missing open_orders (provide [] when none)")
    pending_cash = Decimal(0)
    pending_gross = Decimal(0)
    pending_units = Decimal(0)
    orders_priced = True
    symbol = normalize_symbol(ticker)
    for order in book.open_orders or ():
        same = normalize_symbol(order.ticker) == symbol
        order_marks = {
            p.mark_price for p in book.positions_in(order.ticker) if p.mark_price is not None
        }
        mark = Decimal(str(next(iter(order_marks)))) if len(order_marks) == 1 else None
        order_price = Decimal(str(order.limit_price)) if order.limit_price is not None else mark
        if order_price is None:
            reasons.append(f"missing price for open order {order.ticker}")
            orders_priced = False
            continue
        units = Decimal(str(order.quantity))
        # Reserve both sides as gross commitments; never net pending buys/sells.
        pending_gross += units * max(order_price, mark or order_price)
        match order.side:
            case "buy":
                pending_cash += units * order_price
                if same:
                    pending_units += units
            case "sell":
                pass
            case unreachable:
                assert_never(unreachable)
    capacity = None
    if (
        price is not None
        and equity is not None
        and exposure is not None
        and maxima
        and book.cash is not None
        and gross is not None
        and book.open_orders is not None
        and orders_priced
        and quantity >= 0
    ):
        gearing = Decimal(str(leverage)) if leverage is not None else Decimal(1)
        portfolio_gearing = (
            Decimal(str(limits.max_leverage))
            if limits and limits.max_leverage is not None
            else Decimal(1)
        )
        reserve = (
            Decimal(str(limits.cash_reserve))
            if limits and limits.cash_reserve is not None
            else Decimal(0)
        )
        capacity = max(
            Decimal(0),
            min(
                min(maxima) * equity / price - quantity - pending_units,
                (
                    (Decimal(str(book.cash)) - reserve) * min(gearing, portfolio_gearing)
                    - pending_cash
                )
                / price,
                (equity * portfolio_gearing - gross - pending_gross) / price,
            ),
        )
    additional = None
    if capacity is not None and gap is not None and stop_quantity is not None and price is not None:
        additional = max(
            Decimal(0),
            min(capacity, gap / price - pending_units, stop_quantity - quantity - pending_units),
        )
    return SizingGuidance(
        exposure, weight, gap, stop_quantity, capacity, additional, tuple(reasons)
    )
