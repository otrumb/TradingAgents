"""Decimal EMA watch rules with deterministic quality-first wait precedence."""

from datetime import datetime
from decimal import Context, Decimal, localcontext
from typing import Literal, TypedDict, assert_never

from .crypto_models import Asset
from .crypto_spot_models import SpotSymbol
from .crypto_watch_models import (
    DURATIONS,
    FRESHNESS,
    REQUIRED,
    Direction,
    WatchDecision,
    WatchSnapshot,
)


class _Evidence(TypedDict):
    asset: Asset
    symbol: SpotSymbol
    source: Literal["https://fapi.binance.com"]
    as_of: datetime | None
    cutoff: datetime | None
    latest_close_times: tuple[datetime | None, ...]
    statuses: tuple[Literal["ok", "unavailable", "insufficient_history", "stale"], ...]


class _Signals(TypedDict):
    bias: Direction
    setup: Direction
    trigger: Direction


def _ema(closes: tuple[Decimal, ...], period: int) -> tuple[Decimal, ...]:
    """Seed EMA with first close; recurse with alpha=2/(period+1), precision 28."""
    with localcontext(Context(prec=28)):
        alpha = Decimal(2) / Decimal(period + 1)
        values = [closes[0]]
        for close in closes[1:]:
            values.append(values[-1] + alpha * (close - values[-1]))
        return tuple(values)


def evaluate_watch(snapshot: WatchSnapshot) -> WatchDecision:
    """Resolve quality, bias, conflict, setup, then trigger; emit one wait reason."""
    cutoff = snapshot.cutoff
    statuses: list[Literal["ok", "unavailable", "insufficient_history", "stale"]] = []
    for series in snapshot.series:
        candles = series.candles
        if cutoff is None or series.failure is not None or not candles:
            statuses.append("unavailable")
        elif len(candles) < REQUIRED[series.interval] or any(
            b.open_time - a.open_time != DURATIONS[series.interval]
            for a, b in zip(candles, candles[1:], strict=False)
        ):
            statuses.append("insufficient_history")
        elif cutoff - candles[-1].close_time > FRESHNESS[series.interval]:
            statuses.append("stale")
        else:
            statuses.append("ok")
    evidence: _Evidence = {
        "asset": snapshot.asset,
        "symbol": snapshot.symbol,
        "source": snapshot.source,
        "as_of": cutoff,
        "cutoff": cutoff,
        "latest_close_times": tuple(
            s.candles[-1].close_time if s.candles else None for s in snapshot.series
        ),
        "statuses": tuple(statuses),
    }
    for reason in ("unavailable", "insufficient_history", "stale"):
        if reason in statuses:
            return WatchDecision(**evidence, reasons=(reason,))
    daily, four_hour, hourly = (series.candles for series in snapshot.series)
    daily_closes = tuple(c.close for c in daily)
    ema20 = _ema(daily_closes, 20)[-1]
    ema50 = _ema(daily_closes, 50)[-1]
    bias: Direction = (
        "long"
        if daily[-1].close > ema20 > ema50
        else "short"
        if daily[-1].close < ema20 < ema50
        else "neutral"
    )
    averages = _ema(tuple(c.close for c in four_hour), 20)
    latest, previous = four_hour[-1], four_hour[-2]
    setup: Direction = "neutral"
    # Each pullback candle compares against its own EMA, excluding latest candle.
    prior = tuple(zip(four_hour[-4:-1], averages[-4:-1], strict=True))
    if (
        latest.close > averages[-1]
        and latest.close > previous.high
        and any(min(c.low, c.close) <= average for c, average in prior)
    ):
        setup = "long"
    elif (
        latest.close < averages[-1]
        and latest.close < previous.low
        and any(max(c.high, c.close) >= average for c, average in prior)
    ):
        setup = "short"
    trigger: Direction = (
        "long"
        if hourly[-1].close > max(c.high for c in hourly[-21:-1])
        else "short"
        if hourly[-1].close < min(c.low for c in hourly[-21:-1])
        else "neutral"
    )
    signals: _Signals = {"bias": bias, "setup": setup, "trigger": trigger}
    if bias == "neutral":
        return WatchDecision(**evidence, **signals, reasons=("neutral_bias",))
    if (setup != "neutral" and setup != bias) or (trigger != "neutral" and trigger != bias):
        return WatchDecision(**evidence, **signals, reasons=("timeframe_conflict",))
    if setup == "neutral":
        return WatchDecision(**evidence, **signals, reasons=("missing_setup",))
    if trigger == "neutral":
        return WatchDecision(**evidence, **signals, reasons=("missing_trigger",))
    match bias:
        case "long":
            action = "consider_long"
        case "short":
            action = "consider_short"
        case unreachable:
            assert_never(unreachable)
    return WatchDecision(**evidence, **signals, action=action)


def render_watch(decision: WatchDecision) -> str:
    """Render provenance and stage evidence without wall-clock-dependent content."""
    stamp = decision.as_of.isoformat() if decision.as_of is not None else "unavailable"
    lines = [
        f"- Asset: {decision.asset}",
        f"- Symbol: {decision.symbol}",
        f"- Source: {decision.source}",
        f"- as_of: {stamp}",
        f"- Decision: {decision.action}",
        f"- Wait reason: {', '.join(decision.reasons) or 'none'}",
        "",
        "| Stage | Signal | Latest close time | Status |",
        "| --- | --- | --- | --- |",
    ]
    for label, signal, close_time, status in zip(
        ("1d Bias", "4h Setup", "1h Trigger"),
        (decision.bias, decision.setup, decision.trigger),
        decision.latest_close_times,
        decision.statuses,
        strict=True,
    ):
        stamp = close_time.isoformat() if close_time is not None else "unavailable"
        lines.append(f"| {label} | {signal} | {stamp} | {status} |")
    return "\n".join(lines)
