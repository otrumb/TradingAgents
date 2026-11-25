"""Decimal EMA watch rules with deterministic quality-first wait precedence."""

from datetime import datetime
from decimal import Decimal
from typing import Literal, TypedDict, assert_never

from .crypto_models import Asset, ContextContractError
from .crypto_spot_models import SpotSymbol
from .crypto_watch_diagnostics import diagnose_watch
from .crypto_watch_models import (
    DURATIONS,
    FRESHNESS,
    REQUIRED,
    Direction,
    WatchCondition,
    WatchDecision,
    WatchSnapshot,
)
from .crypto_watch_report import render_watch_details
from .crypto_watch_scenario_models import HypotheticalPlan


class _Evidence(TypedDict):
    asset: Asset
    symbol: SpotSymbol
    source: Literal["https://fapi.binance.com"]
    as_of: datetime | None
    cutoff: datetime | None
    observable_at: datetime | None
    latest_close_times: tuple[datetime | None, ...]
    statuses: tuple[Literal["ok", "unavailable", "insufficient_history", "stale"], ...]
    latest_closes: tuple[Decimal | None, ...]
    conditions: tuple[WatchCondition, ...]


class _Signals(TypedDict):
    bias: Direction
    setup: Direction
    trigger: Direction


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
        "observable_at": snapshot.observable_at,
        "latest_close_times": tuple(
            s.candles[-1].close_time if s.candles else None for s in snapshot.series
        ),
        "statuses": tuple(statuses),
        "latest_closes": tuple(s.candles[-1].close if s.candles else None for s in snapshot.series),
        "conditions": (),
    }
    for reason in ("unavailable", "insufficient_history", "stale"):
        if reason in statuses:
            return WatchDecision(**evidence, reasons=(reason,))
    evidence["conditions"] = diagnose_watch(snapshot)
    directions: list[Direction] = []
    for interval in ("1d", "4h", "1h"):
        long = all(
            c.met
            for c in evidence["conditions"]
            if c.interval == interval and c.name.startswith("long:")
        )
        short = all(
            c.met
            for c in evidence["conditions"]
            if c.interval == interval and c.name.startswith("short:")
        )
        directions.append("long" if long else "short" if short else "neutral")
    bias, setup, trigger = directions
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
    four = snapshot.series[1].candles
    if cutoff is None or snapshot.observable_at is None:
        raise ContextContractError(field="watch aligned cutoff")
    plan = HypotheticalPlan(
        direction=bias,
        decision_as_of=cutoff,
        observable_at=snapshot.observable_at,
        invalidation=min(c.low for c in four[-4:-1])
        if bias == "long"
        else max(c.high for c in four[-4:-1]),
        swing_start=four[-4].open_time,
        swing_end=four[-2].close_time,
        confirmed_at=four[-1].close_time,
    )
    return WatchDecision(**evidence, **signals, action=action, hypothetical=plan)


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
    return "\n".join(lines) + "\n\n" + render_watch_details(decision)
