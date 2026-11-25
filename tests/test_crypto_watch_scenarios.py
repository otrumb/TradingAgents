from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tests.crypto_watch_fixtures import CUTOFF, watch_wire as watch_wire, wire_rows
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_backtest import (
    evaluate_hypothetical,
    evaluate_watch_history,
)
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch, render_watch
from tradingagents.dataflows.crypto_watch_models import (
    DURATIONS,
    WatchCandle,
    WatchRequest,
    WatchSeries,
    WatchSnapshot,
)
from tradingagents.dataflows.crypto_watch_report import render_hypothetical_outcome
from tradingagents.dataflows.crypto_watch_scenario_models import (
    HypotheticalOutcome,
    HypotheticalPlan,
    WatchCosts,
)


def plan(direction="long", **changes):
    return HypotheticalPlan(
        direction=direction,
        decision_as_of=CUTOFF,
        observable_at=CUTOFF,
        invalidation=Decimal(90 if direction == "long" else 110),
        swing_start=CUTOFF - timedelta(hours=16),
        swing_end=CUTOFF - timedelta(hours=4, milliseconds=1),
        confirmed_at=CUTOFF - timedelta(milliseconds=1),
        **changes,
    )


def future(count=1, **last):
    candles = []
    for index in range(count):
        values = {"open": "100", "high": "105", "low": "95", "close": "102"}
        if index == count - 1:
            values.update(last)
        start = CUTOFF + timedelta(hours=index)
        candles.append(
            WatchCandle(
                interval="1h",
                open_time=start,
                close_time=start + timedelta(hours=1, milliseconds=-1),
                volume=Decimal(1),
                **values,
            )
        )
    return WatchSeries(interval="1h", candles=tuple(candles))


@pytest.mark.parametrize(
    "direction,prices,exit_price",
    [
        ("long", {"low": "90"}, "90"),
        ("short", {"high": "110"}, "110"),
        ("long", {"open": "85", "low": "80"}, "85"),
        ("short", {"open": "115", "high": "120"}, "115"),
    ],
)
def test_touch_including_equality_and_gap_prices_adversely(direction, prices, exit_price):
    outcome = evaluate_hypothetical(plan(direction), future(**prices), CUTOFF + timedelta(hours=1))
    assert outcome.status == "stopped"
    assert outcome.entry_price == Decimal(prices.get("open", "100"))
    assert outcome.exit_price == Decimal(exit_price)
    assert outcome.closed_holding_bars == 1


@pytest.mark.parametrize(
    "count,status", [(167, "pending_outcome"), (168, "expired"), (169, "expired")]
)
def test_expiry_counts_entry_bar_and_ignores_bar_169(count, status):
    outcome = evaluate_hypothetical(plan(), future(count), CUTOFF + timedelta(hours=count))
    assert outcome.status == status
    assert outcome.closed_holding_bars == min(count, 168)
    assert outcome.exit_price == (Decimal(102) if count >= 168 else None)


def test_touch_on_expiry_bar_wins_over_expiry_close():
    outcome = evaluate_hypothetical(plan(), future(168, low="90"), CUTOFF + timedelta(hours=168))
    assert outcome.status == "stopped"
    assert outcome.exit_price == Decimal(90)


@pytest.mark.parametrize("microseconds,status", [(0, "pending_entry"), (1, "pending_outcome")])
def test_future_bar_must_be_strictly_finalized(microseconds, status):
    bars = future()
    cutoff = bars.candles[0].close_time + timedelta(microseconds=microseconds)
    outcome = evaluate_hypothetical(plan(), bars, cutoff)
    assert outcome.status == status


def test_missing_entry_bar_never_substitutes_later_open():
    bars = WatchSeries(interval="1h", candles=future(2).candles[1:])
    outcome = evaluate_hypothetical(plan(), bars, CUTOFF + timedelta(hours=2))
    assert outcome.status == "pending_entry"
    assert outcome.entry_price is None


def test_gap_prevents_claiming_expiry_or_scanning_later_touch():
    bars = future(3, low="90")
    bars = WatchSeries(interval="1h", candles=(bars.candles[0], bars.candles[2]))
    outcome = evaluate_hypothetical(plan(), bars, CUTOFF + timedelta(hours=3))
    assert outcome.status == "pending_outcome"
    assert outcome.closed_holding_bars == 1
    assert outcome.net_return is None


@pytest.mark.parametrize("direction,close,gross", [("long", "120", "0.2"), ("short", "80", "0.2")])
def test_explicit_decimal_costs_debit_both_directions(direction, close, gross):
    costs = WatchCosts(
        round_trip_fee=Decimal("0.002"),
        slippage_per_side=Decimal("0.003"),
        funding_debit_per_8h=Decimal("0.004"),
    )
    prices = (
        {"close": close, "high": "120"} if direction == "long" else {"close": close, "low": "80"}
    )
    outcome = evaluate_hypothetical(
        plan(direction, costs=costs), future(168, **prices), CUTOFF + timedelta(hours=168)
    )
    assert outcome.gross_return == Decimal(gross)
    assert outcome.fee_slippage_return == Decimal("0.008")
    assert outcome.funding_return == Decimal("0.084")
    assert outcome.cost_return == Decimal("0.092")
    assert outcome.net_return == Decimal("0.108")


def test_completed_outcome_report_renders_each_cost_component():
    outcome = HypotheticalOutcome(
        status="stopped",
        reason="invalidation",
        evaluated_as_of=CUTOFF + timedelta(hours=2),
        entry_bar=CUTOFF,
        entry_price=Decimal("100"),
        outcome_bar=CUTOFF + timedelta(hours=1),
        exit_price=Decimal("110"),
        closed_holding_bars=1,
        gross_return=Decimal("0.1"),
        fee_slippage_return=Decimal("0.008"),
        funding_return=Decimal("0.004"),
        cost_return=Decimal("0.012"),
        net_return=Decimal("0.088"),
    )

    report = render_hypothetical_outcome(outcome)

    assert "gross=0.1; fee+slippage=0.008; funding=0.004; total cost=0.012; net=0.088" in report


@pytest.mark.parametrize("direction", [1, -1])
def test_decision_freezes_confirmed_swing_and_serializes_pending_plan(watch_wire, direction):
    watch_wire["rows"] = {i: wire_rows(i, direction) for i in DURATIONS}
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    decision = evaluate_watch(snapshot)
    assert decision.hypothetical.invalidation == Decimal(1 if direction > 0 else 400)
    assert decision.hypothetical.confirmed_at == snapshot.series[1].candles[-1].close_time
    assert decision.latest_closes == tuple(s.candles[-1].close for s in snapshot.series)
    assert "pending_entry" in render_watch(decision)
    assert (
        HypotheticalPlan.model_validate_json(decision.hypothetical.model_dump_json())
        == decision.hypothetical
    )


def test_wait_report_has_exact_missing_pullback_and_no_hypothetical_entry(watch_wire):
    watch_wire["rows"]["4h"][-2][3] = watch_wire["rows"]["4h"][-2][4]
    decision = evaluate_watch(collect_watch(WatchRequest(asset="BTC", timeout_seconds=7)))
    missing = [c.name for c in decision.conditions if c.name.startswith("long:") and not c.met]
    assert missing == ["long:min_pullback_low_minus_own_ema20"]
    assert decision.hypothetical is None
    assert "Hypothetical entry: none (waiting). Invalidation: N/A." in render_watch(decision)


def test_history_filters_future_signals_but_uses_future_outcomes(watch_wire):
    snapshot = collect_watch(WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7))
    series = (
        *snapshot.series[:2],
        WatchSeries(interval="1h", candles=snapshot.series[2].candles + future(low="90").candles),
    )
    archive = WatchSnapshot(
        asset="BTC",
        symbol="BTCUSDT",
        as_of=CUTOFF + timedelta(hours=1),
        cutoff=CUTOFF + timedelta(hours=1),
        observable_at=CUTOFF + timedelta(hours=1),
        server_time=None,
        series=series,
    )
    (result,) = evaluate_watch_history(archive, (CUTOFF,))
    assert result.decision == evaluate_watch(snapshot)
    assert result.decision.observable_at == result.decision.hypothetical.observable_at == CUTOFF
    assert result.hypothetical_outcome.entry_price == Decimal(100)
    assert result.hypothetical_outcome.status == "pending_outcome"


def test_historical_evaluator_rejects_live_snapshot(watch_wire):
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    with pytest.raises(ContextContractError, match="historical watch archive"):
        evaluate_watch_history(snapshot, (CUTOFF,))


@pytest.mark.parametrize("value", ["-0.1", "NaN", "Infinity"])
def test_cost_boundary_rejects_invalid_rates(value):
    with pytest.raises(ValidationError):
        WatchCosts(round_trip_fee=Decimal(value))


def test_hypothetical_plan_has_no_executable_trade_fields():
    fields = set(plan().model_dump(mode="json"))

    assert fields.isdisjoint(
        {
            "quantity",
            "leverage",
            "order_type",
            "execution_method",
            "exchange_credentials",
        }
    )
