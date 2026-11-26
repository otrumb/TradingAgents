from datetime import timedelta
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from tests.crypto_watch_fixtures import CUTOFF, MS, watch_wire as watch_wire
from tests.test_crypto_watch_scenarios import future
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_aggregate import (
    HistoricalWatchAggregate,
    aggregate_watch_history,
    render_watch_aggregate,
)
from tradingagents.dataflows.crypto_watch_backtest import (
    HistoricalWatchResult,
    evaluate_hypothetical,
    evaluate_watch_history,
)
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import (
    EPOCH,
    WatchDecision,
    WatchRequest,
    WatchSnapshot,
)
from tradingagents.reporting import write_report_tree


@pytest.mark.parametrize("limit", [1, 100, 300, 1500])
def test_history_limit_reaches_public_collector(watch_wire, limit):
    watch_wire["limit"] = limit
    snapshot = collect_watch(
        WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7, history_limit=limit)
    )
    assert tuple(len(s.candles) for s in snapshot.series) == (55, 24, 24)
    assert [query["limit"] for _, query in watch_wire["calls"]] == [[str(limit)]] * 3


def test_default_history_limit_preserves_graph_collection(watch_wire):
    request = WatchRequest(asset="BTC", timeout_seconds=7)
    snapshot = collect_watch(request)
    assert request.history_limit == 100
    assert snapshot.cutoff == CUTOFF
    assert [query["limit"] for path, query in watch_wire["calls"] if path.endswith("klines")] == [
        ["100"]
    ] * 3


@pytest.mark.parametrize("limit", [0, 1501, -1, 1.5, True])
def test_history_limit_rejects_outside_binance_integer_bounds(limit):
    with pytest.raises(ValidationError):
        WatchRequest(asset="BTC", history_limit=limit)


def test_extended_collected_archive_resolves_168_bar_expiry_without_future_signals(watch_wire):
    original = collect_watch(WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7))
    rows = watch_wire["rows"]["1h"]
    first = rows[0]
    prefix = []
    for hours in range(107, 0, -1):
        row = list(first)
        row[0] -= hours * 3600000
        row[6] -= hours * 3600000
        prefix.append(row)
    tail = []
    for candle in future(169).candles:
        tail.append(
            [
                (candle.open_time - EPOCH) // MS,
                str(candle.open),
                str(candle.high),
                str(candle.low),
                str(candle.close),
                "1",
                (candle.close_time - EPOCH) // MS,
                "100",
                1,
                "1",
                "100",
                "0",
            ]
        )
    watch_wire["rows"]["1h"] = prefix + rows + tail
    end = CUTOFF + timedelta(hours=169)
    watch_wire["limit"] = 300
    watch_wire["end"] = (end - EPOCH) // MS - 1
    archive = collect_watch(
        WatchRequest(asset="BTC", as_of=end, timeout_seconds=7, history_limit=300)
    )

    stamp = CUTOFF + timedelta(seconds=9)
    (result,) = evaluate_watch_history(archive, (stamp,))

    assert len(archive.series[2].candles) == 300
    expected = WatchSnapshot.model_validate(original.model_dump() | {"observable_at": stamp})
    assert result.decision == evaluate_watch(expected)
    assert result.hypothetical_outcome.status == "expired"
    assert result.hypothetical_outcome.closed_holding_bars == 168
    assert result.hypothetical_outcome.entry_bar == CUTOFF + timedelta(hours=1)
    assert result.hypothetical_outcome.outcome_bar == end - MS


def test_aggregate_counts_and_exact_completed_decimal_means(watch_wire):
    snapshot = collect_watch(WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7))
    decision = evaluate_watch(snapshot)
    plan = decision.hypothetical
    outcomes = (
        evaluate_hypothetical(plan, future(0), CUTOFF),
        evaluate_hypothetical(plan, future(1), CUTOFF + timedelta(hours=1)),
        evaluate_hypothetical(plan, future(1, low="1"), CUTOFF + timedelta(hours=1)),
        evaluate_hypothetical(plan, future(168), CUTOFF + timedelta(hours=168)),
    )
    wait = WatchDecision(
        asset="BTC",
        symbol="BTCUSDT",
        as_of=CUTOFF,
        cutoff=CUTOFF,
        observable_at=CUTOFF,
        reasons=("unavailable",),
    )
    results = (HistoricalWatchResult(decision=wait, hypothetical_outcome=None),) + tuple(
        HistoricalWatchResult(decision=decision, hypothetical_outcome=o) for o in outcomes
    )

    with localcontext() as context:
        context.prec = 6
        summary = aggregate_watch_history(results)

    assert (
        summary.total_decisions,
        summary.aligned_plans,
        summary.pending_entries,
        summary.pending_outcomes,
        summary.stopped,
        summary.expired,
        summary.completed_count,
    ) == (5, 4, 1, 1, 1, 1, 2)
    assert summary.mean_gross_return == Decimal("-0.485")
    assert summary.mean_fee_slippage_return == Decimal("0.002")
    assert summary.mean_funding_return == Decimal("0.00105625")
    assert summary.mean_cost_return == Decimal("0.00305625")
    assert summary.mean_net_return == Decimal("-0.48805625")
    assert HistoricalWatchAggregate.model_validate_json(summary.model_dump_json()) == summary
    assert "net=-0.48805625" in render_watch_aggregate(summary)


@pytest.mark.parametrize("pending", [False, True])
def test_zero_completed_has_no_return_means_or_performance_claim(watch_wire, pending):
    results = ()
    if pending:
        decision = evaluate_watch(collect_watch(WatchRequest(asset="BTC", timeout_seconds=7)))
        outcome = evaluate_hypothetical(decision.hypothetical, future(0), decision.observable_at)
        results = (HistoricalWatchResult(decision=decision, hypothetical_outcome=outcome),)
    summary = aggregate_watch_history(results)
    assert summary.completed_count == 0
    assert summary.mean_gross_return is summary.mean_cost_return is summary.mean_net_return is None
    assert summary.mean_fee_slippage_return is summary.mean_funding_return is None
    assert (
        summary.overlapping_pair_count,
        summary.overlap_pair_fraction,
        summary.peak_concurrent,
        summary.non_overlapping_count,
    ) == (0, Decimal(0), 0, 0)
    assert "no performance claim" in render_watch_aggregate(summary)


def test_watch_decision_sidecar_round_trips_existing_state(watch_wire, tmp_path):
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    decision = evaluate_watch(snapshot)
    state = {
        "company_of_interest": "BTC-USD",
        "asset_type": "crypto",
        "crypto_futures_watch_evidence": snapshot.model_dump_json(),
        "crypto_futures_watch_data": decision.model_dump_json(),
    }
    write_report_tree(state, "BTC", tmp_path, html=False)
    sidecar = tmp_path / "evidence" / "crypto_futures_watch_decision.json"
    assert WatchDecision.model_validate_json(sidecar.read_text(encoding="utf-8")) == decision


def test_report_removes_stale_watch_decision_sidecar(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    sidecar = evidence / "crypto_futures_watch_decision.json"
    sidecar.write_text("stale", encoding="utf-8")
    write_report_tree({}, "BTC", tmp_path, html=False)
    assert not sidecar.exists()


def test_invalid_watch_decision_never_becomes_sidecar(watch_wire, tmp_path):
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    state = {
        "company_of_interest": "BTC-USD",
        "asset_type": "crypto",
        "crypto_futures_watch_evidence": snapshot.model_dump_json(),
        "crypto_futures_watch_data": '{"action":"consider_long"}',
    }
    with pytest.raises((ValidationError, ContextContractError)):
        write_report_tree(state, "BTC", tmp_path, html=False)
    assert not (tmp_path / "evidence" / "crypto_futures_watch_decision.json").exists()
