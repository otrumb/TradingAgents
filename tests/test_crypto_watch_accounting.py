from datetime import timedelta
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from tests.crypto_watch_fixtures import CUTOFF, MS, watch_wire as watch_wire
from tests.test_crypto_watch_scenarios import future, plan
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_aggregate import aggregate_watch_history
from tradingagents.dataflows.crypto_watch_backtest import (
    HistoricalWatchResult,
    evaluate_hypothetical,
    evaluate_watch_history,
)
from tradingagents.dataflows.crypto_watch_models import EPOCH, WatchDecision, WatchRequest
from tradingagents.dataflows.crypto_watch_scenario_models import HypotheticalOutcome


def test_each_wait_reason_counts_decisions_once_even_when_reasons_cooccur():
    reasons = (
        "unavailable",
        "insufficient_history",
        "stale",
        "neutral_bias",
        "timeframe_conflict",
        "missing_setup",
        "missing_trigger",
    )
    decision = WatchDecision(
        asset="BTC",
        symbol="BTCUSDT",
        as_of=CUTOFF,
        cutoff=CUTOFF,
        observable_at=CUTOFF,
        reasons=(*reasons, "unavailable"),
    )

    summary = aggregate_watch_history(
        (
            HistoricalWatchResult(
                decision=decision,
                hypothetical_outcome=None,
            ),
        )
    )

    assert summary.total_decisions == 1
    assert summary.aligned_plans == summary.completed_count == 0
    assert tuple(getattr(summary, reason) for reason in reasons) == (1,) * 7


def test_history_accounts_for_insufficient_warmup(watch_wire):
    end = CUTOFF + timedelta(seconds=9)
    watch_wire["end"] = (end - EPOCH) // MS - 1
    archive = collect_watch(WatchRequest(asset="BTC", as_of=end, timeout_seconds=7))
    times = (end - timedelta(hours=1), end)

    summary = aggregate_watch_history(evaluate_watch_history(archive, times))

    assert summary.total_decisions == 2
    assert summary.insufficient_history == 1
    assert summary.aligned_plans == summary.pending_entries == 1


@pytest.mark.parametrize("offsets", [(0, 0), (0, -1)])
def test_history_rejects_duplicate_or_unsorted_decision_times(watch_wire, offsets):
    archive = collect_watch(WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7))
    times = tuple(CUTOFF + timedelta(hours=offset) for offset in offsets)

    with pytest.raises(ContextContractError, match="strictly increasing"):
        evaluate_watch_history(archive, times)


@pytest.mark.parametrize(
    "intervals,pairs,peak,selected",
    [
        (((0, 2), (2, 3), (4, 5)), 1, 2, 2),
        (((0, 9), (1, 2), (3, 4), (5, 6)), 3, 2, 3),
        (((0, 2), (0, 2), (0, 2)), 3, 3, 1),
        (((0, 0),), 0, 1, 1),
    ],
)
def test_closed_interval_overlap_and_earliest_finish_keep_all_means(
    intervals, pairs, peak, selected
):
    results = []
    for index, (start, end) in enumerate(reversed(intervals)):
        scenario = plan()
        decision = WatchDecision(
            asset="BTC",
            symbol="BTCUSDT",
            as_of=CUTOFF,
            cutoff=CUTOFF,
            observable_at=CUTOFF,
            action="consider_long",
            bias="long",
            setup="long",
            trigger="long",
            statuses=("ok",) * 3,
            hypothetical=scenario,
        )
        outcome = HypotheticalOutcome(
            status="stopped",
            reason="test stop",
            evaluated_as_of=CUTOFF + timedelta(hours=10),
            entry_bar=CUTOFF + timedelta(hours=start),
            entry_price=Decimal(100),
            outcome_bar=CUTOFF + timedelta(hours=end),
            exit_price=Decimal(90),
            closed_holding_bars=1,
            gross_return=Decimal(index),
            fee_slippage_return=Decimal("0.01"),
            funding_return=Decimal("0.02"),
            cost_return=Decimal("0.03"),
            net_return=Decimal(index) - Decimal("0.03"),
        )
        results.append(HistoricalWatchResult(decision=decision, hypothetical_outcome=outcome))

    summary = aggregate_watch_history(tuple(results))

    count = len(intervals)
    assert summary.completed_count == count
    assert summary.overlapping_pair_count == pairs
    assert summary.overlap_pair_fraction == (
        Decimal(pairs) / (count * (count - 1) // 2) if count > 1 else Decimal(0)
    )
    assert summary.peak_concurrent == peak
    assert summary.non_overlapping_count == selected
    assert summary.mean_gross_return == Decimal(count - 1) / 2
    assert summary.mean_net_return == Decimal(count - 1) / 2 - Decimal("0.03")


@pytest.mark.parametrize(
    "changes",
    [
        {"cost_return": "0.5"},
        {"net_return": "0.5"},
        {"fee_slippage_return": None},
        {"funding_return": None},
        {"funding_return": "-0.1"},
        {"gross_return": "NaN"},
    ],
)
def test_completed_outcome_rejects_missing_costs_and_broken_invariants(changes):
    outcome = evaluate_hypothetical(plan(), future(low="90"), CUTOFF + timedelta(hours=1))

    with pytest.raises(ValidationError):
        HypotheticalOutcome.model_validate({**outcome.model_dump(), **changes})


def test_outcome_cost_invariants_ignore_callers_decimal_precision():
    with localcontext() as context:
        context.prec = 3
        outcome = evaluate_hypothetical(plan(), future(low="90"), CUTOFF + timedelta(hours=1))

    assert outcome.fee_slippage_return == Decimal("0.002")
    assert outcome.funding_return == Decimal("0.0000125")
    assert outcome.cost_return == Decimal("0.0020125")
    assert outcome.net_return == Decimal("-0.1020125")


@pytest.mark.parametrize("field", ["fee_slippage_return", "funding_return"])
def test_pending_outcomes_reject_cost_components(field):
    outcome = evaluate_hypothetical(plan(), future(0), CUTOFF)

    with pytest.raises(ValidationError, match="pending returns"):
        HypotheticalOutcome.model_validate({**outcome.model_dump(), field: Decimal(0)})
