from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from tests.crypto_watch_fixtures import CUTOFF, MS, watch_wire as watch_wire
from tests.test_crypto_watch_scenarios import future, plan
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_backtest import evaluate_hypothetical
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch, render_watch
from tradingagents.dataflows.crypto_watch_models import (
    EPOCH,
    WatchDecision,
    WatchRequest,
    WatchSnapshot,
)
from tradingagents.dataflows.crypto_watch_scenario_models import HypotheticalPlan


@pytest.mark.parametrize("milliseconds,entry_hour", [(0, 10), (1, 11), (2000, 11)])
def test_live_entry_uses_server_observability_not_safety_cutoff(
    watch_wire, milliseconds: int, entry_hour: int
) -> None:
    server = CUTOFF + timedelta(hours=10, milliseconds=milliseconds)
    cutoff = server - timedelta(seconds=5)
    watch_wire["clock"] = {"serverTime": (server - EPOCH) // MS}
    watch_wire["end"] = (cutoff - EPOCH) // MS - 1
    for interval, hours in (("4h", 8), ("1h", 10)):
        for row in watch_wire["rows"][interval]:
            row[0] += hours * 3600000
            row[6] += hours * 3600000
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))

    decision = evaluate_watch(snapshot)
    outcome = evaluate_hypothetical(decision.hypothetical, future(12), CUTOFF + timedelta(hours=12))

    assert snapshot.cutoff == decision.cutoff == decision.hypothetical.decision_as_of == cutoff
    assert (
        snapshot.observable_at
        == decision.observable_at
        == decision.hypothetical.observable_at
        == server
    )
    assert outcome.entry_bar == CUTOFF + timedelta(hours=entry_hour)
    assert outcome.closed_holding_bars == 12 - entry_hour
    assert all(c.close_time < cutoff for s in snapshot.series for c in s.candles)
    assert f"observable_at={server.isoformat()}" in render_watch(decision)
    assert WatchDecision.model_validate_json(decision.model_dump_json()) == decision


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("mutation", ["missing", "naive", "cutoff", "later"])
def test_snapshot_rejects_forged_observability(watch_wire, historical: bool, mutation: str) -> None:
    snapshot = collect_watch(
        WatchRequest(asset="BTC", as_of=CUTOFF if historical else None, timeout_seconds=7)
    )
    data = snapshot.model_dump()
    match mutation:
        case "missing":
            data["observable_at"] = None
        case "naive":
            data["observable_at"] = snapshot.observable_at.replace(tzinfo=None)
        case "cutoff":
            data["observable_at"] = snapshot.cutoff - MS if historical else snapshot.cutoff
        case "later":
            data["observable_at"] = snapshot.observable_at + MS
    with pytest.raises(ValidationError):
        WatchSnapshot.model_validate(data)


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("mutation", ["missing", "naive", "before_cutoff", "plan_mismatch"])
def test_decision_rejects_forged_observability(watch_wire, waiting: bool, mutation: str) -> None:
    if waiting:
        watch_wire["rows"]["1d"] = []
    decision = evaluate_watch(collect_watch(WatchRequest(asset="BTC", timeout_seconds=7)))
    data = decision.model_dump()
    match mutation:
        case "missing":
            data["observable_at"] = None
        case "naive":
            data["observable_at"] = decision.observable_at.replace(tzinfo=None)
        case "before_cutoff":
            data["observable_at"] = decision.cutoff - MS
        case "plan_mismatch":
            if waiting:
                data["as_of"] = decision.cutoff - MS
            else:
                data["observable_at"] = decision.observable_at + MS
    with pytest.raises(ValidationError):
        WatchDecision.model_validate(data)


@pytest.mark.parametrize("observable", [CUTOFF - MS, CUTOFF.replace(tzinfo=None)])
def test_plan_rejects_naive_or_pre_evidence_observability(observable: datetime) -> None:
    with pytest.raises(ValidationError):
        HypotheticalPlan.model_validate(plan().model_dump() | {"observable_at": observable})


@pytest.mark.parametrize("as_of", [CUTOFF, CUTOFF.replace(tzinfo=None)])
def test_outcome_rejects_naive_or_pre_observable_evaluation(as_of: datetime) -> None:
    delayed = HypotheticalPlan.model_validate(
        plan().model_dump() | {"observable_at": CUTOFF + timedelta(seconds=2)}
    )
    with pytest.raises(ContextContractError, match="hypothetical future scope"):
        evaluate_hypothetical(delayed, future(), as_of)
