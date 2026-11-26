import json
from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from tradingagents.agents.state import AgentState
from tradingagents.dataflows.crypto_models import SourceFailure
from tradingagents.dataflows.crypto_pattern_models import (
    CryptoPatternEvidence,
    CryptoPatternSummary,
)
from tradingagents.dataflows.crypto_pattern_runtime import classic_pattern_evidence
from tradingagents.dataflows.crypto_spot_models import CompletedDailyCandle, SpotPriceContext
from tradingagents.dataflows.crypto_watch import collect_watch as collect_finalized_watch
from tradingagents.dataflows.crypto_watch_models import DURATIONS, WatchSeries, WatchSnapshot
from tradingagents.graph import run_decisions, trading_graph
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph

END_AT = datetime(2024, 1, 3, tzinfo=UTC)
RETRIEVED_AT = datetime(2026, 1, 3, tzinfo=UTC)


def spot_context(asset="BTC", *, historical=True, indices=(0,)):
    analysis_date = date(2024, 1, 2) if historical else None
    candles = tuple(
        CompletedDailyCandle(
            asset=asset,
            symbol=f"{asset}USDT",
            retrieved_at=RETRIEVED_AT,
            server_time=None if historical else END_AT + timedelta(seconds=5),
            cutoff=END_AT,
            analysis_date=analysis_date,
            finality_lag_ms=0 if historical else 5000,
            open_time=END_AT - timedelta(days=160 - i),
            close_time=END_AT - timedelta(days=159 - i, milliseconds=1),
            open=100,
            high=110,
            low=90,
            close=100,
            volume=1,
        )
        for i in indices
    )
    return SpotPriceContext(
        asset=asset,
        analysis_date=analysis_date,
        daily_candles=candles,
        semantics="historical_finalized" if historical else "current_run_only",
    )


def watch_snapshot(asset="BTC"):
    return WatchSnapshot(
        asset=asset,
        symbol=f"{asset}USDT",
        cutoff=END_AT,
        as_of=END_AT,
        observable_at=END_AT + timedelta(seconds=4),
        server_time=None,
        series=tuple(WatchSeries(interval=i) for i in ("1d", "4h", "1h")),
    )


@pytest.fixture
def graph(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network fetch")

    monkeypatch.setattr(requests.Session, "send", forbidden)
    instance = object.__new__(TradingAgentsGraph)
    instance.propagator = Propagator()
    instance.resolve_instrument_context = lambda *args: "instrument identity"
    return instance


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("historical", [False, True])
def test_run_reuses_finalized_objects_once(monkeypatch, graph, asset, historical):
    spot = spot_context(asset, historical=historical, indices=range(160))
    watches, wire_calls = [], []
    clock = END_AT + timedelta(minutes=30, seconds=5)

    def send(session, request, **kwargs):
        url = urlsplit(request.url)
        wire_calls.append(url.path)
        payload = {"serverTime": int(clock.timestamp() * 1000)}
        if url.path == "/fapi/v1/klines":
            query = parse_qs(url.query)
            duration = DURATIONS[query["interval"][0]]
            rows = [
                [
                    int((END_AT - duration * (160 - i)).timestamp() * 1000),
                    "100",
                    "110",
                    "90",
                    "100",
                    "1",
                    int((END_AT - duration * (159 - i)).timestamp() * 1000) - 1,
                    "100",
                    1,
                    "1",
                    "100",
                    "0",
                ]
                for i in range(160 if historical else 161)
            ]
            payload = rows[-int(query["limit"][0]) :]
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    fetches, executions = [], []

    def collect_spot(request):
        fetches.append(request)
        return spot

    def collect_watch(request):
        fetches.append(request)
        watch = collect_finalized_watch(request)
        watches.append(watch)
        return watch

    def evidence(supplied_spot, supplied_watch, observed_at):
        assert supplied_spot is spot and supplied_watch is watches[0]
        executions.append(observed_at)
        return classic_pattern_evidence(supplied_spot, supplied_watch, observed_at)

    monkeypatch.setattr(trading_graph, "collect_spot_prices", collect_spot)
    monkeypatch.setattr(run_decisions, "collect_watch", collect_watch)
    monkeypatch.setattr(run_decisions, "classic_pattern_evidence", evidence)
    trade_date = "2024-01-02" if historical else datetime.now(UTC).date().isoformat()

    state = graph.create_run_state(f"{asset}-USD", trade_date, "crypto")

    full = CryptoPatternEvidence.model_validate_json(state["crypto_pattern_evidence"])
    compact = CryptoPatternSummary.model_validate_json(state["crypto_pattern_report"])
    assert executions == [RETRIEVED_AT]
    assert len(fetches) == 2
    assert fetches[0].analysis_date == (date(2024, 1, 2) if historical else None)
    assert fetches[1].as_of == (END_AT if historical else None)
    assert fetches[1].history_limit == 161
    assert wire_calls.count("/fapi/v1/klines") == 3
    assert wire_calls.count("/fapi/v1/time") == (0 if historical else 2)
    assert full.analysis_date == spot.analysis_date
    assert compact == full.summary()
    assert [(row.status, row.bar_count) for row in full.streams] == [("ok", 160)] * 4
    assert full.streams[0].provenance.retrieved_at == RETRIEVED_AT
    assert all(row.required_bars == 160 for row in full.streams)
    assert all(row.provenance.retrieved_at == watches[0].observable_at for row in full.streams[1:])
    assert all(row.last_bar_at is None or row.last_bar_at < END_AT for row in full.streams)


@pytest.mark.parametrize(
    "asset_type,ticker", [("stock", "BTC"), ("stock", "NVDA"), ("crypto", "DOGE-USD")]
)
def test_unsupported_assets_leave_empty_fields_without_execution(
    monkeypatch, graph, asset_type, ticker
):
    def forbidden(*args):
        pytest.fail("Unsupported asset invoked evidence or collector")

    monkeypatch.setattr(trading_graph, "collect_spot_prices", forbidden)
    monkeypatch.setattr(run_decisions, "collect_watch", forbidden)
    monkeypatch.setattr(run_decisions, "classic_pattern_evidence", forbidden)

    state = graph.create_run_state(ticker, "2024-01-02", asset_type)

    assert state["crypto_pattern_evidence"] == state["crypto_pattern_report"] == ""


@pytest.mark.parametrize(
    "indices,status",
    [((), "unavailable"), ((0,), "insufficient_history"), ((0, 2), "missing_bars")],
)
def test_stream_statuses_survive_state_serialization(monkeypatch, indices, status):
    spot = spot_context(indices=indices)
    monkeypatch.setattr(run_decisions, "collect_watch", lambda request: watch_snapshot())
    state = Propagator().create_initial_state("BTC-USD", "2024-01-02", "crypto")

    run_decisions.initialize_decisions(state, None, (spot,), current=False)

    full = CryptoPatternEvidence.model_validate_json(state["crypto_pattern_evidence"])
    compact = CryptoPatternSummary.model_validate_json(state["crypto_pattern_report"])
    assert [row.status for row in full.streams] == [
        status,
        "unavailable",
        "unavailable",
        "unavailable",
    ]
    assert [row.status for row in compact.streams] == [row.status for row in full.streams]


@pytest.mark.parametrize("supplied", [False, True])
def test_missing_spot_evidence_fails_closed(monkeypatch, supplied):
    failure = SourceFailure(kind="transport", detail="offline")
    spot = SpotPriceContext(
        asset="BTC",
        semantics="historical_finalized",
        analysis_date=date(2024, 1, 2),
        daily_failure=failure,
    )
    monkeypatch.setattr(run_decisions, "collect_watch", lambda request: watch_snapshot())
    state = Propagator().create_initial_state("BTC-USD", "2024-01-02", "crypto")

    run_decisions.initialize_decisions(state, None, (spot,) if supplied else (), current=False)

    stream = CryptoPatternEvidence.model_validate_json(state["crypto_pattern_evidence"]).streams[0]
    assert stream.status == "unavailable" and stream.events == ()
    assert stream.failure == (
        failure
        if supplied
        else SourceFailure(kind="unavailable", detail="Run-start Spot evidence unavailable")
    )


def test_checkpoint_resume_preserves_evidence_without_recompute(monkeypatch, graph):
    monkeypatch.setattr(trading_graph, "collect_spot_prices", lambda request: spot_context())
    monkeypatch.setattr(run_decisions, "collect_watch", lambda request: watch_snapshot())
    initial = graph.create_run_state("BTC-USD", "2024-01-02", "crypto")
    workflow = StateGraph(AgentState)
    workflow.add_node("finish", lambda state: {"final_rating": "Hold"})
    workflow.add_edge(START, "finish")
    workflow.add_edge("finish", END)
    compiled = workflow.compile(checkpointer=InMemorySaver(), interrupt_before=["finish"])
    config = {"configurable": {"thread_id": "pattern-resume"}}
    compiled.invoke(initial, config)

    def forbidden(*args):
        pytest.fail("Resume recomputed or fetched evidence")

    monkeypatch.setattr(trading_graph, "collect_spot_prices", forbidden)
    monkeypatch.setattr(run_decisions, "collect_watch", forbidden)
    monkeypatch.setattr(run_decisions, "classic_pattern_evidence", forbidden)
    graph._resuming = True
    resumed = compiled.invoke(graph.create_run_state("BTC-USD", "2024-01-02", "crypto"), config)
    run_decisions.initialize_decisions(resumed, None, (), current=False)

    assert resumed["final_rating"] == "Hold"
    assert resumed["crypto_pattern_evidence"] == initial["crypto_pattern_evidence"]
    assert resumed["crypto_pattern_report"] == initial["crypto_pattern_report"]
