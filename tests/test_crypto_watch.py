from datetime import timedelta
from decimal import Decimal

import pytest
import requests
from pydantic import ValidationError

from tests.crypto_watch_fixtures import CUTOFF, MS, watch_wire as watch_wire
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import (
    DURATIONS,
    EPOCH,
    WatchDecision,
    WatchRequest,
    WatchSnapshot,
)


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
def test_current_contract_common_exchange_cutoff_and_decimal(watch_wire, asset):
    watch_wire["symbol"] = f"{asset}USDT"
    snapshot = collect_watch(WatchRequest(asset=asset, timeout_seconds=7))

    assert [path for path, _ in watch_wire["calls"]] == ["/fapi/v1/time"] + ["/fapi/v1/klines"] * 3
    assert snapshot.cutoff == CUTOFF
    assert snapshot.server_time == CUTOFF + timedelta(seconds=5)
    assert snapshot.market == "USD-M perpetual"
    assert snapshot.symbol == f"{asset}USDT"
    assert tuple(s.interval for s in snapshot.series) == ("1d", "4h", "1h")
    assert all(s.candles[-1].close_time == CUTOFF - MS for s in snapshot.series)
    assert snapshot.series[0].candles[-1].close == Decimal("308")
    with pytest.raises(ValidationError):
        snapshot.series[0].candles[-1].close = Decimal("1")


@pytest.mark.parametrize("offset,removed", [(0, 1), (1, 0), (5000, 0)])
def test_strict_finality_at_boundary_and_open_rows(watch_wire, offset, removed):
    cutoff = CUTOFF - MS + offset * MS
    watch_wire["end"] = (cutoff - EPOCH) // MS - 1
    watch_wire["clock"] = {"serverTime": (cutoff - EPOCH) // MS + 5000}
    for interval, rows in watch_wire["rows"].items():
        future = list(rows[-1])
        future[0] += DURATIONS[interval] // MS
        future[6] += DURATIONS[interval] // MS
        rows.append(future)

    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))

    assert tuple(len(s.candles) for s in snapshot.series) == tuple(
        n - removed for n in (55, 24, 24)
    )
    assert all(c.close_time < cutoff for s in snapshot.series for c in s.candles)


def test_historical_uses_end_time_never_current_clock(watch_wire):
    watch_wire["clock"] = AssertionError("historical clock forbidden")
    snapshot = collect_watch(WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7))

    assert len(watch_wire["calls"]) == 3
    assert all(path == "/fapi/v1/klines" for path, _ in watch_wire["calls"])
    assert snapshot.server_time is None and snapshot.cutoff == snapshot.as_of == CUTOFF
    assert evaluate_watch(snapshot).action == "consider_long"


@pytest.mark.parametrize(
    "payload,kind",
    zip(
        ("redirect", "429", requests.Timeout("offline"), [["bad"]]),
        ("parse", "http", "transport", "parse"),
        strict=True,
    ),
)
def test_failure_local_and_waits(watch_wire, payload, kind):
    watch_wire["rows"]["4h"] = payload
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))

    assert snapshot.series[1].failure.kind == kind
    assert snapshot.series[1].candles == ()
    assert (len(snapshot.series[0].candles), len(snapshot.series[2].candles)) == (55, 24)
    assert evaluate_watch(snapshot).reasons == ("unavailable",)


def test_failed_clock_never_collects_candles(watch_wire):
    watch_wire["clock"] = requests.Timeout("offline")
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    assert len(watch_wire["calls"]) == 1
    assert snapshot.cutoff is None
    assert evaluate_watch(snapshot).reasons == ("unavailable",)


def test_model_rejects_scope_and_decision_forgery(watch_wire):
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    for updates in ({"symbol": "ETHUSDT"}, {"cutoff": CUTOFF - MS}, {"series": ()}):
        with pytest.raises(ValidationError):
            WatchSnapshot.model_validate(snapshot.model_dump() | updates)
    data = {
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "as_of": CUTOFF,
        "cutoff": CUTOFF,
        "action": "consider_long",
    }
    with pytest.raises(ValidationError):
        WatchDecision(**data)


@pytest.mark.parametrize("column,value", [(0, 1), (2, "1"), (4, "NaN"), ("duplicate", None)])
def test_malformed_evidence_cannot_signal(watch_wire, column, value):
    rows = watch_wire["rows"]["1d"]
    if column == "duplicate":
        rows.append(list(rows[0]))
    else:
        rows[0][column] = rows[0][column] + value if column == 0 else value
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    assert snapshot.series[0].failure.kind == "parse"
    assert evaluate_watch(snapshot).action == "wait"
