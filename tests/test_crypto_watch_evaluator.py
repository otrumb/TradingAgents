import json
from datetime import timedelta

import pytest

from tests.crypto_watch_fixtures import CUTOFF, watch_wire as watch_wire, wire_rows
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch, render_watch
from tradingagents.dataflows.crypto_watch_models import DURATIONS, WatchRequest, WatchSnapshot


@pytest.mark.parametrize("direction,signal", [(1, "long"), (-1, "short")])
def test_aligned_signals_and_markdown_fields(watch_wire, direction, signal):
    watch_wire["rows"] = {i: wire_rows(i, direction) for i in DURATIONS}
    snapshot = collect_watch(WatchRequest(asset="BTC", timeout_seconds=7))
    decision = evaluate_watch(snapshot)
    assert decision.bias == decision.setup == decision.trigger == signal
    assert decision.action == f"consider_{signal}"
    text = render_watch(decision)
    metadata = (line[2:].split(": ", 1) for line in text.splitlines() if line.startswith("- "))
    expected_values = (
        f"BTC BTCUSDT https://fapi.binance.com 2024-05-01T00:00:00+00:00 consider_{signal} none"
    )
    keys = ("Asset", "Symbol", "Source", "as_of", "Decision", "Wait reason")
    assert dict(metadata) == dict(zip(keys, expected_values.split(), strict=True))
    table = [
        tuple(cell.strip() for cell in line.strip("|").split("|"))
        for line in text.splitlines()
        if line.startswith("| ")
    ][2:]
    stages = ("1d Bias", "4h Setup", "1h Trigger")
    assert table == [(stage, signal, "2024-04-30T23:59:59.999000+00:00", "ok") for stage in stages]
    assert json.loads(decision.model_dump_json())["action"] == f"consider_{signal}"
    assert render_watch(evaluate_watch(snapshot)) == render_watch(decision)


@pytest.mark.parametrize(
    "case,reason",
    zip(
        ["empty", "short", "stale", "gap", "conflict", "trigger", "setup", "neutral"],
        [
            "unavailable",
            "insufficient_history",
            "stale",
            "insufficient_history",
            "timeframe_conflict",
            "missing_trigger",
            "missing_setup",
            "neutral_bias",
        ],
        strict=True,
    ),
)
def test_wait_conditions_block_direction(watch_wire, case, reason):
    rows = watch_wire["rows"]
    if case == "empty":
        rows["1d"] = []
    elif case == "short":
        rows["1d"] = rows["1d"][-54:]
    elif case == "stale":
        for row in rows["1h"]:
            row[0] -= 3 * 3600000
            row[6] -= 3 * 3600000
    elif case == "gap":
        del rows["1d"][2]
    elif case == "conflict":
        rows["4h"] = wire_rows("4h", -1)
    elif case == "setup":
        rows["4h"][-2][3] = rows["4h"][-2][4]
    elif case == "neutral":
        for row in rows["1d"]:
            row[1:5] = ["200", "201", "199", "200"]
    else:
        rows["1h"][-1][4] = rows["1h"][-2][4]
        rows["1h"][-1][3] = rows["1h"][-2][3]
    decision = evaluate_watch(collect_watch(WatchRequest(asset="BTC", timeout_seconds=7)))
    assert decision.action == "wait"
    assert decision.reasons == (reason,)


@pytest.mark.parametrize("interval,hours", [("1d", 48), ("4h", 8), ("1h", 2)])
@pytest.mark.parametrize("excess,reason", [(0, ()), (1, ("stale",))])
def test_freshness_exact_threshold(watch_wire, interval, hours, excess, reason):
    snapshot = collect_watch(WatchRequest(asset="BTC", as_of=CUTOFF, timeout_seconds=7))
    data = snapshot.model_dump()
    latest = next(s for s in snapshot.series if s.interval == interval).candles[-1].close_time
    data["cutoff"] = data["as_of"] = latest + timedelta(hours=hours, microseconds=excess)
    for other in data["series"]:
        shift = timedelta(hours=hours) - DURATIONS[other["interval"]]
        if other["interval"] != interval and shift > timedelta(0):
            for candle in other["candles"]:
                candle["open_time"] += shift
                candle["close_time"] += shift
    decision = evaluate_watch(WatchSnapshot.model_validate(data))
    assert decision.reasons == reason


@pytest.mark.parametrize("interval", ("4h", "1h"))
@pytest.mark.parametrize("count,reason", [(20, ("insufficient_history",)), (21, ())])
def test_minimum_history(watch_wire, interval, count, reason):
    watch_wire["rows"][interval] = watch_wire["rows"][interval][-count:]
    decision = evaluate_watch(collect_watch(WatchRequest(asset="BTC", timeout_seconds=7)))
    assert decision.reasons == reason


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "mutation,reason",
    zip(
        ["outside_three", "break_equal", "ema_equal", "trigger_equal", "opposite_trigger"],
        [
            "missing_setup",
            "missing_setup",
            "missing_setup",
            "missing_trigger",
            "timeframe_conflict",
        ],
        strict=True,
    ),
)
def test_strict_setup_and_trigger_rules(watch_wire, direction, mutation, reason):
    watch_wire["rows"] = {i: wire_rows(i, direction) for i in DURATIONS}
    four, hour = watch_wire["rows"]["4h"], watch_wire["rows"]["1h"]
    match mutation:
        case "outside_three":
            four[-2][3 if direction > 0 else 2] = four[-2][4]
            four[-5][3 if direction > 0 else 2] = "1" if direction > 0 else "400"
        case "break_equal":
            four[-2][2 if direction > 0 else 3] = four[-1][4]
        case "ema_equal":
            for row in four:
                row[1:5] = ["200", "201", "199", "200"]
        case "trigger_equal":
            hour[-2][2 if direction > 0 else 3] = hour[-1][4]
        case "opposite_trigger":
            watch_wire["rows"]["1h"] = wire_rows("1h", -direction)
    decision = evaluate_watch(collect_watch(WatchRequest(asset="BTC", timeout_seconds=7)))
    assert decision.reasons == (reason,)
