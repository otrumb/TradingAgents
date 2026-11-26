from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest
import ta_patterns as tap

from tradingagents.dataflows.crypto_pattern_catalog import CallableDescriptor, load_pattern_catalog
from tradingagents.dataflows.crypto_pattern_inputs import owned_fixture
from tradingagents.dataflows.crypto_pattern_models import PatternStreamKey
from tradingagents.dataflows.crypto_pattern_runtime import (
    ClosedPatternStream,
    _events,
    classic_pattern_evidence,
    evaluate_stream,
    watch_stream,
)
from tradingagents.dataflows.crypto_spot_models import CompletedDailyCandle, SpotPriceContext
from tradingagents.dataflows.crypto_watch_models import WatchCandle, WatchSeries, WatchSnapshot

END = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def closed():
    data = owned_fixture()
    candles = tuple(
        WatchCandle(
            interval="1d",
            open_time=END - timedelta(days=160 - i),
            close_time=END - timedelta(days=159 - i, milliseconds=1),
            **{
                field: Decimal(str(getattr(data, field)[i]))
                for field in ("open", "high", "low", "close", "volume")
            },
        )
        for i in range(160)
    )
    return ClosedPatternStream(
        PatternStreamKey(asset="BTC", market="perpetual", timeframe="1d"),
        candles,
        END + timedelta(seconds=4),
        "/fapi/v1/klines",
    )


@pytest.mark.parametrize(
    "length,missing,status",
    [
        (160, True, "missing_bars"),
        (20, False, "insufficient_history"),
        (0, False, "unavailable"),
    ],
)
def test_ineligible_stream_never_calls_detector(closed, monkeypatch, length, missing, status):
    def forbidden(*args):
        pytest.fail("ineligible stream executed detector")

    monkeypatch.setattr(tap, "doji", forbidden)
    candles = closed.candles[:length]
    if missing:
        candles = candles[:80] + candles[81:]
    result = evaluate_stream(replace(closed, candles=candles))
    assert result.status == status
    assert result.events == ()


def test_signed_events_timing_anchor_and_alias_identity(closed):
    result = evaluate_stream(closed)
    assert result.status == "ok"
    assert result.events
    assert len({event.pattern_id for event in result.events}) == len(result.events)
    assert all(event.pattern_id != "chart.three_dc" for event in result.events)
    for event in result.events:
        assert event.signal_at == closed.candles[-1].close_time
        assert event.signal_at == event.confirmation_at
        assert event.observable_at == closed.observed_at >= event.signal_at
        assert event.anchor_status == "provider_not_exposed"
        assert event.pivot_at is event.extreme_at is None
        if event.pattern_id in {"candles.doji", "chart.inside_day", "chart.three_day_compression"}:
            assert event.direction == "non_directional"
        if event.pattern_id == "candles.black_candle":
            assert event.direction == "bearish" and event.raw_signal == -1


def test_detector_failure_keeps_sibling_events_and_streams(closed, monkeypatch):
    def broken(*args):
        raise RuntimeError("fixture dependency failure")

    monkeypatch.setattr(tap, "doji", broken)
    failed = evaluate_stream(closed)
    independent = evaluate_stream(replace(closed, candles=()))
    assert failed.status == "execution_error"
    assert failed.detector_errors == ("candles.doji:RuntimeError:fixture dependency failure",)
    assert any(event.pattern_id == "candles.white_candle" for event in failed.events)
    assert independent.status == "unavailable"


def test_same_bar_confirmed_over_forming_and_busted_failure(closed):
    entry = next(row for row in load_pattern_catalog().entries if row.id == "chart.inside_day")

    def signal(*args, **kwargs):
        result = np.zeros(len(args[0]), dtype=np.int8)
        result[-1] = 1
        return result

    descriptor = CallableDescriptor(replace(entry, modes=("confirmed", "forming")), signal)
    events = _events(descriptor, closed)
    assert len(events) == 1 and events[0].status == "confirmed"
    forming = _events(replace(descriptor, entry=replace(entry, modes=("forming",))), closed)
    assert forming[0].confirmation_at is None and forming[0].status == "forming"
    busted = _events(replace(descriptor, entry=replace(entry, name="busted_double_top")), closed)
    assert busted[0].status == "failed" and busted[0].confirmation_at is None


def test_existing_spot_and_watch_models_run_without_refetch(closed):
    candles = tuple(
        CompletedDailyCandle(
            **c.model_dump(exclude={"interval"}),
            asset="BTC",
            symbol="BTCUSDT",
            retrieved_at=closed.observed_at,
            server_time=END + timedelta(seconds=5),
            cutoff=END,
        )
        for c in closed.candles
    )
    spot = SpotPriceContext(asset="BTC", daily_candles=candles)
    snapshot = WatchSnapshot(
        asset="BTC",
        symbol="BTCUSDT",
        cutoff=END,
        observable_at=closed.observed_at,
        server_time=None,
        as_of=END,
        series=(
            WatchSeries(interval="1d", candles=closed.candles),
            WatchSeries(interval="4h"),
            WatchSeries(interval="1h"),
        ),
    )
    result = classic_pattern_evidence(spot, snapshot, closed.observed_at)
    assert [stream.status for stream in result.streams] == [
        "ok",
        "ok",
        "unavailable",
        "unavailable",
    ]
    assert result.catalog.total_count == 300
    assert all("candles" not in row for row in result.model_dump()["streams"])
    assert watch_stream("BTC", snapshot.series[0], closed.observed_at).candles == closed.candles


@pytest.mark.parametrize(
    "bar,pattern_id,direction,sign",
    [
        ((104, 105, 98, 100), "candles.black_candle", "bearish", -1),
        ((100, 104, 99, 103), "candles.white_candle", "bullish", 1),
        ((102, 103, 101, 102), "candles.doji", "non_directional", 1),
    ],
)
def test_latest_positive_sign_uses_catalog_direction(closed, bar, pattern_id, direction, sign):
    last = WatchCandle.model_validate(
        closed.candles[-1].model_dump()
        | {
            field: Decimal(value)
            for field, value in zip(("open", "high", "low", "close"), bar, strict=True)
        }
    )
    result = evaluate_stream(replace(closed, candles=closed.candles[:-1] + (last,)))
    event = next(event for event in result.events if event.pattern_id == pattern_id)
    assert (event.direction, event.raw_signal) == (direction, sign)
