from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.agents import market_tools
from tradingagents.dataflows import crypto_market
from tradingagents.dataflows.crypto_spot_models import CompletedDailyCandle, SpotPriceContext


@pytest.fixture
def history() -> SpotPriceContext:
    server = datetime(2026, 6, 10, 0, 0, 5, tzinfo=UTC)
    start = datetime(2025, 10, 1, tzinfo=UTC)
    return SpotPriceContext(
        asset="BTC",
        daily_candles=tuple(
            CompletedDailyCandle(
                asset="BTC",
                symbol="BTCUSDT",
                retrieved_at=server,
                server_time=server,
                cutoff=server - timedelta(seconds=5),
                open_time=start + timedelta(days=i),
                close_time=start + timedelta(days=i + 1, milliseconds=-1),
                open=100 + i,
                high=103 + i,
                low=99 + i,
                close=101 + i,
                volume=1000 + i,
            )
            for i in range(220)
        ),
    )


@pytest.mark.parametrize(
    "indicator,required",
    [
        ("close_10_ema", 10),
        ("close_50_sma", 50),
        ("close_200_sma", 200),
        ("rsi", 14),
        ("boll", 20),
        ("boll_ub", 20),
        ("boll_lb", 20),
        ("macd", 26),
        ("macds", 34),
        ("macdh", 34),
        ("atr", 14),
        ("vwma", 14),
    ],
)
@pytest.mark.parametrize("offset", [-1, 0])
def test_indicator_readiness_uses_selected_finalized_row_threshold(
    history, indicator, required, offset
):
    count = required + offset
    as_of = history.daily_candles[count - 1].open_time.date().isoformat()

    result = market_tools.get_indicators.func(
        "BTC-USD",
        indicator,
        as_of,
        asset_type="crypto",
        crypto_market_data=history.model_dump_json(),
    )

    if offset < 0:
        assert result.startswith("NO_DATA_AVAILABLE")
        assert f"{indicator} requires {required} finalized rows; available {count}" in result
    else:
        assert f"rows: {count}\n" in result
        assert result.splitlines()[-1].startswith(f"{as_of}: ")
        assert "N/A" not in result.splitlines()[-1]
        assert result.split("\n\n")[-1].splitlines()[-2].endswith(": N/A")


def test_snapshot_keeps_prices_and_ready_indicators_when_history_is_short(history):
    as_of = history.daily_candles[13].open_time.date().isoformat()

    result = crypto_market.crypto_verified_snapshot(
        history.model_dump_json(),
        "BTC-USD",
        as_of,
        30,
        ("close_10_ema", "rsi", "close_200_sma", "macds"),
    )

    assert "- Close: 114" in result
    assert "- rsi: 100" in result
    assert "- close_10_ema: N/A" not in result
    assert (
        "- close_200_sma: N/A (close_200_sma requires 200 finalized rows; available 14)" in result
    )
    assert "- macds: N/A (macds requires 34 finalized rows; available 14)" in result


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_latest_is_unavailable_on_both_surfaces(history, monkeypatch, value):
    real_wrap = crypto_market.wrap

    def nonfinite_wrap(frame):
        stock = real_wrap(frame)
        stock["rsi"] = value
        return stock

    monkeypatch.setattr(crypto_market, "wrap", nonfinite_wrap)
    payload = history.model_dump_json()
    direct = market_tools.get_indicators.func(
        "BTC-USD",
        "rsi",
        "2026-06-01",
        asset_type="crypto",
        crypto_market_data=payload,
    )
    snapshot = crypto_market.crypto_verified_snapshot(
        payload, "BTC-USD", "2026-06-01", 30, ("rsi",)
    )

    assert direct.startswith("NO_DATA_AVAILABLE")
    assert "rsi latest value is nonfinite" in direct
    assert "- rsi: N/A (rsi latest value is nonfinite)" in snapshot
    assert "- Close: 320" in snapshot


@pytest.mark.parametrize("surface", ["direct", "snapshot"])
@pytest.mark.parametrize("removed,missing", [(30, "none"), (5, "2025-10-06")])
def test_as_of_coverage_excludes_future_gaps_and_keeps_selected_gaps(
    history, surface, removed, missing
):
    evidence = SpotPriceContext(
        asset="BTC",
        daily_candles=tuple(c for i, c in enumerate(history.daily_candles) if i != removed),
    )
    payload = evidence.model_dump_json()
    if surface == "direct":
        result = crypto_market.crypto_indicators(
            payload, "BTC-USD", "close_10_ema", "2025-10-20", 30
        )
    else:
        result = crypto_market.crypto_verified_snapshot(payload, "BTC-USD", "2025-10-20", 30)

    facts = dict(line.split(": ", 1) for line in result.splitlines() if ": " in line)
    assert facts["rows"] == ("20" if removed == 30 else "19")
    assert facts["missing_dates"] == missing
    assert facts["missing_day_count"] == ("0" if removed == 30 else "1")
    assert facts["coverage"] == ("contiguous" if removed == 30 else "warning")
    if removed == 5:
        assert not any(
            line.startswith(("2025-10-06:", "- 2025-10-06:")) for line in result.splitlines()
        )
