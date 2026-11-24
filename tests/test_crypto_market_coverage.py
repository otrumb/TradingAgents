import json
from datetime import UTC, date, datetime, timedelta

import pytest

from tradingagents.agents import market_tools
from tradingagents.dataflows import crypto_market
from tradingagents.dataflows.crypto_spot_models import CompletedDailyCandle, SpotPriceContext


@pytest.fixture
def context() -> SpotPriceContext:
    server_time = datetime(2026, 6, 10, 0, 0, 5, tzinfo=UTC)
    start = datetime(2026, 6, 1, tzinfo=UTC)
    return SpotPriceContext(
        asset="BTC",
        daily_candles=tuple(
            CompletedDailyCandle(
                asset="BTC",
                symbol="BTCUSDT",
                retrieved_at=server_time,
                server_time=server_time,
                cutoff=server_time - timedelta(seconds=5),
                open_time=start + timedelta(days=index),
                close_time=start + timedelta(days=index + 1, milliseconds=-1),
                open=100,
                high=102,
                low=99,
                close=101,
                volume=1000,
            )
            for index in range(7)
        ),
    )


@pytest.fixture(
    params=[
        (market_tools.get_stock_data.func, ("2026-05-01", "2026-06-10")),
        (market_tools.get_indicators.func, ("rsi", "2026-06-10")),
        (market_tools.get_verified_market_snapshot.func, ("2026-06-10",)),
    ]
)
def surface(request):
    return request.param


@pytest.mark.parametrize(
    "kind", ["empty", "malformed", "schema", "no_candles", "asset", "symbol", "request"]
)
def test_invalid_evidence_returns_no_data(context, surface, kind):
    payloads = {
        "empty": "",
        "malformed": "{",
        "schema": "{}",
        "no_candles": SpotPriceContext(asset="BTC").model_dump_json(),
        "asset": json.dumps(context.model_dump(mode="json") | {"asset": "ETH"}),
        "symbol": context.model_dump_json().replace("BTCUSDT", "ETHUSDT"),
        "request": context.model_dump_json(),
    }
    render, args = surface

    result = render(
        "ETH-USD" if kind == "request" else "BTC-USD",
        *args,
        asset_type="crypto",
        crypto_market_data=payloads[kind],
    )

    assert result.startswith("NO_DATA_AVAILABLE")


@pytest.mark.parametrize(
    "removed,missing",
    [
        ((), ()),
        ((2,), (date(2026, 6, 3),)),
        ((1, 2, 5), (date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 6))),
        ((0, 6), ()),
    ],
)
def test_coverage_counts_only_interior_missing_utc_days(context, removed, missing):
    evidence = SpotPriceContext(
        asset="BTC",
        daily_candles=tuple(c for i, c in enumerate(context.daily_candles) if i not in removed),
    )

    coverage = crypto_market.daily_coverage(evidence)

    assert coverage.missing_day_count == len(missing)
    assert coverage.missing_dates == missing


def test_coverage_bounds_explicit_dates(context):
    first, last = context.daily_candles[0], context.daily_candles[-1]
    first = CompletedDailyCandle.model_validate(
        first.model_dump()
        | {
            "open_time": datetime(2026, 5, 1, tzinfo=UTC),
            "close_time": datetime(2026, 5, 2, tzinfo=UTC) - timedelta(milliseconds=1),
        }
    )
    evidence = SpotPriceContext(asset="BTC", daily_candles=(first, last))

    result = crypto_market.provenance(evidence)

    facts = dict(line.split(": ", 1) for line in result.splitlines())
    assert facts["missing_day_count"] == "36"
    assert facts["missing_dates"] == ", ".join(f"2026-05-{day:02}" for day in range(2, 12))
    assert facts["missing_dates_omitted"] == "26"


def test_gap_warning_keeps_usable_rows_on_each_surface(context, surface):
    evidence = SpotPriceContext(
        asset="BTC",
        daily_candles=context.daily_candles[:2] + context.daily_candles[3:],
    )
    render, args = surface

    result = render(
        "BTC-USD", *args, asset_type="crypto", crypto_market_data=evidence.model_dump_json()
    )

    if render is market_tools.get_indicators.func:
        assert result.startswith("NO_DATA_AVAILABLE")
        assert "requires 14 finalized rows; available 6" in result
        return
    facts = dict(line.split(": ", 1) for line in result.splitlines() if ": " in line)
    assert facts["rows"] == "6"
    assert facts["missing_day_count"] == "1"
    assert facts["missing_dates"] == "2026-06-03"
    assert facts["coverage"] == "warning"
    assert "2026-06-07" in result
    assert "NO_DATA_AVAILABLE" not in result


def test_unsupported_indicator_remains_rejected_with_empty_evidence():
    result = market_tools.get_indicators.func(
        "BTC-USD",
        "not_an_indicator",
        "2026-06-10",
        asset_type="crypto",
        crypto_market_data="",
    )

    assert result.startswith("Unsupported crypto indicator:")


def test_contiguous_truncated_history_has_no_warning(context):
    result = crypto_market.provenance(context)

    facts = dict(line.split(": ", 1) for line in result.splitlines())
    assert facts["coverage"] == "contiguous"
    assert facts["missing_day_count"] == "0"
    assert facts["missing_dates"] == "none"
    assert facts["missing_dates_omitted"] == "0"


@pytest.mark.parametrize("symbol", ["BTC-USD", "BTCUSDT", "btc-usdt"])
def test_matching_symbol_aliases_keep_evidence(context, symbol):
    parsed, frame = crypto_market.parse_crypto_market_data(context.model_dump_json(), symbol)

    assert parsed == context
    assert len(frame) == 7


def test_crypto_indicator_programming_errors_propagate(context, monkeypatch):
    context = SpotPriceContext(
        asset="BTC",
        daily_candles=tuple(
            CompletedDailyCandle.model_validate(
                c.model_dump()
                | {
                    "open_time": c.open_time - timedelta(days=7),
                    "close_time": c.close_time - timedelta(days=7),
                }
            )
            for c in context.daily_candles
        )
        + context.daily_candles,
    )

    def broken_wrap(frame):
        raise ValueError("calculation bug")

    monkeypatch.setattr(crypto_market, "wrap", broken_wrap)

    with pytest.raises(ValueError, match="calculation bug"):
        market_tools.get_indicators.func(
            "BTC-USD",
            "rsi",
            "2026-06-10",
            asset_type="crypto",
            crypto_market_data=context.model_dump_json(),
        )
