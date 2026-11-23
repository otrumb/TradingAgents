import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.crypto_exchange import (
    parse_book,
    parse_dvol,
    parse_ls,
    parse_oi,
    parse_options,
    parse_premium,
)
from tradingagents.dataflows.crypto_models import (
    CryptoContext,
    Metric,
    SnapshotRequest,
    SourceSnapshot,
)


@pytest.mark.parametrize(
    "rate,expected", [(None, None), ("0", Decimal(0)), ("-0.001", Decimal("-0.001"))]
)
def test_funding_preserves_null_zero_and_signed_fraction(
    rate: str | None, expected: Decimal | None
) -> None:
    raw = json.dumps(
        {
            "symbol": "ETHUSDT",
            "time": 1704067200000,
            "markPrice": "2300.1",
            "indexPrice": "2300",
            "lastFundingRate": rate,
        }
    )
    metrics = parse_premium(raw)
    assert metrics[2].value == expected
    assert metrics[0].observed_at == datetime(2024, 1, 1, tzinfo=UTC)
    assert metrics[0].coverage == "ETHUSDT"


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"time": 1}',
        "null",
        '{"symbol":"ETHUSDT","time":1,"markPrice":"NaN","indexPrice":0,"lastFundingRate":0}',
    ],
)
def test_malformed_premium_is_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        parse_premium(raw)


def test_long_short_selects_latest_timestamp_not_response_order() -> None:
    rows = [
        {
            "symbol": "BTCUSDT",
            "timestamp": ts,
            "longShortRatio": ratio,
            "longAccount": "0.6",
            "shortAccount": "0.4",
        }
        for ts, ratio in [(2000, "1.5"), (1000, "2")]
    ]
    result = parse_ls(json.dumps(rows))
    assert result[0].value == Decimal("1.5")
    assert result[0].resolution == "5m"


@pytest.mark.parametrize(
    "age,value,expected",
    [
        (60, Decimal(0), "fresh"),
        (61, Decimal(0), "stale"),
        (-1, Decimal(1), "future"),
        (0, None, "unknown"),
    ],
)
def test_freshness_uses_observation_not_fetch_time(
    age: int, value: Decimal | None, expected: str
) -> None:
    fetched = datetime(2024, 1, 1, tzinfo=UTC)
    metric = Metric(
        name="funding",
        value=value,
        unit="fraction",
        observed_at=fetched - timedelta(seconds=age),
        resolution="instant",
        coverage="ETHUSDT",
    )
    assert metric.freshness(fetched, timedelta(seconds=60)) == expected


def test_untimestamped_value_has_unknown_freshness() -> None:
    metric = Metric(
        name="tvl",
        value=Decimal(0),
        unit="USD",
        observed_at=None,
        resolution="latest",
        coverage="Ethereum",
    )
    assert metric.freshness(datetime(2024, 1, 1, tzinfo=UTC), timedelta(days=1)) == "unknown"


def test_options_keep_instrument_units_and_null_iv() -> None:
    raw = '{"result":[{"instrument_name":"ETH-29MAR24-3000-C","creation_timestamp":1704067200000,"open_interest":12,"volume":0,"mark_iv":null}]}'
    result = parse_options(raw)
    assert [(m.value, m.unit) for m in result] == [
        (Decimal(12), "base asset"),
        (Decimal(0), "base asset"),
        (None, "percent annualized"),
    ]
    assert result[2].coverage == "ETH-29MAR24-3000-C"


def test_wrong_venue_cannot_enter_context() -> None:
    with pytest.raises(ValidationError, match="venue"):
        SourceSnapshot(
            source="binance",
            venue="chain aggregate",
            endpoint="https://example.com",
            fetched_at=datetime(2024, 1, 1, tzinfo=UTC),
            observed_at=None,
            resolution="instant",
            coverage="ETHUSDT",
        )


def test_request_rejects_historical_and_unsupported_assets() -> None:
    for raw in ['{"asset":"ETH","as_of":"2024-01-01"}', '{"asset":"XRP"}']:
        with pytest.raises(ValidationError):
            SnapshotRequest.model_validate_json(raw)


@pytest.mark.parametrize("asset", ["ETH", "BTC", "SOL"])
def test_supported_assets_round_trip_through_context(asset: str) -> None:
    request = SnapshotRequest.model_validate({"asset": asset})
    context = CryptoContext(asset=request.asset, snapshots=())
    assert CryptoContext.model_validate_json(context.render()).asset == asset


def test_context_is_frozen_and_rendering_is_canonical_json() -> None:
    context = CryptoContext(asset="ETH", snapshots=())
    with pytest.raises(ValidationError, match="frozen"):
        context.asset = "BTC"
    rendered = context.render()
    assert json.loads(rendered)["semantics"] == "current_snapshot"
    assert rendered == json.dumps(json.loads(rendered), sort_keys=True, indent=2)


def test_open_interest_and_book_parse_decimal_units() -> None:
    oi = parse_oi('{"symbol":"BTCUSDT","time":1704067200000,"openInterest":"1.25"}')
    book = parse_book(
        '{"symbol":"BTCUSDT","time":1704067200000,"bidPrice":"2300.1","bidQty":"0","askPrice":"2300.2","askQty":null}'
    )
    assert oi[0].value == Decimal("1.25")
    assert oi[0].unit == "base asset"
    assert [m.value for m in book] == [Decimal("2300.1"), Decimal(0), Decimal("2300.2"), None]


def test_dvol_uses_latest_candle_close() -> None:
    result = parse_dvol('{"result":{"data":[[2000,11,12,10,13],[1000,21,22,20,23]]}}')
    assert result[0].value == Decimal(13)
    assert result[0].observed_at == datetime(1970, 1, 1, 0, 0, 2, tzinfo=UTC)
