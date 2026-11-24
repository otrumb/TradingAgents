import json
from datetime import UTC, date, datetime
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from pydantic import ValidationError

from tradingagents.dataflows.crypto_models import SnapshotRequest
from tradingagents.dataflows.crypto_spot import collect_spot_prices
from tradingagents.dataflows.crypto_spot_models import (
    IntradayPrice,
    SpotPriceRequest,
    utc_milliseconds,
)

OPEN = 1704067200000
CLOSE = OPEN + 86400000 - 1


def row(open_time=OPEN):
    return [
        open_time,
        "100",
        "120",
        "90",
        "110",
        "20",
        open_time + 86400000 - 1,
        "2000",
        10,
        "10",
        "1000",
        "0",
    ]


@pytest.fixture
def spot_wire(monkeypatch):
    payloads = {
        "/api/v3/time": {"serverTime": CLOSE + 5001},
        "/api/v3/klines": [row(), row(OPEN + 86400000)],
        "/api/v3/ticker/price": {"symbol": "BTCUSDT", "price": "115.12345678"},
    }

    def send(session, request, **kwargs):
        assert session.trust_env is False
        assert session.auth is None
        assert request.method == "GET"
        assert not {"authorization", "x-mbx-apikey"} & {h.lower() for h in request.headers}
        assert kwargs["allow_redirects"] is False
        assert 0 < kwargs["timeout"] <= 60
        url = urlsplit(request.url)
        assert url.scheme == "https"
        assert url.netloc == "data-api.binance.vision"
        assert url.path in payloads
        if url.path == "/api/v3/klines":
            query = parse_qs(url.query)
            assert query.pop("symbol")[0] in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
            end_time = query.pop("endTime", None)
            if end_time is not None:
                assert end_time == [str(CLOSE)]
            assert query == {
                "interval": ["1d"],
                "timeZone": ["0"],
                "limit": ["365"],
            }
        if url.path == "/api/v3/ticker/price" and isinstance(payloads[url.path], dict):
            requested = parse_qs(url.query)["symbol"][0]
            assert requested in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        payload = payloads[url.path]
        if isinstance(payload, requests.RequestException):
            raise payload
        response = requests.Response()
        response.status_code = 200
        if payload == "redirect":
            response.status_code = 302
            response.headers["Location"] = "https://example.com/private"
        if payload == "rate_limit":
            response.status_code = 429
        response.request = request
        response.url = request.url
        response._content = (
            b"not JSON" if payload == "invalid_json" else json.dumps(payload).encode()
        )
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return payloads


@pytest.mark.parametrize("lag,count", [(0, 0), (4999, 0), (5000, 0), (5001, 1)])
def test_strict_finality_lag_excludes_boundary_and_open_candle(spot_wire, lag, count):
    spot_wire["/api/v3/time"] = {"serverTime": CLOSE + lag}

    result = collect_spot_prices(SnapshotRequest(asset="BTC"))

    assert len(result.daily_candles) == count
    assert result.intraday.price == Decimal("115.12345678")
    if count:
        assert result.daily_candles[-1].close_time == utc_milliseconds(CLOSE)
        assert result.daily_candles[-1].close == Decimal("110")
    else:
        assert result.daily_failure.kind == "parse"


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
def test_symbol_mapping_and_serialized_provenance(spot_wire, asset):
    spot_wire["/api/v3/ticker/price"]["symbol"] = f"{asset}USDT"

    result = collect_spot_prices(SnapshotRequest(asset=asset))

    candle = result.daily_candles[-1]
    assert candle.symbol == f"{asset}USDT"
    assert candle.venue == "Binance Spot"
    assert candle.source == "https://data-api.binance.vision"
    serialized = json.loads(result.model_dump_json())
    for observation, endpoint, data_source in [
        (serialized["daily_candles"][0], "/api/v3/klines", "Database"),
        (serialized["intraday"], "/api/v3/ticker/price", "Memory"),
    ]:
        assert observation["provider"] == "binance"
        assert observation["market"] == "spot"
        assert observation["base_url"] == "https://data-api.binance.vision"
        assert observation["endpoint"] == endpoint
        assert observation["data_source"] == data_source
    assert candle.server_time == utc_milliseconds(CLOSE + 5001)
    assert candle.cutoff == utc_milliseconds(CLOSE + 1)
    assert candle.open_time == datetime(2024, 1, 1, tzinfo=UTC)
    assert candle.interval == "1d" and candle.timezone == "0"
    assert candle.finality == "closed"
    assert candle.retrieved_at.utcoffset().total_seconds() == 0
    assert result.intraday.observed_at is None
    assert result.intraday.finality == "intraday_not_daily_close"
    assert json.loads(result.model_dump_json())["daily_candles"][0]["close_time"].endswith("Z")
    with pytest.raises(ValidationError):
        candle.close = Decimal("999")


@pytest.mark.parametrize(
    "path,payload,failed",
    [
        ("/api/v3/time", {"serverTime": True}, "daily"),
        ("/api/v3/klines", [["bad"]], "daily"),
        ("/api/v3/klines", "invalid_json", "daily"),
        ("/api/v3/klines", "redirect", "daily"),
        ("/api/v3/klines", "rate_limit", "daily"),
        ("/api/v3/klines", [row(), row()], "daily"),
        ("/api/v3/klines", requests.Timeout("offline"), "daily"),
        ("/api/v3/ticker/price", {"symbol": "ETHUSDT", "price": "1"}, "intraday"),
        ("/api/v3/ticker/price", {"symbol": "BTCUSDT", "price": "NaN"}, "intraday"),
    ],
)
def test_endpoint_failure_preserves_other_evidence(spot_wire, path, payload, failed):
    spot_wire[path] = payload

    result = collect_spot_prices(SnapshotRequest(asset="BTC"))

    if failed == "daily":
        assert result.daily_candles == ()
        assert result.daily_failure is not None
        assert result.intraday.price == Decimal("115.12345678")
    else:
        assert result.intraday is None
        assert result.intraday_failure.kind == "parse"
        assert result.daily_candles[-1].close == Decimal("110")


def test_intraday_cannot_be_constructed_from_daily_candle(spot_wire):
    result = collect_spot_prices(SnapshotRequest(asset="BTC"))

    with pytest.raises(ValidationError):
        IntradayPrice.model_validate(result.daily_candles[-1].model_dump())


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
def test_historical_day_uses_end_time_and_filters_future_rows(spot_wire, asset, monkeypatch):
    del spot_wire["/api/v3/time"]
    del spot_wire["/api/v3/ticker/price"]
    boundary = row()
    boundary[6] = CLOSE + 1
    spot_wire["/api/v3/klines"] = [row(OPEN + 86400000), boundary, row()]
    original_send = requests.Session.send

    def historical_send(session, request, **kwargs):
        assert parse_qs(urlsplit(request.url).query)["endTime"] == ["1704153599999"]
        return original_send(session, request, **kwargs)

    monkeypatch.setattr(requests.Session, "send", historical_send)

    result = collect_spot_prices(SpotPriceRequest(asset=asset, analysis_date="2024-01-01"))

    assert result.daily_failure is None
    assert result.semantics == "historical_finalized"
    assert result.analysis_date == date(2024, 1, 1)
    assert result.intraday is None and result.intraday_failure is None
    assert [(c.symbol, c.close_time, c.close) for c in result.daily_candles] == [
        (f"{asset}USDT", datetime(2024, 1, 1, 23, 59, 59, 999000, tzinfo=UTC), Decimal("110"))
    ]
    candle = result.daily_candles[0]
    assert candle.cutoff == datetime(2024, 1, 2, tzinfo=UTC)
    assert candle.server_time is None
    assert candle.finality_lag_ms == 0
    assert candle.data_source == "Database"
    summary = result.summary()
    assert summary.analysis_date == date(2024, 1, 1)
    assert summary.semantics == "historical_finalized"
    assert summary.latest_finalized == candle


def test_historical_failure_never_falls_back_to_current_evidence(spot_wire):
    del spot_wire["/api/v3/time"]
    del spot_wire["/api/v3/ticker/price"]
    spot_wire["/api/v3/klines"] = "rate_limit"

    result = collect_spot_prices(SpotPriceRequest(asset="BTC", analysis_date="2024-01-01"))

    assert result.daily_candles == ()
    assert result.daily_failure.kind == "http"
    assert result.intraday is None and result.intraday_failure is None
