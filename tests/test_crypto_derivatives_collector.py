import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from pydantic import ValidationError

from tradingagents.dataflows import crypto_derivatives as collector
from tradingagents.dataflows.crypto_derivatives_models import DerivativesContractError
from tradingagents.dataflows.crypto_models import SnapshotRequest

NOW = datetime(2026, 11, 24, 12, tzinfo=UTC)


def funding(symbol="BTCUSDT", hours=1, rate="0"):
    return {
        "symbol": symbol,
        "fundingTime": int((NOW - timedelta(hours=hours)).timestamp() * 1000),
        "fundingRate": rate,
        "markPrice": "100",
        "rateType": "Regular",
    }


def interest(symbol="BTCUSDT", hours=0):
    return {
        "symbol": symbol,
        "timestamp": int((NOW - timedelta(hours=hours)).timestamp() * 1000),
        "sumOpenInterest": "100",
        "sumOpenInterestValue": "10000",
        "CMCCirculatingSupply": "100000",
    }


@pytest.mark.parametrize(
    "parse,row", [(collector.parse_funding, funding), (collector.parse_interest, interest)]
)
def test_parsers_dedupe_identical_rows_and_reject_disorder_conflicts_and_wrong_symbol(parse, row):
    assert len(parse(json.dumps([row(hours=2), row(hours=2), row(hours=1)]), "BTCUSDT")) == 2
    for payload in ([row(hours=1), row(hours=2)], [row(symbol="ETHUSDT")]):
        with pytest.raises(DerivativesContractError):
            parse(json.dumps(payload), "BTCUSDT")
    changed = row()
    changed["fundingRate" if parse is collector.parse_funding else "sumOpenInterest"] = "10"
    with pytest.raises(DerivativesContractError):
        parse(json.dumps([row(), changed]), "BTCUSDT")


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '[{"symbol":"BTCUSDT","fundingTime":true,"fundingRate":"0"}]',
        '[{"symbol":"BTCUSDT","fundingTime":1,"fundingRate":0}]',
        '[{"symbol":"BTCUSDT","fundingTime":1,"fundingRate":"NaN"}]',
    ],
)
def test_funding_strict_wire_rejects_invalid_rows(raw):
    with pytest.raises(ValidationError):
        collector.parse_funding(raw, "BTCUSDT")


@pytest.fixture
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return NOW

    monkeypatch.setattr(collector, "datetime", Clock)


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("failed", [None, "funding", "interest"])
def test_public_get_contract_and_independent_failures(monkeypatch, fixed_clock, asset, failed):
    paths = []

    def send(session, request, **kwargs):
        assert session.trust_env is False
        assert request.method == "GET"
        assert not {"Authorization", "X-MBX-APIKEY"} & request.headers.keys()
        assert kwargs["timeout"] == 3
        url = urlsplit(request.url)
        assert url.scheme == "https" and url.netloc == "fapi.binance.com"
        params = parse_qs(url.query)
        symbol = f"{asset}USDT"
        assert params["symbol"] == [symbol]
        assert params["endTime"] == [str(int(NOW.timestamp() * 1000))]
        response = requests.Response()
        response.request = request
        response.status_code = 200
        response.url = request.url
        if url.path == "/fapi/v1/fundingRate":
            paths.append("funding")
            assert params == {
                "symbol": [symbol],
                "startTime": [str(int((NOW - timedelta(days=30)).timestamp() * 1000))],
                "endTime": params["endTime"],
                "limit": ["1000"],
            }
            payload = [funding(symbol)]
        elif url.path == "/futures/data/openInterestHist":
            paths.append("interest")
            assert params == {
                "symbol": [symbol],
                "startTime": [str(int((NOW - timedelta(hours=26)).timestamp() * 1000))],
                "endTime": params["endTime"],
                "period": ["1h"],
                "limit": ["500"],
            }
            payload = [
                interest(symbol, 24),
                interest(symbol, 4),
                interest(symbol, 1),
                interest(symbol),
            ]
        else:
            pytest.fail(f"Forbidden endpoint: {request.url}")
        if paths[-1] == failed:
            response.status_code = 451
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = collector.collect_derivatives(SnapshotRequest(asset=asset, timeout_seconds=3))
    assert paths == ["funding", "interest"]
    assert result.symbol == f"{asset}USDT"
    assert bool(result.funding_failure) == (failed == "funding")
    assert bool(result.interest_failure) == (failed == "interest")
    assert result.features[2].value == (None if failed == "interest" else Decimal(0))
    assert bool(result.funding) == (failed != "funding")


def test_redirect_not_followed_and_other_endpoint_survives(monkeypatch, fixed_clock):
    calls = []

    def send(session, request, **kwargs):
        calls.append(request.url)
        response = requests.Response()
        response.request = request
        response.status_code = 302 if len(calls) == 1 else 200
        response.headers["Location"] = "https://example.com/private/order"
        response._content = b"[]"
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = collector.collect_derivatives(SnapshotRequest(asset="BTC"))
    assert len(calls) == 2
    assert "redirect" in result.funding_failure
    assert result.interest_failure is None


def test_funding_pagination_advances_start_and_preserves_all_events(monkeypatch, fixed_clock):
    starts = []
    first = [funding(hours=720 - i / 2) for i in range(1000)]

    def send(session, request, **kwargs):
        response = requests.Response()
        response.request = request
        response.status_code = 200
        url = urlsplit(request.url)
        if url.path.endswith("fundingRate"):
            starts.append(int(parse_qs(url.query)["startTime"][0]))
            payload = first if len(starts) == 1 else [funding(hours=1)]
        else:
            payload = []
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = collector.collect_derivatives(SnapshotRequest(asset="BTC"))
    assert starts[1] == first[-1]["fundingTime"] + 1
    assert len(result.funding) == 1001
    assert result.funding_failure is None


@pytest.mark.parametrize("mode", ["transport", "malformed", "future", "outside_window"])
def test_collection_rejects_invalid_evidence_without_losing_other_endpoint(
    monkeypatch, fixed_clock, mode
):
    def send(session, request, **kwargs):
        response = requests.Response()
        response.request = request
        response.status_code = 200
        if "fundingRate" in request.url:
            if mode == "transport":
                raise requests.Timeout("bounded timeout")
            payload = [funding(hours=-1 if mode == "future" else 721)]
            raw = "[" if mode == "malformed" else json.dumps(payload)
        else:
            raw = json.dumps([interest(hours=1), interest()])
        response._content = raw.encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = collector.collect_derivatives(SnapshotRequest(asset="BTC"))
    assert result.funding == ()
    assert result.funding_failure
    assert len(result.funding_failure) <= 240
    assert result.features[0].value is None
    assert result.features[2].value == Decimal(0)


def test_bulk_parse_failure_stays_compact():
    with pytest.raises(ValidationError) as caught:
        collector.parse_funding(json.dumps([{"unexpected": "secret"}] * 1000), "BTCUSDT")
    detail = collector.parse_failure(caught.value)
    assert len(detail) <= 240
    assert "secret" not in detail
