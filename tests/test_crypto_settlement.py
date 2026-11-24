import json
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest
import requests

from tradingagents.dataflows.crypto_settlement import SettlementPriceError, fetch_crypto_returns
from tradingagents.memory import TradingMemoryLog, settlement

DAY = 86400000
ENTRY = 1767398400000  # Saturday, 2026-01-03 UTC; decision Friday January 2.
EXIT_CLOSE = ENTRY + 3 * DAY - 1


def row(opened, close):
    return [opened, close, close, close, close, "1", opened + DAY - 1, "1", 1, "1", "1", "0"]


@pytest.fixture
def wire(monkeypatch):
    state = {
        "clock": {"serverTime": EXIT_CLOSE + 5001},
        "BTCUSDT": [row(ENTRY + i * DAY, str(100 + i * 10)) for i in range(3)],
        "ETHUSDT": [row(ENTRY + i * DAY, str(200 + i * 30)) for i in range(3)],
        "SOLUSDT": [row(ENTRY + i * DAY, str(50 + i * 10)) for i in range(3)],
        "calls": [],
    }

    def send(session, request, **kwargs):
        assert session.trust_env is False and session.auth is None
        assert request.method == "GET"
        assert not {"authorization", "x-mbx-apikey"} & {h.lower() for h in request.headers}
        assert kwargs["allow_redirects"] is False
        assert 0 < kwargs["timeout"] <= 60
        url = urlsplit(request.url)
        assert (url.scheme, url.netloc) == ("https", "data-api.binance.vision")
        state["calls"].append(url.path)
        if url.path == "/api/v3/time":
            payload = state["clock"]
        else:
            assert url.path == "/api/v3/klines"
            query = parse_qs(url.query)
            assert query["interval"] == ["1d"] and query["timeZone"] == ["0"]
            assert query["startTime"] == [str(ENTRY)]
            assert query["endTime"] == [str(EXIT_CLOSE)]
            assert query["limit"] == ["3"]
            payload = state[query["symbol"][0]]
        if isinstance(payload, requests.RequestException):
            raise payload
        response = requests.Response()
        response.status_code = 302 if payload == "redirect" else 429 if payload == "429" else 200
        if payload == "redirect":
            response.headers["Location"] = "https://example.com/private"
        response.request = request
        response.url = request.url
        response._content = b"bad JSON" if payload == "bad JSON" else json.dumps(payload).encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(settlement, "get_closes", lambda *args: pytest.fail("Yahoo for crypto"))
    return state


@pytest.mark.parametrize(
    "ticker,raw,alpha",
    [
        ("BTC-USD", 0.2, 0.2),
        (" btcusdt+ ", 0.2, 0.2),
        ("BTC-USDC", 0.2, 0.2),
        ("ETHUSD", 0.3, 0.1),
        ("eth-usdt", 0.3, 0.1),
        ("ETH-USDC", 0.3, 0.1),
        ("SOL-USD", 0.4, 0.2),
        ("SOLUSDT", 0.4, 0.2),
        ("solusdc+", 0.4, 0.2),
    ],
)
def test_aliases_settle_weekends_on_exact_days(wire, ticker, raw, alpha):
    result = fetch_crypto_returns(ticker, "2026-01-02", 2)

    assert result == (pytest.approx(raw), pytest.approx(alpha), 2, "2026-01-05")
    assert wire["calls"].count("/api/v3/time") == 1
    assert len(wire["calls"]) == (2 if ticker.strip().upper().startswith("BTC") else 3)


@pytest.mark.parametrize("lag,complete", [(0, False), (4999, False), (5000, False), (5001, True)])
def test_finality_boundary(wire, lag, complete):
    wire["clock"] = {"serverTime": EXIT_CLOSE + lag}

    result = fetch_crypto_returns("BTCUSD", "2026-01-02", 2)

    assert result == ((0.2, 0.2, 2, "2026-01-05") if complete else (None,) * 4)


@pytest.mark.parametrize(
    "symbol,index", [("ETHUSDT", 0), ("ETHUSDT", 1), ("ETHUSDT", 2), ("BTCUSDT", 1)]
)
def test_missing_day_in_either_leg_stays_pending(wire, symbol, index):
    del wire[symbol][index]

    assert fetch_crypto_returns("ETHUSD", "2026-01-02", 2) == (None,) * 4


@pytest.mark.parametrize(
    "payload,reason",
    [
        ("redirect", "invalid response"),
        ("429", "HTTP 429"),
        (requests.Timeout("secret body"), "Timeout"),
        ("bad JSON", "invalid response"),
        ([[1]], "invalid response"),
        ({"error": "private body"}, "invalid response"),
        ([row(ENTRY, "100"), row(ENTRY, "101")], "dates/order"),
        ([row(ENTRY + DAY, "100"), row(ENTRY, "101")], "dates/order"),
        ([row(ENTRY - DAY, "100")], "dates/order"),
        ([row(ENTRY, "0")], "invalid response"),
        ([row(ENTRY, "-1")], "invalid response"),
        ([row(ENTRY, "NaN")], "invalid response"),
        ([row(ENTRY, "Infinity")], "invalid response"),
        ([row(ENTRY + 1, "100")], "invalid response"),
    ],
)
def test_bad_candles_or_transport_fail_compactly(wire, payload, reason):
    wire["BTCUSDT"] = payload

    with pytest.raises(SettlementPriceError, match=reason) as caught:
        fetch_crypto_returns("BTCUSD", "2026-01-02", 2)

    assert len(str(caught.value)) < 100
    assert "secret" not in str(caught.value) and "private body" not in str(caught.value)


@pytest.mark.parametrize(
    "clock", [{"serverTime": True}, {}, "redirect", "429", requests.Timeout("secret"), "bad JSON"]
)
def test_clock_failure_stops_before_prices(wire, clock):
    wire["clock"] = clock

    with pytest.raises(SettlementPriceError):
        fetch_crypto_returns("ETHUSD", "2026-01-02", 2)

    assert wire["calls"] == ["/api/v3/time"]


@pytest.mark.parametrize(
    "ticker,label,raw,alpha",
    [
        ("BTCUSDT", "USDT cash", 0.2, 0.2),
        ("ETH-USD", "BTCUSDT", 0.3, 0.1),
        ("SOLUSDC+", "BTCUSDT", 0.4, 0.2),
    ],
)
def test_pending_routing_records_exit_date_and_point_in_time_context(
    wire,
    tmp_path,
    ticker,
    label,
    raw,
    alpha,
):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md")})
    log.store_decision(ticker, "2026-01-02", "Rating: Buy")

    class Reflector:
        def reflect_on_final_decision(self, **kwargs):
            assert kwargs["benchmark_name"] == label
            assert kwargs["raw_return"] == pytest.approx(raw)
            assert kwargs["alpha_return"] == pytest.approx(alpha)
            assert kwargs["holding_days"] == 2
            return "weekend lesson"

    result = settlement.settle_pending(
        ticker,
        log,
        Reflector(),
        {"crypto_holding_period_days": 2, "benchmark_ticker": "QQQ"},
    )

    assert result.settled == [(ticker, "2026-01-02")] and result.failed == []
    assert log.load_entries()[0]["resolved"] == "2026-01-05"
    assert log.get_past_context(ticker, as_of="2026-01-04") == ""
    assert "weekend lesson" in log.get_past_context(ticker, as_of="2026-01-05")


@pytest.mark.parametrize("payload,failed", [([], False), ("429", True)])
def test_incomplete_or_failed_crypto_keeps_memory_pending(wire, tmp_path, payload, failed):
    wire["BTCUSDT"] = payload
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md")})
    log.store_decision("BTCUSD", "2026-01-02", "Rating: Buy")

    result = settlement.settle_pending("BTCUSD", log, None, {"crypto_holding_period_days": 2})

    assert result.settled == []
    assert bool(result.failed) == failed
    assert log.load_entries()[0]["pending"] is True


def test_stock_pending_still_uses_yahoo_and_configured_benchmark(monkeypatch, tmp_path):
    days = pd.bdate_range("2026-01-02", periods=4)
    prices = {
        "NVDA": pd.Series([999, 100, 110, 130], index=days),
        "QQQ": pd.Series([999, 200, 210, 220], index=days),
    }
    monkeypatch.setattr(settlement, "get_closes", lambda ticker, *args: prices[ticker])
    monkeypatch.setattr(settlement, "_now", lambda: pd.Timestamp("2026-02-01", tz="UTC"))
    monkeypatch.setattr(requests.Session, "send", lambda *a, **kw: pytest.fail("Binance for stock"))
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md")})
    log.store_decision("NVDA", "2026-01-02", "Rating: Buy")

    class Reflector:
        def reflect_on_final_decision(self, **kwargs):
            assert kwargs["benchmark_name"] == "QQQ"
            assert kwargs["raw_return"] == pytest.approx(0.3)
            assert kwargs["alpha_return"] == pytest.approx(0.2)
            return "stock lesson"

    result = settlement.settle_pending(
        "NVDA",
        log,
        Reflector(),
        {"holding_period_days": 2, "benchmark_ticker": "QQQ"},
    )

    assert result.settled == [("NVDA", "2026-01-02")] and result.failed == []
    assert log.load_entries()[0]["resolved"] == "2026-01-07"
