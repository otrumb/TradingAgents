import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio import PortfolioContext


@pytest.mark.parametrize("historical", [False, True])
def test_run_collects_public_spot_once_per_asset_before_render(monkeypatch, historical):
    import tradingagents.graph.trading_graph as module
    import tradingagents.portfolio_runtime as runtime

    now = datetime(2026, 8, 14, 12, tzinfo=UTC)

    class Clock(datetime):
        @staticmethod
        def now(tz):
            return now

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(runtime, "datetime", Clock)
    from tradingagents.dataflows.crypto_watch_models import WatchSeries, WatchSnapshot

    def watch(request):
        cutoff = request.as_of or now
        return WatchSnapshot(
            asset=request.asset,
            symbol=f"{request.asset}USDT",
            cutoff=cutoff,
            as_of=cutoff,
            server_time=None,
            series=tuple(WatchSeries(interval=i) for i in ("1d", "4h", "1h")),
        )

    monkeypatch.setattr("tradingagents.graph.run_decisions.collect_watch", watch)
    calls = []
    opened = int(datetime(2026, 8, 12, tzinfo=UTC).timestamp()) * 1000

    def send(session, request, **kwargs):
        url = urlsplit(request.url)
        symbol = parse_qs(url.query).get("symbol", [None])[0]
        calls.append((url.path, symbol))
        payloads = {
            "/api/v3/time": {"serverTime": int(now.timestamp()) * 1000},
            "/api/v3/klines": [
                [
                    opened,
                    "100",
                    "120",
                    "90",
                    "110",
                    "20",
                    opened + 86400000 - 1,
                    "2000",
                    10,
                    "10",
                    "1000",
                    "0",
                ]
            ],
            "/api/v3/ticker/price": {"symbol": symbol, "price": "115"},
        }
        if historical:
            assert parse_qs(url.query)["endTime"] == [str(opened + 2 * 86400000 - 1)]
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payloads[url.path]).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    graph = object.__new__(TradingAgentsGraph)
    graph.propagator = Propagator()
    graph.resolve_instrument_context = lambda *args: ""
    book = PortfolioContext.model_validate(
        {
            "currency": "USD",
            "cash": 1000,
            "positions": [
                {"ticker": "BTCUSD", "quantity": 1},
                {"ticker": "BTCUSDT", "quantity": 2},
                {"ticker": "ETHUSD", "quantity": 1},
                {"ticker": "SOLUSD", "quantity": 1},
            ],
        }
    )
    before = book.model_dump_json()

    state = graph.create_run_state(
        "BTC-USD", "2026-08-13" if historical else "2026-08-14", "crypto", book
    )

    expected = (
        [("/api/v3/klines", "BTCUSDT")]
        if historical
        else [
            call
            for asset in ("BTC", "ETH", "SOL")
            for call in (
                ("/api/v3/time", None),
                ("/api/v3/klines", f"{asset}USDT"),
                ("/api/v3/ticker/price", f"{asset}USDT"),
            )
        ]
    )
    assert calls == expected
    spot = json.loads(state["crypto_market_data"])
    assert len(spot["daily_candles"]) == 1
    valuation = json.loads(state["portfolio_context"].split("): ")[-1])
    assert len(valuation["marks"]) == (0 if historical else 3)
    assert book.model_dump_json() == before
