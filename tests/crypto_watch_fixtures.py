import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from tradingagents.dataflows.crypto_watch_models import DURATIONS, EPOCH

CUTOFF = datetime(2024, 5, 1, tzinfo=UTC)
MS = timedelta(milliseconds=1)


def wire_rows(interval, direction=1):
    duration = DURATIONS[interval]
    count = 55 if interval == "1d" else 24
    rows = []
    for index in range(count):
        start = CUTOFF - (count - index) * duration
        price = Decimal(200) + direction * Decimal(index * 2)
        prices = list(map(str, (price, price + 1, price - 1, price)))
        times = ((start - EPOCH) // MS, (start + duration - EPOCH) // MS - 1)
        rows.append([times[0], *prices, "20", times[1], *["2000", 10, "10", "1000", "0"]])
    if interval == "4h":
        rows[-2][3 if direction > 0 else 2] = "1" if direction > 0 else "400"
    return rows


@pytest.fixture
def watch_wire(monkeypatch):
    state = {
        "calls": [],
        "clock": {"serverTime": (CUTOFF - EPOCH) // MS + 5000},
        "rows": {interval: wire_rows(interval) for interval in DURATIONS},
    }

    def send(session, request, **kwargs):
        assert session.trust_env is False
        assert session.auth is None
        assert request.method == "GET"
        assert not {"authorization", "x-mbx-apikey"} & {h.lower() for h in request.headers}
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == 7
        url = urlsplit(request.url)
        assert (url.scheme, url.netloc) == ("https", "fapi.binance.com")
        state["calls"].append((url.path, parse_qs(url.query)))
        if url.path == "/fapi/v1/time":
            assert not url.query
            payload = state["clock"]
        else:
            assert url.path == "/fapi/v1/klines"
            query = parse_qs(url.query)
            interval = query["interval"][0]
            assert query == {
                "symbol": [state.get("symbol", "BTCUSDT")],
                "interval": [interval],
                "limit": [str(state.get("limit", 100))],
                "endTime": [str(state.get("end", (CUTOFF - EPOCH) // MS - 1))],
            }
            payload = state["rows"][interval]
        if isinstance(payload, requests.RequestException):
            raise payload
        response = requests.Response()
        response.status_code = 302 if payload == "redirect" else 429 if payload == "429" else 200
        response.headers["Location"] = "https://example.com/private"
        response.request = request
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return state
