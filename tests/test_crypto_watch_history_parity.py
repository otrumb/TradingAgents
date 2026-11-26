import json
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from tests.crypto_watch_fixtures import CUTOFF, MS
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import (
    DURATIONS,
    EPOCH,
    WatchRequest,
    WatchSnapshot,
)


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("historical", [False, True], ids=["current-open", "closed-boundary"])
def test_full_decision_matches_legacy_api_window(monkeypatch, asset, historical):
    cutoff = CUTOFF if historical else CUTOFF + timedelta(minutes=30)
    universe = {}
    for interval, duration in DURATIONS.items():
        universe[interval] = [
            [
                (CUTOFF + (i - 160) * duration - EPOCH) // MS,
                str(200 + i * 2),
                str(201 + i * 2),
                str(199 + i * 2),
                str(200 + i * 2),
                "20",
                (CUTOFF + (i - 159) * duration - EPOCH) // MS - 1,
                "2000",
                10,
                "10",
                "1000",
                "0",
            ]
            for i in range(220)
        ]
    universe["4h"][158][3] = "1"

    def send(session, request, **kwargs):
        url = urlsplit(request.url)
        payload = {"serverTime": (cutoff + timedelta(seconds=5) - EPOCH) // MS}
        if url.path == "/fapi/v1/klines":
            query = parse_qs(url.query)
            eligible = [
                row for row in universe[query["interval"][0]] if row[0] <= int(query["endTime"][0])
            ]
            payload = eligible[-int(query["limit"][0]) :]
        else:
            assert url.path == "/fapi/v1/time" and not historical
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    request = WatchRequest(asset=asset, as_of=cutoff if historical else None)

    legacy = collect_watch(request)
    enhanced = collect_watch(request.model_copy(update={"history_limit": 161}))
    saved = WatchSnapshot.model_validate_json(enhanced.model_dump_json())
    legacy_decision = evaluate_watch(legacy)

    assert tuple(len(s.candles) for s in legacy.series) == (100 if historical else 99,) * 3
    assert tuple(len(s.candles) for s in saved.series) == (160,) * 3
    assert all(c.close_time < cutoff for s in saved.series for c in s.candles)
    assert legacy_decision.action == "consider_long"
    shortened = legacy.model_copy(
        update={
            "series": tuple(s.model_copy(update={"candles": s.candles[1:]}) for s in legacy.series)
        }
    )
    assert evaluate_watch(shortened).conditions != legacy_decision.conditions
    assert evaluate_watch(saved) == legacy_decision
