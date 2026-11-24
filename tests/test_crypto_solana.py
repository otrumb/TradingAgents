import json
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from pydantic import ValidationError

from tradingagents.dataflows.crypto_context import collect_crypto_context
from tradingagents.dataflows.crypto_models import SnapshotRequest
from tradingagents.dataflows.crypto_solana import parse_performance, parse_supply

PERFORMANCE = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": [
        {"slot": 987, "numTransactions": 120, "numNonVoteTransactions": 30, "samplePeriodSecs": 60}
    ],
}
SUPPLY = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "context": {"slot": 988},
        "value": {
            "total": 5_000_000_001,
            "circulating": 4_000_000_000,
            "nonCirculating": 1_000_000_001,
        },
    },
}


def test_performance_separates_vote_inclusive_and_non_vote_tps():
    metrics = parse_performance(json.dumps(PERFORMANCE))
    assert [(m.name, m.value) for m in metrics] == [
        ("sample_slot", Decimal(987)),
        ("transactions_per_second_including_votes", Decimal(2)),
        ("non_vote_transactions_per_second", Decimal("0.5")),
    ]
    assert all(m.observed_at is None for m in metrics)
    assert "60s" in metrics[1].resolution
    assert "including votes" in metrics[1].unit


def test_missing_non_vote_count_is_unknown_not_zero():
    raw = '{"jsonrpc":"2.0","id":1,"result":[{"slot":987,"numTransactions":120,"samplePeriodSecs":60}]}'
    assert parse_performance(raw)[2].value is None


def test_supply_preserves_finalized_slot_and_exact_lamport_conversion():
    metrics = parse_supply(json.dumps(SUPPLY))
    assert [(m.value, m.unit) for m in metrics] == [
        (Decimal(988), "slot"),
        (Decimal("5.000000001"), "SOL"),
        (Decimal(4), "SOL"),
        (Decimal("1.000000001"), "SOL"),
    ]
    assert all(m.observed_at is None for m in metrics)
    assert "finalized at slot 988" in metrics[1].resolution


@pytest.mark.parametrize(
    "raw",
    [
        '{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"unavailable"}}',
        '{"jsonrpc":"2.0","id":1,"result":[]}',
        '{"jsonrpc":"2.0","id":1,"result":[{"slot":1,"numTransactions":1,"samplePeriodSecs":0}]}',
        '{"jsonrpc":"2.0","id":1,"result":[{"slot":1,"numTransactions":-1,"samplePeriodSecs":60}]}',
    ],
)
def test_invalid_performance_is_rejected(raw):
    with pytest.raises(ValidationError):
        parse_performance(raw)


@pytest.mark.parametrize("failure", [None, "http", "transport", "parse", "redirect"])
def test_sol_collection_uses_fixed_read_methods_and_keeps_partial_results(monkeypatch, failure):
    seen = []
    received_after = datetime.now(UTC)

    def send(session, request, **kwargs):
        seen.append(request)
        assert session.trust_env is False
        assert kwargs["allow_redirects"] is False
        assert "Authorization" not in request.headers
        assert "Cookie" not in request.headers
        url = urlsplit(request.url)
        response = requests.Response()
        response.request = request
        response.status_code = 200
        if request.method == "POST":
            assert request.url == "https://api.mainnet.solana.com/"
            body = json.loads(request.body)
            if body["method"] == "getRecentPerformanceSamples":
                assert body == {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getRecentPerformanceSamples",
                    "params": [1],
                }
                payload = PERFORMANCE
            else:
                assert body == {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSupply",
                    "params": [
                        {"commitment": "finalized", "excludeNonCirculatingAccountsList": True}
                    ],
                }
                payload = SUPPLY
                if failure == "http":
                    response.status_code = 429
                elif failure == "transport":
                    raise requests.Timeout("RPC timeout")
                elif failure == "parse":
                    payload = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000}}
                elif failure == "redirect":
                    response.status_code = 307
                    response.headers["Location"] = "https://other.invalid"
        else:
            assert request.method == "GET"
            assert url.hostname in {"fapi.binance.com", "api.llama.fi", "stablecoins.llama.fi"}
            if url.hostname == "fapi.binance.com":
                assert parse_qs(url.query)["symbol"] == ["SOLUSDT"]
                row = {
                    "symbol": "SOLUSDT",
                    "time": 1704067200000,
                    "timestamp": 1704067200000,
                    "markPrice": "150",
                    "indexPrice": "149",
                    "lastFundingRate": "0",
                    "openInterest": "12",
                    "longShortRatio": "1",
                    "longAccount": "0.5",
                    "shortAccount": "0.5",
                    "bidPrice": "149",
                    "bidQty": "3",
                    "askPrice": "150",
                    "askQty": "4",
                }
                if url.path.endswith(("fundingRate", "openInterestHist")):
                    payload = []
                else:
                    payload = [row] if "globalLongShort" in url.path else row
            elif url.hostname == "api.llama.fi":
                payload = [{"name": "Bitcoin", "tvl": 999}, {"name": "Solana", "tvl": 123}]
            else:
                payload = [
                    {"name": "Bitcoin", "totalCirculatingUSD": {"peggedUSD": 999}},
                    {"name": "Solana", "totalCirculatingUSD": {"peggedUSD": 456}},
                ]
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    context = collect_crypto_context(SnapshotRequest(asset="SOL"))

    assert len(seen) == 10
    assert {urlsplit(request.url).path for request in seen[-2:]} == {
        "/fapi/v1/fundingRate",
        "/futures/data/openInterestHist",
    }
    assert context.derivatives.symbol == "SOLUSDT"
    assert context.derivatives.funding_failure is None
    assert context.derivatives.interest_failure is None
    assert len(context.snapshots) == 10
    binance = [s for s in context.snapshots if s.source == "binance"]
    assert len(binance) == 4
    assert all(s.failure is None and s.coverage == "SOLUSDT" for s in binance)
    chain = [s for s in context.snapshots if s.source == "defillama"]
    assert [s.metrics[0].value for s in chain] == [Decimal(123), Decimal(456)]
    assert all(s.coverage == "Solana" for s in chain)
    rpc = [s for s in context.snapshots if s.source == "solana_rpc"]
    assert rpc[0].failure is None
    assert rpc[0].metrics[1].value == Decimal(2)
    assert rpc[0].endpoint == "https://api.mainnet.solana.com#getRecentPerformanceSamples"
    assert rpc[1].endpoint == "https://api.mainnet.solana.com#getSupply"
    assert all(s.fetched_at >= received_after and s.observed_at is None for s in rpc)
    if failure is None:
        assert rpc[1].failure is None
        assert rpc[1].metrics[1].value == Decimal("5.000000001")
    else:
        assert rpc[1].metrics == ()
        assert rpc[1].failure.kind == ("parse" if failure == "redirect" else failure)
    unsupported = [s for s in context.snapshots if s.source in {"deribit", "growthepie"}]
    assert {s.source for s in unsupported} == {"deribit", "growthepie"}
    assert all(s.failure.kind == "unavailable" and s.metrics == () for s in unsupported)
