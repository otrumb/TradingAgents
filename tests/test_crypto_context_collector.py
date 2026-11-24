import json
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from tradingagents.dataflows.crypto_chain import parse_growth, parse_stables, parse_tvl
from tradingagents.dataflows.crypto_context import collect_crypto_context
from tradingagents.dataflows.crypto_models import ContextContractError, SnapshotRequest


@pytest.mark.parametrize(
    "value,expected", [(None, None), (0, Decimal(0)), (123.5, Decimal("123.5"))]
)
def test_chain_values_preserve_zero_null_and_timestamp_absence(
    value: float | None, expected: Decimal | None
) -> None:
    result = parse_tvl(json.dumps([{"name": "Ethereum", "tvl": value}]), "Ethereum")
    assert result[0].value == expected
    assert result[0].observed_at is None
    assert result[0].unit == "USD"


def test_stables_uses_usd_valuation_and_excludes_other_pegs() -> None:
    result = parse_stables(
        '[{"name":"Ethereum","totalCirculatingUSD":{"peggedUSD":0,"peggedEUR":999},"totalCirculating":{"peggedUSD":12}}]',
        "Ethereum",
    )
    assert result[0].value == Decimal(0)
    assert result[0].observed_at is None


def test_missing_chain_is_failure_not_zero_or_other_network() -> None:
    with pytest.raises(ContextContractError):
        parse_stables('[{"name":"Ethereum","totalCirculatingUSD":{"peggedUSD":99}}]', "Bitcoin")


def test_growth_selects_latest_matching_network_even_when_null() -> None:
    raw = json.dumps(
        [
            {"metric_key": "daa", "origin_key": "ethereum", "date": "2024-01-02", "value": None},
            {"metric_key": "daa", "origin_key": "base", "date": "2024-01-03", "value": 999},
            {"metric_key": "daa", "origin_key": "ethereum", "date": "2024-01-01", "value": 123},
        ]
    )
    result = parse_growth(raw, "daa")
    assert result[0].value is None
    assert result[0].observed_at == datetime(2024, 1, 2, tzinfo=UTC)
    assert result[0].unit == "active addresses"


@pytest.fixture
def public_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return datetime(2024, 1, 10, tzinfo=UTC)

    monkeypatch.setattr("tradingagents.dataflows.crypto_context.datetime", FixedDateTime)

    def send(
        session: requests.Session, request: requests.PreparedRequest, **kwargs: float
    ) -> requests.Response:
        assert request.method == "GET"
        assert "Authorization" not in request.headers
        url = urlsplit(request.url or "")
        symbol = parse_qs(url.query).get("symbol", ["ETHUSDT"])[0]
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.url = request.url or ""
        response.encoding = "utf-8"
        payload = "{}"
        if url.path.endswith(("fundingRate", "openInterestHist")):
            payload = "[]"
        elif url.path.endswith("premiumIndex"):
            payload = json.dumps(
                {
                    "symbol": symbol,
                    "time": 1704067200000,
                    "markPrice": "2300",
                    "indexPrice": "2299",
                    "lastFundingRate": "0",
                }
            )
        elif url.path.endswith("openInterest"):
            response.status_code = 429
        elif url.path.endswith("globalLongShortAccountRatio"):
            payload = "not JSON"
        elif url.path.endswith("bookTicker"):
            raise requests.Timeout("wire timeout")
        elif url.path.endswith("get_book_summary_by_currency"):
            asset = parse_qs(url.query)["currency"][0]
            payload = json.dumps(
                {
                    "result": [
                        {
                            "instrument_name": f"{asset}-29MAR24-3000-C",
                            "creation_timestamp": 1704067200000,
                            "open_interest": 7,
                            "volume": 0,
                            "mark_iv": None,
                        }
                    ]
                }
            )
        elif url.path.endswith("get_volatility_index_data"):
            payload = '{"result":{"data":[[1704067200000,40,50,30,45]],"continuation":null}}'
        elif url.path.endswith("/chains"):
            payload = '[{"name":"Ethereum","tvl":123},{"name":"Bitcoin","tvl":456}]'
        elif url.path.endswith("stablecoinchains"):
            payload = '[{"name":"Ethereum","totalCirculatingUSD":{"peggedUSD":0}}]'
        elif "/export/" in url.path:
            key = {
                "daa.json": "daa",
                "fees.json": "fees_paid_usd",
                "stables_mcap.json": "stables_mcap",
            }[url.path.rsplit("/", 1)[1]]
            payload = json.dumps(
                [{"metric_key": key, "origin_key": "ethereum", "date": "2024-01-01", "value": 10}]
            )
        else:
            pytest.fail(f"Unexpected public endpoint: {request.url}")
        response._content = payload.encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)


@pytest.mark.parametrize("asset,count", [("ETH", 11), ("BTC", 9)])
def test_collect_preserves_other_sources_when_individual_endpoints_fail(
    public_wire: None, asset: str, count: int
) -> None:
    context = collect_crypto_context(SnapshotRequest.model_validate({"asset": asset}))
    assert len(context.snapshots) == count
    assert [s.failure.kind if s.failure else None for s in context.snapshots[:4]] == [
        None,
        "http",
        "parse",
        "transport",
    ]
    assert context.snapshots[0].metrics[2].value == Decimal(0)
    assert context.snapshots[0].venue == "Binance USD-M Futures"
    assert context.snapshots[4].venue == "Deribit"
    assert context.snapshots[6].metrics[0].value == (123 if asset == "ETH" else 456)
    assert context.snapshots[6].observed_at is None
    assert any("stale" in u for u in context.snapshots[0].unknowns)
    assert json.loads(context.render())["asset"] == asset
    if asset == "BTC":
        assert context.snapshots[-1].failure is not None
        assert context.snapshots[-1].failure.kind == "unavailable"
        assert context.snapshots[-1].metrics == ()
