import json
from datetime import UTC, datetime
from html.parser import HTMLParser

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from cli.models import AnalystType, AssetType
from cli.prompts import detect_asset_type, parse_analysts
from tradingagents.agents.analysts import crypto_analyst
from tradingagents.agents.context import get_instrument_context_from_state
from tradingagents.dataflows import crypto_context, crypto_spot
from tradingagents.dataflows.crypto_models import CryptoContext
from tradingagents.dataflows.crypto_spot_models import CryptoAnalystContext, SpotPriceContext
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup, _analyst_graph
from tradingagents.reporting import write_report_tree


@pytest.mark.parametrize("asset", [AssetType.CRYPTO, AssetType.STOCK])
def test_crypto_selection_requires_crypto_asset(asset):
    if asset == AssetType.CRYPTO:
        assert parse_analysts("crypto", asset) == [AnalystType.CRYPTO]
    else:
        with pytest.raises(ValueError, match="not available"):
            parse_analysts("crypto", asset)


def test_crypto_is_not_added_to_existing_selection():
    assert parse_analysts("market,news", AssetType.CRYPTO) == [
        AnalystType.MARKET,
        AnalystType.NEWS,
    ]


@pytest.mark.parametrize("ticker", ["BTC-USD", "ETH-USD", "SOL-USD"])
def test_historical_graph_propagates_unavailable_without_fetching(monkeypatch, ticker):
    def forbidden(request):
        pytest.fail("historical graph fetched current snapshot")

    monkeypatch.setattr(crypto_context, "collect_crypto_context", forbidden)
    monkeypatch.setattr(crypto_spot, "collect_spot_prices", forbidden)
    llm = FakeListChatModel(responses=["**Rating**: Hold"])
    graph = GraphSetup(llm, llm, ConditionalLogic(1, 1), 2).setup_graph(["crypto"]).compile()
    initial = Propagator().create_initial_state(ticker, "2000-01-01", "crypto")
    assert initial["crypto_report"] == ""

    result = graph.invoke(initial)

    assert "unavailable" in result["crypto_report"].lower()
    assert "2000-01-01" in result["crypto_report"]
    assert result["final_rating"] == "Hold"
    assert result["crypto_report"] in get_instrument_context_from_state(result)


@pytest.mark.parametrize(
    "ticker,asset",
    [
        ("ETH-USD", "ETH"),
        ("BTC-USD", "BTC"),
        ("SOL-USD", "SOL"),
        (" solusdt+ ", "SOL"),
        ("SOL-USDC", "SOL"),
        ("SOL", "SOL"),
    ],
)
def test_current_adapter_uses_typed_collector_request(monkeypatch, ticker, asset):
    requests = []

    def collect(request):
        requests.append(request)
        return CryptoContext(asset=request.asset, snapshots=())

    monkeypatch.setattr(crypto_context, "collect_crypto_context", collect)
    monkeypatch.setattr(
        crypto_spot, "collect_spot_prices", lambda request: SpotPriceContext(asset=request.asset)
    )

    report = crypto_analyst.collect_crypto_report(
        ticker,
        datetime.now(UTC).date().isoformat(),
        SpotPriceContext(asset=asset).model_dump_json(),
    )

    assert requests[0].asset == asset
    assert CryptoAnalystContext.model_validate_json(report).spot_prices.asset == asset


@pytest.mark.parametrize("ticker", ["SOL-USD", "solusdt", "SOL-USDC", "SOLUSD+"])
def test_cli_normalizes_and_classifies_sol(ticker):
    normalized = normalize_symbol(ticker)
    assert normalized == "SOL-USD"
    assert detect_asset_type(normalized) == AssetType.CRYPTO


def test_unsupported_asset_never_reaches_collector(monkeypatch):
    def forbidden(request):
        pytest.fail("unsupported asset fetched context")

    monkeypatch.setattr(crypto_context, "collect_crypto_context", forbidden)
    monkeypatch.setattr(crypto_spot, "collect_spot_prices", forbidden)
    assert "unavailable" in crypto_analyst.collect_crypto_report(
        "XRP-USD", datetime.now(UTC).date().isoformat()
    )


def test_stock_node_rejects_crypto_before_collection():
    node = crypto_analyst.create_crypto_analyst(FakeListChatModel(responses=["unused"]))
    state = Propagator().create_initial_state("AAPL", "2000-01-01")
    with pytest.raises(crypto_analyst.CryptoAssetRequiredError):
        node(state)


class ReportHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


def test_saved_crypto_section_is_written_and_removed_when_absent(tmp_path):
    state = {"crypto_report": "Funding observation: 0.003%", "trade_date": "2000-01-01"}

    path = write_report_tree(state, "BTC-USD", tmp_path)

    assert (tmp_path / "1_analysts" / "crypto.md").read_text() == state["crypto_report"]
    assert state["crypto_report"] in path.read_text()
    parser = ReportHTML()
    parser.feed((tmp_path / "complete_report.html").read_text(encoding="utf-8"))
    assert state["crypto_report"] in "".join(parser.text)

    write_report_tree({}, "AAPL", tmp_path)
    assert not (tmp_path / "1_analysts" / "crypto.md").exists()


def test_current_node_preserves_evidence_in_saved_reports_and_state_log(monkeypatch, tmp_path):
    from tests.test_cli_display import _bare_graph, _state
    from tests.test_crypto_market_tools import _spot_context
    from tradingagents.dataflows.crypto_models import Metric, SourceSnapshot
    from tradingagents.dataflows.crypto_spot_models import IntradayPrice

    now = datetime.now(UTC)
    evidence = _spot_context().model_copy(
        update={
            "intraday": IntradayPrice(
                asset="BTC", symbol="BTCUSDT", retrieved_at=now, server_time=now, price="123.45"
            ),
        }
    )
    full_context = CryptoContext(
        asset="BTC",
        snapshots=(
            SourceSnapshot(
                source="deribit",
                venue="Deribit",
                endpoint="https://example.com/options",
                fetched_at=now,
                observed_at=now,
                resolution="instant",
                coverage="BTC options",
                metrics=tuple(
                    Metric(
                        name="option_open_interest",
                        value=index,
                        unit="contracts",
                        observed_at=now,
                        resolution="instant",
                        coverage=f"BTC-OPTIONS-{index}",
                    )
                    for index in range(2000)
                ),
            ),
        ),
    )
    collected = []

    def collect(request):
        collected.append(request.asset)
        return full_context

    monkeypatch.setattr(crypto_context, "collect_crypto_context", collect)
    monkeypatch.setattr(crypto_spot, "collect_spot_prices", lambda request: evidence)
    llm = FakeListChatModel(responses=["Spot assessment"])
    received = []

    def invoke(self, messages):
        received.append(json.loads(messages[-1].content.split("\n", 2)[-1]))
        return AIMessage(content="Spot assessment")

    monkeypatch.setattr(FakeListChatModel, "invoke", invoke)
    state = Propagator().create_initial_state(
        "BTC-USD", now.date().isoformat(), "crypto", crypto_market_data=evidence.model_dump_json()
    )

    graph = _analyst_graph(
        ANALYST_NODE_SPECS["crypto"], crypto_analyst.create_crypto_analyst(llm), 2
    )
    result = graph.invoke(state)

    report = result["crypto_report"]
    assert received[0]["spot_prices"]["daily_row_count"] == 60
    assert "daily_candles" not in received[0]["spot_prices"]
    assert "snapshots" not in received[0]
    assert collected == ["BTC"]
    assert received[0]["sources"][0]["option_instrument_count"] == 2000
    assert len(received[0]["sources"][0]["option_families"]) == 1
    assert received[0]["sources"][0]["metrics"] == []
    serialized = json.loads(report.split("```json\n")[1].split("\n```")[0])
    assert serialized == received[0]
    assert serialized["spot_prices"]["latest_finalized"] == evidence.daily_candles[-1].model_dump(
        mode="json"
    )
    write_report_tree(state | result, "BTC-USD", tmp_path / "reports")
    sidecars = tmp_path / "reports" / "evidence"
    assert (
        SpotPriceContext.model_validate_json((sidecars / "crypto_spot.json").read_text())
        == evidence
    )
    assert (
        CryptoContext.model_validate_json((sidecars / "crypto_context.json").read_text())
        == full_context
    )
    assert (tmp_path / "reports" / "1_analysts" / "crypto.md").read_text() == report
    parser = ReportHTML()
    parser.feed((tmp_path / "reports" / "complete_report.html").read_text())
    for text in (
        report,
        (tmp_path / "reports" / "complete_report.md").read_text(),
        get_instrument_context_from_state(state | result),
        "".join(parser.text),
    ):
        start = text.index('{\n  "asset"')
        rendered, _ = json.JSONDecoder().raw_decode(text[start:])
        assert "snapshots" not in rendered
        assert "daily_candles" not in rendered["spot_prices"]
        assert rendered["spot_prices"]["daily_row_count"] == 60
    log_state = _state("BTC-USD") | result
    _bare_graph(tmp_path)._log_state(now.date().isoformat(), log_state)
    log = next(tmp_path.rglob("full_states_log*.json"))
    assert json.loads(log.read_text(encoding="utf-8"))["crypto_report"] == report


def test_report_reuse_removes_only_owned_crypto_sidecars(tmp_path):
    evidence = SpotPriceContext(asset="BTC")
    context = CryptoContext(asset="BTC", snapshots=())
    write_report_tree(
        {"crypto_market_data": evidence.model_dump_json(), "crypto_context_data": context.render()},
        "BTC-USD",
        tmp_path,
    )
    (tmp_path / "evidence" / "notes.txt").write_text("keep")

    write_report_tree({}, "AAPL", tmp_path)

    assert {p.name for p in (tmp_path / "evidence").iterdir()} == {"notes.txt"}


def test_historical_graph_retains_compact_spot_without_current_context(monkeypatch, tmp_path):
    from datetime import date

    from tests.test_crypto_market_tools import _spot_context

    candles = tuple(
        candle.model_copy(
            update={
                "analysis_date": date(2026, 6, 1),
                "cutoff": datetime(2026, 6, 2, tzinfo=UTC),
                "server_time": None,
                "finality_lag_ms": 0,
            }
        )
        for candle in _spot_context().daily_candles
    )
    spot = SpotPriceContext(
        asset="BTC",
        semantics="historical_finalized",
        analysis_date=date(2026, 6, 1),
        daily_candles=candles,
    )

    def forbidden(request):
        pytest.fail("historical analyst fetched live evidence")

    monkeypatch.setattr(crypto_context, "collect_crypto_context", forbidden)
    monkeypatch.setattr(crypto_spot, "collect_spot_prices", forbidden)
    llm = FakeListChatModel(responses=["**Rating**: Hold"])
    graph = GraphSetup(llm, llm, ConditionalLogic(1, 1), 2).setup_graph(["crypto"]).compile()
    state = Propagator().create_initial_state(
        "BTC-USD", "2026-06-01", "crypto", crypto_market_data=spot.model_dump_json()
    )

    result = graph.invoke(state)
    write_report_tree(result, "BTC-USD", tmp_path)

    summary = json.loads(result["crypto_report"].split("```json\n")[1].split("\n```")[0])
    assert summary == spot.summary().model_dump(mode="json")
    assert summary["daily_row_count"] == 60
    assert summary["intraday"] is None
    assert result["crypto_context_data"] == ""
    assert result["crypto_market_data"] == spot.model_dump_json()
    assert (
        SpotPriceContext.model_validate_json(
            (tmp_path / "evidence" / "crypto_spot.json").read_text()
        )
        == spot
    )
    assert not (tmp_path / "evidence" / "crypto_context.json").exists()
