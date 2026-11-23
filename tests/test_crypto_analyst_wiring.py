from datetime import date
from html.parser import HTMLParser

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cli.models import AnalystType, AssetType
from cli.prompts import detect_asset_type, parse_analysts
from tradingagents.agents.analysts import crypto_analyst
from tradingagents.agents.context import get_instrument_context_from_state
from tradingagents.dataflows import crypto_context
from tradingagents.dataflows.crypto_models import CryptoContext
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup
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

    report = crypto_analyst.collect_crypto_report(ticker, date.today().isoformat())

    assert requests[0].asset == asset
    assert CryptoContext.model_validate_json(report).asset == asset


@pytest.mark.parametrize("ticker", ["SOL-USD", "solusdt", "SOL-USDC", "SOLUSD+"])
def test_cli_normalizes_and_classifies_sol(ticker):
    normalized = normalize_symbol(ticker)
    assert normalized == "SOL-USD"
    assert detect_asset_type(normalized) == AssetType.CRYPTO


def test_unsupported_asset_never_reaches_collector(monkeypatch):
    def forbidden(request):
        pytest.fail("unsupported asset fetched context")

    monkeypatch.setattr(crypto_context, "collect_crypto_context", forbidden)
    assert "unavailable" in crypto_analyst.collect_crypto_report(
        "XRP-USD", date.today().isoformat()
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
