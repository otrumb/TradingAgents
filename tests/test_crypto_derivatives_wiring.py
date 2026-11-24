import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from tradingagents.agents.analysts.crypto_analyst import create_crypto_analyst
from tradingagents.dataflows import crypto_context, crypto_derivatives
from tradingagents.dataflows.crypto_derivatives_features import with_features
from tradingagents.dataflows.crypto_derivatives_models import DerivativesHistory, FundingObservation
from tradingagents.dataflows.crypto_models import CryptoContext
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import _analyst_graph
from tradingagents.reporting import write_report_tree


def test_graph_prompt_report_and_sidecar_keep_history_separate(monkeypatch, tmp_path):
    now = datetime.now(UTC)
    history = with_features(
        DerivativesHistory(
            symbol="BTCUSDT",
            retrieved_at=now,
            features=(),
            funding=tuple(
                FundingObservation(
                    symbol="BTCUSDT", observed_at=now - timedelta(hours=h), rate=Decimal("0.001")
                )
                for h in range(720, 0, -1)
            ),
        )
    )
    monkeypatch.setattr(
        crypto_context,
        "collect_crypto_context",
        lambda request: CryptoContext(asset="BTC", snapshots=(), derivatives=history),
    )
    received = []

    def invoke(self, messages):
        received.append(json.loads(messages[-1].content.split("\n", 2)[-1]))
        return AIMessage(content="Funding assessment")

    monkeypatch.setattr(FakeListChatModel, "invoke", invoke)
    graph = _analyst_graph(
        ANALYST_NODE_SPECS["crypto"],
        create_crypto_analyst(FakeListChatModel(responses=["unused"])),
        2,
    )
    state = Propagator().create_initial_state("BTC-USD", now.date().isoformat(), "crypto")

    result = graph.invoke(state)
    write_report_tree(result, "BTC-USD", tmp_path)

    assert received[0]["derivatives"] == history.summary().model_dump(mode="json")
    assert "funding" not in received[0]["derivatives"]
    assert (
        DerivativesHistory.model_validate_json(
            (tmp_path / "evidence" / "crypto_derivatives.json").read_text()
        )
        == history
    )
    for name in ("complete_report.md", "complete_report.html"):
        rendered = (tmp_path / name).read_text(encoding="utf-8")
        assert "fundingTime" not in rendered
        assert len(rendered) < 45000
    assert CryptoContext.model_validate_json(result["crypto_context_data"]).derivatives is None
    write_report_tree({}, "AAPL", tmp_path)
    assert not (tmp_path / "evidence" / "crypto_derivatives.json").exists()


@pytest.mark.parametrize("ticker", ["BTC-USD", "ETH-USD", "SOL-USD"])
def test_historical_analyst_never_fetches_derivatives(monkeypatch, ticker):
    def forbidden(request):
        pytest.fail("historical derivative GET")

    monkeypatch.setattr(crypto_derivatives, "collect_derivatives", forbidden)
    monkeypatch.setattr(crypto_context, "collect_derivatives", forbidden)
    state = Propagator().create_initial_state(ticker, "2000-01-01", "crypto")
    result = create_crypto_analyst(FakeListChatModel(responses=["unused"]))(state)
    assert result["crypto_derivatives_data"] == ""


def test_report_rejects_unvalidated_derivatives_sidecar(tmp_path):
    with pytest.raises(ValidationError):
        write_report_tree({"crypto_derivatives_data": '{"symbol":"BTCUSDT"}'}, "BTC-USD", tmp_path)
