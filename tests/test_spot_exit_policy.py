import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from tradingagents.agents.managers import portfolio_manager as pm
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.run_decisions import initialize_decisions
from tradingagents.portfolio import PortfolioContext, Position, load_portfolio
from tradingagents.spot_decision import SpotAllocation, SpotDecision


@pytest.mark.parametrize("policy", [True, False])
def test_supplied_policy_changes_fingerprint_but_omitted_policy_keeps_legacy(policy: bool) -> None:
    raw = '{"cash":null,"currency":null,"positions":[{"ticker":"ETHUSDT","quantity":2.0,"average_price":null}]}'
    legacy = PortfolioContext.model_validate_json(raw)
    supplied = PortfolioContext(
        positions=(Position(ticker="ETHUSDT", quantity=2, exit_on_sell=policy),)
    )

    assert legacy.fingerprint() == hashlib.sha256(raw.encode()).hexdigest()[:12]
    assert supplied.fingerprint() != legacy.fingerprint()
    assert supplied.positions[0].exit_on_sell is policy


@pytest.mark.parametrize("policy", [None, "true", "false", 0, 1, [], {}])
def test_position_rejects_non_boolean_exit_policy(policy) -> None:
    with pytest.raises(ValidationError):
        Position(ticker="ETHUSDT", quantity=2, exit_on_sell=policy)


@pytest.mark.parametrize("policy", [True, False])
def test_graph_enrichment_carries_exit_authorization_without_writing_file(tmp_path, policy):
    from datetime import UTC, datetime

    from tradingagents.dataflows.crypto_spot_models import IntradayPrice, SpotPriceContext

    path = tmp_path / "synthetic-portfolio.json"
    raw = json.dumps(
        {
            "currency": "USD",
            "cash": 600,
            "positions": [
                {"ticker": "ETHUSDT", "quantity": 2, "exit_on_sell": policy},
            ],
        }
    )
    path.write_text(raw, encoding="utf-8")
    original = load_portfolio(path)
    before = original.model_dump_json()
    state = Propagator().create_initial_state("ETHUSD", "2024-01-02", "stock")
    evidence = SpotPriceContext(
        asset="ETH",
        intraday=IntradayPrice(
            asset="ETH",
            symbol="ETHUSDT",
            price=200,
            retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
            server_time=None,
        ),
    )

    result = initialize_decisions(state, original, (evidence,), current=True)

    allocation = SpotAllocation.model_validate_json(result["spot_allocation_data"])
    assert allocation.exit_on_sell is policy
    assert allocation.current_exposure == Decimal(400)
    assert json.loads(result["spot_allocation_data"])["exit_on_sell"] is policy
    assert "sizing" not in json.loads(result["spot_allocation_data"])
    assert original.model_dump_json() == before
    assert path.read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "policy, expected", [(None, "NO TRADE"), (False, "NO TRADE"), (True, "SELL")]
)
def test_manager_sell_obeys_serialized_authorization(monkeypatch, structured, policy, expected):
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

    state = Propagator().create_initial_state("ETHUSD", "2024-01-02", "stock")
    state.update(investment_plan="research", trader_investment_plan="trader")
    allocation = {"ticker": "ETH-USD", "held_quantity": "2"}
    if policy is not None:
        allocation["exit_on_sell"] = policy
    state["spot_allocation_data"] = json.dumps(allocation)
    decision = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="Avoid", investment_thesis="Weak"
    )
    monkeypatch.setattr(pm, "bind_structured", lambda *args: None)
    monkeypatch.setattr(pm, "invoke_structured", lambda *args: decision if structured else None)
    node = pm.create_portfolio_manager(
        SimpleNamespace(invoke=lambda prompt: AIMessage("**Rating**: Sell"))
    )

    result = node(state)

    action = SpotDecision.model_validate_json(result["spot_action_data"])
    assert (action.action, action.quantity) == (expected, Decimal(2 if policy else 0))
    assert result["final_rating"] == "Sell"


@pytest.mark.parametrize(
    "payload",
    ['{"ticker":"ETH-USD","held_quantity":"2","exit_on_sell":"true"}', '{"unexpected":1}'],
)
def test_manager_rejects_malformed_allocation(monkeypatch, payload):
    state = Propagator().create_initial_state("ETHUSD", "2024-01-02", "stock")
    state.update(
        investment_plan="research", trader_investment_plan="trader", spot_allocation_data=payload
    )
    monkeypatch.setattr(pm, "bind_structured", lambda *args: None)
    monkeypatch.setattr(pm, "invoke_structured", lambda *args: None)
    node = pm.create_portfolio_manager(
        SimpleNamespace(invoke=lambda prompt: AIMessage("**Rating**: Sell"))
    )

    with pytest.raises(ValidationError):
        node(state)


def test_manager_missing_allocation_does_not_invent_trade(monkeypatch):
    state = Propagator().create_initial_state("ETHUSD", "2024-01-02", "stock")
    state.update(investment_plan="research", trader_investment_plan="trader")
    monkeypatch.setattr(pm, "bind_structured", lambda *args: None)
    monkeypatch.setattr(pm, "invoke_structured", lambda *args: None)
    node = pm.create_portfolio_manager(
        SimpleNamespace(invoke=lambda prompt: AIMessage("**Rating**: Sell"))
    )

    result = node(state)

    assert result["spot_action_data"] == ""
