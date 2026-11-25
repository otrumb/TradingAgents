import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from tradingagents.agents.managers import portfolio_manager as pm
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import WatchSeries, WatchSnapshot
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio import PortfolioContext, Position
from tradingagents.reporting import write_report_tree
from tradingagents.spot_decision import SpotAllocation, SpotDecision, calculate_spot_allocation

CUTOFF = datetime(2024, 1, 3, tzinfo=UTC)


def snapshot(asset="BTC", cutoff=CUTOFF):
    return WatchSnapshot(
        asset=asset,
        symbol=f"{asset}USDT",
        cutoff=cutoff,
        as_of=cutoff,
        server_time=None,
        series=tuple(WatchSeries(interval=i) for i in ("1d", "4h", "1h")),
    )


def state_for(ticker="BTC-USD", held=0):
    state = Propagator().create_initial_state(ticker, "2024-01-02", "crypto")
    state.update(investment_plan="research", trader_investment_plan="trader")
    book = PortfolioContext(positions=(Position(ticker=ticker, quantity=held, exit_on_sell=True),))
    state["spot_allocation_data"] = calculate_spot_allocation(book, ticker).model_dump_json()
    watch = snapshot(ticker.split("-")[0])
    state["crypto_futures_watch_evidence"] = watch.model_dump_json()
    state["crypto_futures_watch_data"] = evaluate_watch(watch).model_dump_json()
    return state


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "ticker, held, expected", [("BTC-USD", 0, "NO TRADE"), ("ETH-USD", 2, "SELL")]
)
def test_manager_derives_spot_action_and_keeps_journal_rating(
    monkeypatch, tmp_path, structured, ticker, held, expected
):
    from tradingagents.memory import TradingMemoryLog

    state = state_for(ticker, held)
    decision = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="Avoid", investment_thesis="Weak"
    )
    monkeypatch.setattr(pm, "bind_structured", lambda *args: None)
    monkeypatch.setattr(pm, "invoke_structured", lambda *args: decision if structured else None)
    prompts = []

    def invoke(prompt):
        prompts.append(prompt)
        return AIMessage("**Rating**: Sell\n\nAvoid")

    node = pm.create_portfolio_manager(SimpleNamespace(invoke=invoke))
    output = node(state)
    without_watch = {**state, "crypto_futures_watch_evidence": "", "crypto_futures_watch_data": ""}
    control = node(without_watch)
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "journal.md")})
    graph = object.__new__(TradingAgentsGraph)
    graph.memory_log = log
    graph.config = {"results_dir": str(tmp_path)}
    graph.selected_analysts = []
    graph.record_decision(ticker, "2024-01-02", {**state, **output})

    action = SpotDecision.model_validate_json(output["spot_action_data"])
    assert (action.action, action.quantity) == (expected, Decimal(held))
    assert output == control
    assert output["final_rating"] == log.load_entries()[0]["rating"] == "Sell"
    assert all(
        "Futures Watch" not in prompt and "fapi.binance.com" not in prompt for prompt in prompts
    )


@pytest.mark.parametrize("asset_type, ticker", [("stock", "NVDA"), ("crypto", "DOGE-USD")])
def test_unsupported_run_never_collects_watch(monkeypatch, asset_type, ticker):
    import tradingagents.graph.trading_graph as module

    def forbidden(request):
        pytest.fail("unsupported collector called")

    monkeypatch.setattr("tradingagents.graph.run_decisions.collect_watch", forbidden)
    monkeypatch.setattr(module, "collect_spot_prices", forbidden)
    graph = object.__new__(TradingAgentsGraph)
    graph.propagator = Propagator()
    graph.resolve_instrument_context = lambda *args: ""

    state = graph.create_run_state(ticker, "2024-01-02", asset_type)

    assert state["crypto_futures_watch_evidence"] == state["crypto_futures_watch_data"] == ""


def test_historical_watch_uses_trade_day_cutoff_without_exchange_clock(monkeypatch):
    import tradingagents.graph.trading_graph as module
    from tradingagents.dataflows.crypto_spot_models import SpotPriceContext

    requests = []

    def collect(request):
        requests.append(request)
        return snapshot(request.asset, request.as_of)

    monkeypatch.setattr("tradingagents.graph.run_decisions.collect_watch", collect)
    monkeypatch.setattr(
        module,
        "collect_spot_prices",
        lambda r: SpotPriceContext(
            asset=r.asset, analysis_date=r.analysis_date, semantics="historical_finalized"
        ),
    )
    graph = object.__new__(TradingAgentsGraph)
    graph.propagator = Propagator()
    graph.resolve_instrument_context = lambda *args: ""

    state = graph.create_run_state("ETHUSDT", "2024-01-02", "crypto")

    evidence = WatchSnapshot.model_validate_json(state["crypto_futures_watch_evidence"])
    assert len(requests) == 1
    assert evidence.cutoff == evidence.as_of == CUTOFF
    assert evidence.server_time is None
    assert SpotAllocation.model_validate_json(state["spot_allocation_data"]).held_quantity is None


def completed_state(monkeypatch):
    state = state_for()
    monkeypatch.setattr(pm, "bind_structured", lambda *args: None)
    monkeypatch.setattr(pm, "invoke_structured", lambda *args: None)
    state.update(
        pm.create_portfolio_manager(
            SimpleNamespace(invoke=lambda prompt: AIMessage("**Rating**: Sell\n\nOriginal prose"))
        )(state)
    )
    return state


def test_report_summary_sidecar_and_reuse_preserve_prose(monkeypatch, tmp_path):
    state = completed_state(monkeypatch)
    state["market_report"] = "Market analysis"

    path = write_report_tree(
        state, "BTC-USD", tmp_path, settings={"data_vendors": {"prices": "configured"}}
    )
    text = path.read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "evidence/crypto_futures_watch.json").read_text())
    assert text.index("## Decision Summary") < text.index("## I. Analyst")
    assert "Market Bias: Sell" in text and "Spot Action: NO TRADE; quantity 0" in text
    assert "1d bias neutral, 4h setup neutral, 1h trigger neutral" in text
    assert "Data vendors: prices configured" in text
    assert (tmp_path / "5_portfolio/decision.md").read_text() == state["final_trade_decision"]
    assert saved == json.loads(state["crypto_futures_watch_evidence"])
    assert not {"quantity", "leverage", "stop", "target", "orders", "sizing"}.intersection(saved)

    write_report_tree({"market_report": "Stock"}, "NVDA", tmp_path)
    assert not (tmp_path / "evidence/crypto_futures_watch.json").exists()
    assert not (tmp_path / "0_summary/decision.md").exists()


@pytest.mark.parametrize(
    "field",
    [
        "spot_allocation_data",
        "spot_action_data",
        "crypto_futures_watch_evidence",
        "crypto_futures_watch_data",
    ],
)
def test_report_rejects_malformed_deterministic_state(monkeypatch, tmp_path, field):
    state = completed_state(monkeypatch)
    state[field] = '{"unexpected": 1}'

    with pytest.raises(ValidationError):
        write_report_tree(state, "BTC-USD", tmp_path)


def test_report_rejects_valid_but_mismatched_watch(monkeypatch, tmp_path):
    state = completed_state(monkeypatch)
    state["crypto_futures_watch_data"] = evaluate_watch(snapshot("ETH")).model_dump_json()

    with pytest.raises(ContextContractError):
        write_report_tree(state, "BTC-USD", tmp_path)


def test_observed_sources_require_validated_observations(monkeypatch, tmp_path):
    from tradingagents.dataflows.crypto_spot_models import IntradayPrice, SpotPriceContext
    from tradingagents.dataflows.crypto_watch_models import WatchCandle

    state = completed_state(monkeypatch)
    candle = WatchCandle(
        interval="1h",
        open_time=CUTOFF - timedelta(hours=1),
        close_time=CUTOFF - timedelta(milliseconds=1),
        open=100,
        high=110,
        low=90,
        close=105,
        volume=1,
    )
    watch = snapshot().model_copy(
        update={
            "series": (
                WatchSeries(interval="1d"),
                WatchSeries(interval="4h"),
                WatchSeries(interval="1h", candles=(candle,)),
            )
        }
    )
    state["crypto_futures_watch_evidence"] = watch.model_dump_json()
    state["crypto_futures_watch_data"] = evaluate_watch(watch).model_dump_json()
    state["crypto_market_data"] = SpotPriceContext(
        asset="BTC",
        intraday=IntradayPrice(
            asset="BTC",
            symbol="BTCUSDT",
            price=100,
            retrieved_at=CUTOFF,
            server_time=None,
        ),
    ).model_dump_json()

    path = write_report_tree(state, "BTC-USD", tmp_path)

    assert "Observed evidence: Binance Spot / Binance USD-M" in path.read_text(encoding="utf-8")
    assert "Binance Spot / Binance USD-M" in (tmp_path / "complete_report.html").read_text(
        encoding="utf-8"
    )
