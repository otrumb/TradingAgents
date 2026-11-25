import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tradingagents.dataflows.crypto_models import SourceFailure
from tradingagents.dataflows.crypto_spot_models import IntradayPrice, SpotPriceContext
from tradingagents.portfolio import PortfolioContext, load_portfolio
from tradingagents.portfolio_runtime import enrich_portfolio
from tradingagents.portfolio_sizing import calculate_sizing

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    import tradingagents.graph.trading_graph as graph_module
    import tradingagents.portfolio_runtime as module
    from tradingagents.graph.propagation import Propagator
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    class Clock(datetime):
        @staticmethod
        def now(tz):
            return NOW

    calls = []

    def collect(request):
        calls.append(request.asset)
        return (
            evidence(request.asset)
            if request.analysis_date is None
            else SpotPriceContext(
                asset=request.asset,
                analysis_date=request.analysis_date,
                semantics="historical_finalized",
            )
        )

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(graph_module, "datetime", Clock)
    monkeypatch.setattr(graph_module, "collect_spot_prices", collect)
    from tradingagents.dataflows.crypto_watch_models import WatchSeries, WatchSnapshot

    def watch(request):
        cutoff = request.as_of or NOW
        return WatchSnapshot(
            asset=request.asset,
            symbol=f"{request.asset}USDT",
            cutoff=cutoff,
            observable_at=cutoff,
            as_of=cutoff,
            server_time=None,
            series=tuple(WatchSeries(interval=i) for i in ("1d", "4h", "1h")),
        )

    monkeypatch.setattr("tradingagents.graph.run_decisions.collect_watch", watch)
    graph = object.__new__(TradingAgentsGraph)
    graph.propagator = Propagator()
    graph.resolve_instrument_context = lambda *args: ""
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.selected_analysts = ["market"]
    return graph, calls


def evidence(asset="BTC", price="100"):
    return SpotPriceContext(
        asset=asset,
        intraday=IntradayPrice(
            asset=asset,
            symbol=f"{asset}USDT",
            price=Decimal(price),
            retrieved_at=NOW,
            server_time=None,
        ),
    )


def book(**updates):
    return PortfolioContext.model_validate(
        {
            "currency": "USD",
            "cash": 9000,
            "open_orders": [],
            "positions": [
                {
                    "ticker": "BTCUSDT",
                    "quantity": 10,
                    "average_price": 500,
                    "target_weight": 0.5,
                    "max_weight": 0.6,
                    "risk_budget": 300,
                    "stop_price": 90,
                }
            ],
            **updates,
        }
    )


def test_runtime_marks_enable_exact_sizing_without_changing_input():
    original = book()
    before = original.model_dump_json()
    result = enrich_portfolio(original, (evidence(),))
    sizing = calculate_sizing(result.portfolio, "BTC-USD")

    assert (sizing.current_exposure, sizing.current_weight, sizing.additional_quantity) == (
        Decimal(1000),
        Decimal(".1"),
        Decimal(20),
    )
    assert result.portfolio.portfolio_value == 10000
    assert result.marks[0].observed_at is None
    assert result.marks[0].retrieved_at == NOW
    assert original.model_dump_json() == before


def test_graph_enriches_before_render_and_preserves_file_and_signature(runtime, tmp_path):
    graph, calls = runtime
    path = tmp_path / "portfolio.json"
    raw = book().model_dump_json()
    path.write_text(raw, encoding="utf-8")
    original = load_portfolio(path)
    signature = graph._run_signature("crypto", original)

    state = graph.create_run_state("BTC-USD", "2026-08-14", "crypto", original)

    assert "Additional long quantity (advisory only): 20" in state["portfolio_context"]
    metadata = json.loads(state["portfolio_context"].split("): ")[-1])
    assert metadata["marks"][0]["venue"] == "Binance Spot"
    assert metadata["marks"][0]["observed_at"] is None
    assert calls == ["BTC"]
    assert graph._run_signature("crypto", original) == signature
    assert path.read_text(encoding="utf-8") == raw
    assert original.positions[0].mark_price is None


@pytest.mark.parametrize("day", ["2026-08-13", "2024-01-01"])
def test_noncurrent_run_never_collects_current_marks(runtime, day):
    graph, calls = runtime
    state = graph.create_run_state("BTC-USD", day, "crypto", book())

    assert calls == ["BTC"]
    assert "Additional long quantity (advisory only)" not in state["portfolio_context"]
    metadata = json.loads(state["portfolio_context"].split("): ")[-1])
    assert metadata["marks"] == []
    assert metadata["failures"][0]["kind"] == "unavailable"


def test_aliases_collect_once_and_preserve_explicit_marks(runtime):
    graph, calls = runtime
    original = book(
        positions=[
            {"ticker": "BTCUSDT", "quantity": 1},
            {"ticker": "btc-usd", "quantity": 2},
            {"ticker": "ETHUSD", "quantity": 3},
            {"ticker": "SOL-USDT", "quantity": 4},
            {"ticker": "ETHUSDT", "quantity": 5, "mark_price": 250},
            {"ticker": "SOLUSD", "quantity": 0},
        ]
    )
    state = graph.create_run_state("BTC-USD", "2026-08-14", "crypto", original)
    result = enrich_portfolio(original, tuple(evidence(a) for a in calls))

    assert calls == ["BTC", "ETH", "SOL"]
    assert [p.mark_price for p in result.portfolio.positions] == [100, 100, 100, 100, 250, None]
    assert result.portfolio.portfolio_value == 11250
    assert "Current exposure: 300" in state["portfolio_context"]


@pytest.mark.parametrize("ticker", ["AAPL", "XRPUSDT"])
def test_unsupported_nonzero_holding_prevents_derived_value(ticker):
    result = enrich_portfolio(
        book(
            positions=[
                {"ticker": "BTCUSD", "quantity": 1},
                {"ticker": ticker, "quantity": 1},
            ]
        ),
        (evidence(),),
    )

    assert result.portfolio.portfolio_value is None
    assert result.portfolio.positions[1].mark_price is None
    assert result.failures[0].detail.startswith(ticker)


def test_explicit_value_and_mark_win_over_spot():
    result = enrich_portfolio(
        book(
            portfolio_value=50000,
            positions=[
                {"ticker": "BTCUSD", "quantity": 2, "mark_price": 700},
            ],
        ),
        (evidence(),),
    )

    assert result.portfolio.portfolio_value == 50000
    assert result.portfolio.positions[0].mark_price == 700
    assert result.marks == ()
    assert result.value_source == "caller"


@pytest.mark.parametrize("positions", [[], [{"ticker": "AAPL", "quantity": 0}]])
def test_cash_only_and_zero_holdings_derive_value_without_fetch(runtime, positions):
    graph, calls = runtime
    original = book(positions=positions)
    graph.create_run_state("BTC-USD", "2026-08-14", "crypto", original)
    result = enrich_portfolio(original, ())

    assert calls == ["BTC"]
    assert result.portfolio.portfolio_value == 9000


def test_missing_context_remains_distinct_from_flat(runtime):
    graph, calls = runtime
    absent = graph.create_run_state("BTC-USD", "2026-08-14", "crypto")
    flat = graph.create_run_state("BTC-USD", "2026-08-14", "crypto", book(positions=[]))

    assert absent["portfolio_context"] == ""
    assert flat["portfolio_context"] != ""
    assert calls == ["BTC", "BTC"]


def test_failed_intraday_never_uses_entry_or_daily_fallback():
    failure = SpotPriceContext(
        asset="BTC",
        intraday_failure=SourceFailure(
            kind="transport",
            detail="Timeout",
        ),
    )
    result = enrich_portfolio(book(), (failure,))

    assert result.portfolio.positions[0].mark_price is None
    assert result.portfolio.portfolio_value is None
    assert "Timeout" in result.failures[0].detail
    assert calculate_sizing(result.portfolio, "BTCUSD").additional_quantity is None


def test_pending_orders_reserve_capacity_without_affecting_equity():
    result = enrich_portfolio(
        book(
            open_orders=[
                {"ticker": "BTC-USD", "side": "buy", "quantity": 7},
            ]
        ),
        (evidence(),),
    )

    assert result.portfolio.portfolio_value == 10000
    assert calculate_sizing(result.portfolio, "BTCUSD").additional_quantity == Decimal(13)


@pytest.mark.parametrize("currency", [None, "EUR"])
def test_unknown_currency_never_collects_or_converts(runtime, currency):
    graph, calls = runtime
    graph.create_run_state("BTCUSD", "2026-08-14", "crypto", book(currency=currency))
    result = enrich_portfolio(book(currency=currency), (evidence(),))

    assert calls == ["BTC"]
    assert result.portfolio.positions[0].mark_price is None
    assert result.portfolio.portfolio_value is None


def test_run_evidence_reaches_both_analysts_without_recollection(runtime, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import AIMessage

    from tradingagents.agents.analysts import crypto_analyst, market_analyst
    from tradingagents.dataflows import crypto_context, crypto_spot
    from tradingagents.dataflows.crypto_models import CryptoContext

    graph, calls = runtime
    state = graph.create_run_state("BTCUSDT", "2026-08-14", "crypto", book())
    payload = state["crypto_market_data"]
    assert payload == evidence().model_dump_json()

    def forbidden(request):
        pytest.fail("analyst recollected run Spot evidence")

    monkeypatch.setattr(crypto_spot, "collect_spot_prices", forbidden)
    monkeypatch.setattr(
        crypto_analyst, "datetime", type("Clock", (), {"now": staticmethod(lambda tz: NOW)})
    )
    monkeypatch.setattr(
        crypto_context,
        "collect_crypto_context",
        lambda request: CryptoContext(asset=request.asset, snapshots=()),
    )
    received = []

    def invoke(self, messages):
        received.append(json.loads(messages[-1].content.split("\n", 2)[-1]))
        return AIMessage(content="Assessment")

    monkeypatch.setattr(FakeListChatModel, "invoke", invoke)
    monkeypatch.setattr(
        market_analyst, "take_turn", lambda *args: (AIMessage(content="Market"), "Market")
    )
    llm = FakeListChatModel(responses=["unused"])
    market = market_analyst.create_market_analyst(llm)(state)
    crypto_analyst.create_crypto_analyst(llm)(state)

    assert calls == ["BTC"]
    assert market["crypto_market_data"] == payload
    assert received[0]["spot_prices"] == evidence().summary().model_dump(mode="json")
    assert "Additional long quantity (advisory only): 20" in state["portfolio_context"]


@pytest.mark.parametrize("resume", [False, True])
def test_stock_and_resume_never_collect_spot(runtime, resume):
    graph, calls = runtime
    graph._resuming = resume

    state = graph.create_run_state(
        "BTCUSD" if resume else "AAPL", "2026-08-14", "crypto" if resume else "stock", book()
    )

    assert calls == []
    if resume:
        assert state is None
    else:
        assert state["crypto_market_data"] == ""


def test_fresh_runs_collect_new_context_without_shared_cache(runtime):
    graph, calls = runtime

    graph.create_run_state("BTCUSD", "2026-08-14", "crypto")
    state = graph.create_run_state("ETHUSD", "2026-08-14", "crypto")

    assert calls == ["BTC", "ETH"]
    assert json.loads(state["crypto_market_data"])["asset"] == "ETH"
