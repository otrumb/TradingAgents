from datetime import UTC, date, datetime, time, timedelta

from tradingagents.dataflows.crypto_spot_models import SpotPriceContext
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import WatchRequest
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.portfolio import PortfolioContext
from tradingagents.portfolio_runtime import enrich_portfolio, render_runtime_valuation
from tradingagents.spot_decision import calculate_spot_allocation


def initialize_decisions(
    state: dict,
    portfolio: PortfolioContext | None,
    spot_contexts: tuple[SpotPriceContext, ...],
    *,
    current: bool,
) -> dict:
    ticker = state["company_of_interest"]
    valuation = (
        enrich_portfolio(portfolio, spot_contexts if current else ())
        if portfolio is not None
        else None
    )
    state["portfolio_context"] = (
        render_runtime_valuation(valuation, ticker, current) if valuation is not None else ""
    )
    state["spot_allocation_data"] = calculate_spot_allocation(
        valuation.portfolio if valuation is not None else None, ticker
    ).model_dump_json()
    analyzed = crypto_base(ticker) if state["asset_type"] == "crypto" else None
    if analyzed in {"BTC", "ETH", "SOL"}:
        cutoff = (
            None
            if current
            else datetime.combine(
                date.fromisoformat(state["trade_date"]) + timedelta(days=1), time.min, tzinfo=UTC
            )
        )
        watch = collect_watch(WatchRequest(asset=analyzed, as_of=cutoff))
        state["crypto_futures_watch_evidence"] = watch.model_dump_json()
        state["crypto_futures_watch_data"] = evaluate_watch(watch).model_dump_json()
    return state
