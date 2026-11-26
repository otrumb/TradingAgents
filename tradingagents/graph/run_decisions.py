from datetime import UTC, date, datetime, time, timedelta

from tradingagents.dataflows.crypto_models import SourceFailure
from tradingagents.dataflows.crypto_pattern_models import (
    CryptoPatternEvidence,
    CryptoPatternSummary,
)
from tradingagents.dataflows.crypto_pattern_runtime import classic_pattern_evidence
from tradingagents.dataflows.crypto_spot_models import SpotPriceContext
from tradingagents.dataflows.crypto_watch import collect_watch
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import WatchRequest, WatchSnapshot
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
        if state.get("crypto_futures_watch_evidence"):
            watch = WatchSnapshot.model_validate_json(state["crypto_futures_watch_evidence"])
        else:
            watch = collect_watch(WatchRequest(asset=analyzed, as_of=cutoff, history_limit=161))
            state["crypto_futures_watch_evidence"] = watch.model_dump_json()
            state["crypto_futures_watch_data"] = evaluate_watch(watch).model_dump_json()
        if not state.get("crypto_pattern_evidence"):
            spot = next((context for context in spot_contexts if context.asset == analyzed), None)
            if spot is None:
                spot = SpotPriceContext(
                    asset=watch.asset,
                    analysis_date=None if current else date.fromisoformat(state["trade_date"]),
                    semantics="current_run_only" if current else "historical_finalized",
                    daily_failure=SourceFailure(
                        kind="unavailable", detail="Run-start Spot evidence unavailable"
                    ),
                )
            # Empty Spot streams have no retrieval timestamp; only their unavailable
            # status uses the supplied watch observation or the run's analysis clock.
            observed_at = max(
                (candle.retrieved_at for candle in spot.daily_candles),
                default=(
                    spot.intraday.retrieved_at
                    if spot.intraday is not None
                    else watch.observable_at or cutoff or datetime.now(UTC)
                ),
            )
            evidence = classic_pattern_evidence(spot, watch, observed_at)
            canonical = CryptoPatternEvidence.model_validate_json(evidence.model_dump_json())
            summary = CryptoPatternSummary.model_validate_json(
                canonical.summary().model_dump_json()
            )
            state["crypto_pattern_evidence"] = canonical.model_dump_json()
            state["crypto_pattern_report"] = summary.model_dump_json()
    return state
