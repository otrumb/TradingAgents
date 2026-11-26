"""Validate deterministic state and render its report summary."""

from tradingagents.agents.rating import run_rating
from tradingagents.crypto_pattern_report import pattern_sources, validate_patterns
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_spot_models import SpotPriceContext
from tradingagents.dataflows.crypto_watch_evaluator import evaluate_watch
from tradingagents.dataflows.crypto_watch_models import WatchDecision, WatchSnapshot
from tradingagents.dataflows.crypto_watch_report import render_watch_details
from tradingagents.dataflows.symbols import crypto_base, normalize_symbol
from tradingagents.spot_decision import SpotAllocation, SpotDecision, derive_spot_action


def decision_summary(state: dict) -> str:
    allocation = (
        SpotAllocation.model_validate_json(state["spot_allocation_data"])
        if state.get("spot_allocation_data")
        else None
    )
    action = (
        SpotDecision.model_validate_json(state["spot_action_data"])
        if state.get("spot_action_data")
        else None
    )
    snapshot = (
        WatchSnapshot.model_validate_json(state["crypto_futures_watch_evidence"])
        if state.get("crypto_futures_watch_evidence")
        else None
    )
    watch = (
        WatchDecision.model_validate_json(state["crypto_futures_watch_data"])
        if state.get("crypto_futures_watch_data")
        else None
    )
    if allocation is not None and (
        allocation.ticker != normalize_symbol(state["company_of_interest"])
        or (state.get("final_rating") and action is None)
    ):
        raise ContextContractError(field="Spot allocation identity/completion")
    if action is not None and (
        action.allocation != allocation
        or action.market_bias != run_rating(state)
        or action != derive_spot_action(action.allocation, action.market_bias)
    ):
        raise ContextContractError(field="Spot decision consistency")
    if (snapshot is None) != (watch is None) or (
        snapshot is not None
        and (
            watch != evaluate_watch(snapshot)
            or snapshot.asset != crypto_base(state["company_of_interest"])
            or state.get("asset_type") != "crypto"
        )
    ):
        raise ContextContractError(field="futures watch evidence/decision consistency")
    if allocation is None and action is None and watch is None:
        return ""
    lines = [f"- Market Bias: {run_rating(state)}"]
    if allocation is not None:
        held = allocation.held_quantity
        lines.append(
            f"- Spot Position: {allocation.ticker}, {held if held is not None else 'unknown'} units"
        )
    lines.append(
        f"- Spot Action: {action.action}; quantity {action.quantity}"
        if action is not None
        else "- Spot Action: unavailable"
    )
    if watch is not None:
        lines.append(
            f"- Futures Watch: {watch.action}; 1d bias {watch.bias}, "
            f"4h setup {watch.setup}, 1h trigger {watch.trigger}"
        )
        lines.append("- Watch status (1d/4h/1h): " + "/".join(watch.statuses))
        if watch.reasons:
            lines.append("- Watch reasons: " + ", ".join(watch.reasons))
        lines.append(render_watch_details(watch))
    else:
        lines.append("- Futures Watch 1d/4h/1h: unavailable")
    observed = observed_sources(state, include_patterns=False)
    if observed:
        lines.append("- Observed evidence: " + observed)
    return "\n".join(lines)


def observed_sources(state: dict, *, include_patterns: bool = True) -> str:
    sources = []
    if state.get("crypto_market_data"):
        spot = SpotPriceContext.model_validate_json(state["crypto_market_data"])
        if spot.daily_candles or spot.intraday is not None:
            sources.append("Binance Spot")
    if state.get("crypto_futures_watch_evidence"):
        watch = WatchSnapshot.model_validate_json(state["crypto_futures_watch_evidence"])
        if any(series.candles for series in watch.series):
            sources.append("Binance USD-M")
    if include_patterns:
        sources.extend(pattern_sources(validate_patterns(state)))
    return " / ".join(dict.fromkeys(sources))
