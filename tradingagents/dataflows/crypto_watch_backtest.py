"""Isolated historical hypothetical evaluator; no collection, persistence, or live outcomes."""

from datetime import datetime, timedelta
from decimal import Context, Decimal, localcontext
from typing import Final, assert_never

from .crypto_models import ContextContractError, FrozenModel
from .crypto_watch_evaluator import evaluate_watch
from .crypto_watch_models import (
    DURATIONS,
    EPOCH,
    HISTORICAL_OBSERVATION_LATENCY,
    WATCH_SAFETY_LAG,
    WatchDecision,
    WatchSeries,
    WatchSnapshot,
)
from .crypto_watch_scenario_models import HypotheticalOutcome, HypotheticalPlan, WatchCosts

DEFAULT_COSTS: Final = WatchCosts()


class HistoricalWatchResult(FrozenModel):
    decision: WatchDecision
    hypothetical_outcome: HypotheticalOutcome | None


def evaluate_hypothetical(
    plan: HypotheticalPlan,
    future: WatchSeries,
    as_of: datetime,
) -> HypotheticalOutcome:
    """Use finalized consecutive hourly bars; touch beats expiry, gaps price adversely."""
    if (
        as_of.tzinfo is None
        or as_of.utcoffset() is None
        or future.interval != "1h"
        or as_of < plan.observable_at
    ):
        raise ContextContractError(field="hypothetical future scope")
    duration = DURATIONS["1h"]
    offset = plan.observable_at - EPOCH
    first_open = EPOCH + ((offset + duration - timedelta(microseconds=1)) // duration) * duration
    bars = tuple(c for c in future.candles if c.open_time >= first_open and c.close_time < as_of)
    if not bars or bars[0].open_time != first_open:
        return HypotheticalOutcome(
            status="pending_entry",
            reason="next finalized 1h entry bar unavailable",
            evaluated_as_of=as_of,
        )
    entry = bars[0]
    for index, candle in enumerate(bars[:168]):
        if candle.open_time != first_open + index * duration:
            return HypotheticalOutcome(
                status="pending_outcome",
                reason="hourly history gap; outcome unknown",
                evaluated_as_of=as_of,
                entry_bar=entry.open_time,
                entry_price=entry.open,
                closed_holding_bars=index,
            )
        match plan.direction:
            case "long":
                touched = candle.low <= plan.invalidation
                exit_price = min(candle.open, plan.invalidation)
                sign = Decimal(1)
            case "short":
                touched = candle.high >= plan.invalidation
                exit_price = max(candle.open, plan.invalidation)
                sign = Decimal(-1)
            case unreachable:
                assert_never(unreachable)
        if touched or index == 167:
            if not touched:
                exit_price = candle.close
            with localcontext(Context(prec=28)):
                gross = sign * (exit_price - entry.open) / entry.open
                fee_slippage = plan.costs.round_trip_fee + 2 * plan.costs.slippage_per_side
                funding = plan.costs.funding_debit_per_8h * Decimal(index + 1) / 8
                cost = fee_slippage + funding
                net = gross - cost
            return HypotheticalOutcome(
                status="stopped" if touched else "expired",
                reason="inclusive intrabar invalidation touch"
                if touched
                else "168 closed 1h bars; no profit target",
                evaluated_as_of=as_of,
                entry_bar=entry.open_time,
                entry_price=entry.open,
                outcome_bar=candle.close_time,
                exit_price=exit_price,
                closed_holding_bars=index + 1,
                gross_return=gross,
                fee_slippage_return=fee_slippage,
                funding_return=funding,
                cost_return=cost,
                net_return=net,
            )
    return HypotheticalOutcome(
        status="pending_outcome",
        reason="fewer than 168 closed 1h bars without touch",
        evaluated_as_of=as_of,
        entry_bar=entry.open_time,
        entry_price=entry.open,
        closed_holding_bars=len(bars),
    )


def evaluate_watch_history(
    archive: WatchSnapshot,
    decision_times: tuple[datetime, ...],
    costs: WatchCosts = DEFAULT_COSTS,
) -> tuple[HistoricalWatchResult, ...]:
    """Replay observable decision stamps with lagged evidence; future bars only price outcomes."""
    if archive.as_of is None or archive.cutoff is None:
        raise ContextContractError(field="historical watch archive required")
    if any(stamp.tzinfo is None or stamp.utcoffset() is None for stamp in decision_times):
        raise ContextContractError(field="historical decision timestamp")
    if any(left >= right for left, right in zip(decision_times, decision_times[1:], strict=False)):
        raise ContextContractError(
            field="historical decision timestamps must be strictly increasing"
        )
    results = []
    for stamp in decision_times:
        if stamp.tzinfo is None or stamp.utcoffset() is None or stamp > archive.cutoff:
            raise ContextContractError(field="historical decision timestamp")
        cutoff = stamp - HISTORICAL_OBSERVATION_LATENCY - WATCH_SAFETY_LAG
        prefix = WatchSnapshot(
            asset=archive.asset,
            symbol=archive.symbol,
            as_of=cutoff,
            cutoff=cutoff,
            observable_at=stamp,
            server_time=None,
            series=tuple(
                WatchSeries(
                    interval=s.interval,
                    failure=s.failure,
                    candles=tuple(c for c in s.candles if c.close_time < cutoff)[-100:],
                )
                for s in archive.series
            ),
        )
        decision = evaluate_watch(prefix)
        outcome = None
        if decision.hypothetical is not None:
            plan = HypotheticalPlan.model_validate(
                {**decision.hypothetical.model_dump(), "costs": costs}
            )
            decision = WatchDecision.model_validate({**decision.model_dump(), "hypothetical": plan})
            outcome = evaluate_hypothetical(plan, archive.series[2], archive.cutoff)
        results.append(HistoricalWatchResult(decision=decision, hypothetical_outcome=outcome))
    return tuple(results)
