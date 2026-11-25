"""Hypothetical scenario counts and arithmetic means, never portfolio performance."""

from collections import Counter
from datetime import datetime
from decimal import Context, Decimal, localcontext
from heapq import heappop, heappush
from typing import Annotated, Literal, Self, assert_never

from pydantic import Field, model_validator

from .crypto_models import ContextContractError, FrozenModel
from .crypto_watch_backtest import HistoricalWatchResult

Count = Annotated[int, Field(ge=0)]


class HistoricalWatchAggregate(FrozenModel):
    scope: Literal["hypothetical/watch-only"] = "hypothetical/watch-only"
    total_decisions: Count
    unavailable: Count
    insufficient_history: Count
    stale: Count
    neutral_bias: Count
    timeframe_conflict: Count
    missing_setup: Count
    missing_trigger: Count
    aligned_plans: Count
    pending_entries: Count
    pending_outcomes: Count
    stopped: Count
    expired: Count
    completed_count: Count
    mean_gross_return: Decimal | None
    mean_fee_slippage_return: Decimal | None
    mean_funding_return: Decimal | None
    mean_cost_return: Decimal | None
    mean_net_return: Decimal | None
    overlapping_pair_count: Count
    overlap_pair_fraction: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    peak_concurrent: Count
    non_overlapping_count: Count

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if (
            self.completed_count != self.stopped + self.expired
            or self.aligned_plans
            != self.pending_entries + self.pending_outcomes + self.completed_count
            or self.aligned_plans > self.total_decisions
        ):
            raise ContextContractError(field="hypothetical aggregate counts")
        means = (
            self.mean_gross_return,
            self.mean_fee_slippage_return,
            self.mean_funding_return,
            self.mean_cost_return,
            self.mean_net_return,
        )
        if (self.completed_count == 0 and any(v is not None for v in means)) or (
            self.completed_count > 0 and any(v is None for v in means)
        ):
            raise ContextContractError(field="hypothetical aggregate completed means")
        pairs = self.completed_count * (self.completed_count - 1) // 2
        with localcontext(Context(prec=28)):
            fraction = Decimal(self.overlapping_pair_count) / pairs if pairs else Decimal(0)
        if (
            self.overlapping_pair_count > pairs
            or self.overlap_pair_fraction != fraction
            or not 0 <= self.non_overlapping_count <= self.completed_count
            or not 0 <= self.peak_concurrent <= self.completed_count
            or (
                self.completed_count > 0
                and min(self.non_overlapping_count, self.peak_concurrent) == 0
            )
        ):
            raise ContextContractError(field="hypothetical aggregate overlap metrics")
        return self


def aggregate_watch_history(results: tuple[HistoricalWatchResult, ...]) -> HistoricalWatchAggregate:
    """Average completed scenarios equally with Decimal precision 28; pending contributes no return."""
    pending_entries = pending_outcomes = stopped = expired = aligned = 0
    gross: list[Decimal] = []
    costs: list[Decimal] = []
    net: list[Decimal] = []
    fee_slippage: list[Decimal] = []
    funding: list[Decimal] = []
    waits = Counter(reason for result in results for reason in set(result.decision.reasons))
    intervals: list[tuple[datetime, datetime]] = []
    for result in results:
        outcome = result.hypothetical_outcome
        if (result.decision.hypothetical is None) != (outcome is None):
            raise ContextContractError(field="hypothetical aggregate plan/outcome")
        if outcome is None:
            continue
        aligned += 1
        match outcome.status:
            case "pending_entry":
                pending_entries += 1
            case "pending_outcome":
                pending_outcomes += 1
            case "stopped" | "expired":
                stopped += outcome.status == "stopped"
                expired += outcome.status == "expired"
                if (
                    outcome.gross_return is None
                    or outcome.fee_slippage_return is None
                    or outcome.funding_return is None
                    or outcome.cost_return is None
                    or outcome.net_return is None
                    or outcome.entry_bar is None
                    or outcome.outcome_bar is None
                ):
                    raise ContextContractError(field="hypothetical completed returns")
                gross.append(outcome.gross_return)
                fee_slippage.append(outcome.fee_slippage_return)
                funding.append(outcome.funding_return)
                costs.append(outcome.cost_return)
                net.append(outcome.net_return)
                intervals.append((outcome.entry_bar, outcome.outcome_bar))
            case unreachable:
                assert_never(unreachable)
    completed = stopped + expired
    active: list[datetime] = []
    overlapping_pairs = peak = non_overlapping = 0
    for start, end in sorted(intervals):
        while active and active[0] < start:
            heappop(active)
        overlapping_pairs += len(active)
        heappush(active, end)
        peak = max(peak, len(active))
    last_end = None
    for start, end in sorted(intervals, key=lambda interval: (interval[1], interval[0])):
        if last_end is None or start > last_end:
            non_overlapping += 1
            last_end = end
    pairs = completed * (completed - 1) // 2
    with localcontext(Context(prec=28)):
        return HistoricalWatchAggregate(
            total_decisions=len(results),
            unavailable=waits["unavailable"],
            insufficient_history=waits["insufficient_history"],
            stale=waits["stale"],
            neutral_bias=waits["neutral_bias"],
            timeframe_conflict=waits["timeframe_conflict"],
            missing_setup=waits["missing_setup"],
            missing_trigger=waits["missing_trigger"],
            aligned_plans=aligned,
            pending_entries=pending_entries,
            pending_outcomes=pending_outcomes,
            stopped=stopped,
            expired=expired,
            completed_count=completed,
            mean_gross_return=sum(gross, Decimal(0)) / completed if completed else None,
            mean_fee_slippage_return=sum(fee_slippage, Decimal(0)) / completed
            if completed
            else None,
            mean_funding_return=sum(funding, Decimal(0)) / completed if completed else None,
            mean_cost_return=sum(costs, Decimal(0)) / completed if completed else None,
            mean_net_return=sum(net, Decimal(0)) / completed if completed else None,
            overlapping_pair_count=overlapping_pairs,
            overlap_pair_fraction=Decimal(overlapping_pairs) / pairs if pairs else Decimal(0),
            peak_concurrent=peak,
            non_overlapping_count=non_overlapping,
        )


def render_watch_aggregate(summary: HistoricalWatchAggregate) -> str:
    counts = (
        f"Hypothetical/watch-only: decisions={summary.total_decisions}; aligned={summary.aligned_plans}; "
        f"pending_entries={summary.pending_entries}; pending_outcomes={summary.pending_outcomes}; "
        f"stopped={summary.stopped}; expired={summary.expired}; completed={summary.completed_count}."
        f" Wait reason counts (may co-occur): unavailable={summary.unavailable}; "
        f"insufficient_history={summary.insufficient_history}; stale={summary.stale}; "
        f"neutral_bias={summary.neutral_bias}; timeframe_conflict={summary.timeframe_conflict}; "
        f"missing_setup={summary.missing_setup}; missing_trigger={summary.missing_trigger}."
        f" Completed closed-interval overlap: pairs={summary.overlapping_pair_count}; "
        f"pair_fraction={summary.overlap_pair_fraction}; peak_concurrent={summary.peak_concurrent}; "
        f"earliest_finish_non_overlapping={summary.non_overlapping_count}."
        " All completed scenarios enter descriptive means equally. Dependent hypothetical scenarios, "
        "not independent observations or portfolio performance; no performance claim."
    )
    if summary.completed_count == 0:
        return counts + " No completed scenarios; return means N/A; no performance claim."
    return (
        counts + f" Completed scenario means: gross={summary.mean_gross_return}; "
        f"fee+slippage={summary.mean_fee_slippage_return}; funding={summary.mean_funding_return}; "
        f"total cost={summary.mean_cost_return}; net={summary.mean_net_return}."
    )
