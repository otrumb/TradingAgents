"""Hypothetical scenario counts and arithmetic means, never portfolio performance."""

from decimal import Context, Decimal, localcontext
from typing import Annotated, Literal, Self, assert_never

from pydantic import Field, model_validator

from .crypto_models import ContextContractError, FrozenModel
from .crypto_watch_backtest import HistoricalWatchResult

Count = Annotated[int, Field(ge=0)]


class HistoricalWatchAggregate(FrozenModel):
    scope: Literal["hypothetical/watch-only"] = "hypothetical/watch-only"
    total_decisions: Count
    aligned_plans: Count
    pending_entries: Count
    pending_outcomes: Count
    stopped: Count
    expired: Count
    completed_count: Count
    mean_gross_return: Decimal | None
    mean_cost_return: Decimal | None
    mean_net_return: Decimal | None

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if (
            self.completed_count != self.stopped + self.expired
            or self.aligned_plans
            != self.pending_entries + self.pending_outcomes + self.completed_count
            or self.aligned_plans > self.total_decisions
        ):
            raise ContextContractError(field="hypothetical aggregate counts")
        means = (self.mean_gross_return, self.mean_cost_return, self.mean_net_return)
        if (self.completed_count == 0 and any(v is not None for v in means)) or (
            self.completed_count > 0 and any(v is None for v in means)
        ):
            raise ContextContractError(field="hypothetical aggregate completed means")
        return self


def aggregate_watch_history(results: tuple[HistoricalWatchResult, ...]) -> HistoricalWatchAggregate:
    """Average completed scenarios equally with Decimal precision 28; pending contributes no return."""
    pending_entries = pending_outcomes = stopped = expired = aligned = 0
    gross: list[Decimal] = []
    costs: list[Decimal] = []
    net: list[Decimal] = []
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
                    or outcome.cost_return is None
                    or outcome.net_return is None
                ):
                    raise ContextContractError(field="hypothetical completed returns")
                gross.append(outcome.gross_return)
                costs.append(outcome.cost_return)
                net.append(outcome.net_return)
            case unreachable:
                assert_never(unreachable)
    completed = stopped + expired
    with localcontext(Context(prec=28)):
        return HistoricalWatchAggregate(
            total_decisions=len(results),
            aligned_plans=aligned,
            pending_entries=pending_entries,
            pending_outcomes=pending_outcomes,
            stopped=stopped,
            expired=expired,
            completed_count=completed,
            mean_gross_return=sum(gross, Decimal(0)) / completed if completed else None,
            mean_cost_return=sum(costs, Decimal(0)) / completed if completed else None,
            mean_net_return=sum(net, Decimal(0)) / completed if completed else None,
        )


def render_watch_aggregate(summary: HistoricalWatchAggregate) -> str:
    counts = (
        f"Hypothetical/watch-only: decisions={summary.total_decisions}; aligned={summary.aligned_plans}; "
        f"pending_entries={summary.pending_entries}; pending_outcomes={summary.pending_outcomes}; "
        f"stopped={summary.stopped}; expired={summary.expired}; completed={summary.completed_count}."
    )
    if summary.completed_count == 0:
        return counts + " No completed scenarios; return means N/A; no performance claim."
    return (
        counts + f" Completed scenario means: gross={summary.mean_gross_return}; "
        f"cost={summary.mean_cost_return}; net={summary.mean_net_return}."
    )
