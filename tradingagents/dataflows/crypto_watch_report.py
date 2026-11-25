"""Compact watch diagnostics and hypothetical-only scenario reporting."""

from .crypto_watch_models import WatchDecision
from .crypto_watch_scenario_models import HypotheticalOutcome


def render_watch_details(decision: WatchDecision) -> str:
    observable = (
        decision.observable_at.isoformat() if decision.observable_at is not None else "unavailable"
    )
    lines = [
        "Hypothetical/watch-only; no executable instructions.",
        f"Evidence cutoff={decision.cutoff}; observable_at={observable}.",
    ]
    for interval, close, stamp, status in zip(
        ("1d", "4h", "1h"),
        decision.latest_closes,
        decision.latest_close_times,
        decision.statuses,
        strict=True,
    ):
        lines.append(f"{interval}: close={close}; as_of={stamp}; status={status}")
    for condition in decision.conditions:
        lines.append(
            f"{condition.interval} {condition.name}: {condition.observed} "
            f"{condition.comparison} {condition.threshold}; "
            f"{'met' if condition.met else 'MISSING'}; as_of={condition.as_of.isoformat()}"
        )
    plan = decision.hypothetical
    if plan is None:
        lines.append("Hypothetical entry: none (waiting). Invalidation: N/A.")
        return "\n".join(lines)
    costs = plan.costs
    lines.extend(
        (
            f"Hypothetical {plan.direction}: pending_entry; {plan.entry_rule}; price pending.",
            f"Hypothetical invalidation: {plan.invalidation}; {plan.invalidation_rule}; "
            f"swing={plan.swing_start.isoformat()}..{plan.swing_end.isoformat()}; "
            f"confirmed_at={plan.confirmed_at.isoformat()}; decision_as_of={plan.decision_as_of.isoformat()}; "
            f"observable_at={plan.observable_at.isoformat()}.",
            f"Hypothetical outcome rule: {plan.stop_rule}; expiry=168 closed 1h bars; no profit target.",
            f"Hypothetical costs: round_trip_fee={costs.round_trip_fee}; "
            f"slippage_per_side={costs.slippage_per_side}; funding_debit_per_8h={costs.funding_debit_per_8h}; "
            f"{costs.funding_rule}; {costs.provenance}.",
        )
    )
    return "\n".join(lines)


def render_hypothetical_outcome(outcome: HypotheticalOutcome) -> str:
    return (
        f"Hypothetical/watch-only: {outcome.status}; {outcome.reason}; "
        f"evaluated_as_of={outcome.evaluated_as_of.isoformat()}; "
        f"entry={outcome.entry_price} at {outcome.entry_bar}; "
        f"exit={outcome.exit_price} at {outcome.outcome_bar}; "
        f"closed_bars={outcome.closed_holding_bars}; gross={outcome.gross_return}; "
        f"fee+slippage={outcome.fee_slippage_return}; funding={outcome.funding_return}; "
        f"total cost={outcome.cost_return}; net={outcome.net_return}"
    )
