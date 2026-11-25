"""Canonical hypothetical watch-only scenario and cost provenance; never orders."""

from decimal import Decimal
from typing import Annotated, Literal, Self, assert_never

from pydantic import AwareDatetime, Field, model_validator

from .crypto_models import ContextContractError, FrozenModel
from .crypto_spot_models import PositivePrice

Rate = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]


class WatchCosts(FrozenModel):
    """Assumed fractions of entry notional, not exchange quotes or funding history."""

    provenance: Literal["explicit hypothetical assumptions; entry-notional fractions"] = (
        "explicit hypothetical assumptions; entry-notional fractions"
    )
    round_trip_fee: Rate = Decimal("0.001")
    slippage_per_side: Rate = Decimal("0.0005")
    funding_debit_per_8h: Rate = Decimal("0.0001")
    funding_rule: Literal["prorated closed holding bars / 8; debit both directions"] = (
        "prorated closed holding bars / 8; debit both directions"
    )


class HypotheticalPlan(FrozenModel):
    scope: Literal["hypothetical/watch-only"] = "hypothetical/watch-only"
    status: Literal["pending_entry"] = "pending_entry"
    direction: Literal["long", "short"]
    decision_as_of: AwareDatetime
    entry_rule: Literal["next finalized 1h candle open at or after decision_as_of"] = (
        "next finalized 1h candle open at or after decision_as_of"
    )
    invalidation: PositivePrice
    swing_start: AwareDatetime
    swing_end: AwareDatetime
    confirmed_at: AwareDatetime
    invalidation_rule: Literal["extreme of prior three closed 4h pullback bars"] = (
        "extreme of prior three closed 4h pullback bars"
    )
    stop_rule: Literal["intrabar inclusive touch; adverse gap uses candle open"] = (
        "intrabar inclusive touch; adverse gap uses candle open"
    )
    expiry_closed_1h_bars: Literal[168] = 168
    profit_target: None = None
    costs: WatchCosts = WatchCosts()

    @model_validator(mode="after")
    def causal_swing(self) -> Self:
        if not self.swing_start <= self.swing_end < self.confirmed_at < self.decision_as_of:
            raise ContextContractError(field="hypothetical swing provenance")
        return self


class HypotheticalOutcome(FrozenModel):
    scope: Literal["hypothetical/watch-only"] = "hypothetical/watch-only"
    status: Literal["pending_entry", "pending_outcome", "stopped", "expired"]
    reason: str
    evaluated_as_of: AwareDatetime
    entry_bar: AwareDatetime | None = None
    entry_price: PositivePrice | None = None
    outcome_bar: AwareDatetime | None = None
    exit_price: PositivePrice | None = None
    closed_holding_bars: Annotated[int, Field(ge=0, le=168)] = 0
    gross_return: Decimal | None = None
    cost_return: Decimal | None = None
    net_return: Decimal | None = None

    @model_validator(mode="after")
    def coherent_outcome(self) -> Self:
        entered = self.entry_bar is not None and self.entry_price is not None
        completed = all(
            value is not None
            for value in (
                self.outcome_bar,
                self.exit_price,
                self.gross_return,
                self.cost_return,
                self.net_return,
            )
        )
        match self.status:
            case "pending_entry":
                valid = (
                    self.entry_bar is None
                    and self.entry_price is None
                    and self.closed_holding_bars == 0
                )
            case "pending_outcome":
                valid = entered and 0 < self.closed_holding_bars < 168
            case "stopped":
                valid = entered and completed and self.closed_holding_bars > 0
            case "expired":
                valid = entered and completed and self.closed_holding_bars == 168
            case unreachable:
                assert_never(unreachable)
        if not valid:
            raise ContextContractError(field="hypothetical outcome state")
        if self.status in ("pending_entry", "pending_outcome") and any(
            value is not None
            for value in (
                self.outcome_bar,
                self.exit_price,
                self.gross_return,
                self.cost_return,
                self.net_return,
            )
        ):
            raise ContextContractError(field="hypothetical pending returns")
        return self
