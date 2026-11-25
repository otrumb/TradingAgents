"""Immutable Binance USD-M perpetual watch evidence and decisions."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Final, Literal, Self, assert_never

from pydantic import AwareDatetime, Field, model_validator

from .crypto_models import Asset, ContextContractError, FrozenModel, SourceFailure
from .crypto_spot_models import PositivePrice, SpotSymbol
from .crypto_watch_scenario_models import HypotheticalPlan

Interval = Literal["1d", "4h", "1h"]
Direction = Literal["long", "short", "neutral"]
WaitReason = Literal[
    "unavailable",
    "insufficient_history",
    "stale",
    "neutral_bias",
    "missing_setup",
    "missing_trigger",
    "timeframe_conflict",
]
INTERVALS: Final[tuple[Interval, ...]] = ("1d", "4h", "1h")
DURATIONS: Final[dict[Interval, timedelta]] = {
    "1d": timedelta(days=1),
    "4h": timedelta(hours=4),
    "1h": timedelta(hours=1),
}
EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
REQUIRED: Final[dict[Interval, int]] = {"1d": 55, "4h": 21, "1h": 21}
FRESHNESS: Final[dict[Interval, timedelta]] = {
    "1d": timedelta(hours=48),
    "4h": timedelta(hours=8),
    "1h": timedelta(hours=2),
}


class WatchRequest(FrozenModel):
    asset: Asset
    as_of: AwareDatetime | None = None
    history_limit: int = Field(default=100, ge=1, le=1500, strict=True)
    timeout_seconds: float = Field(default=10, gt=0, le=60)


class WatchCandle(FrozenModel):
    interval: Interval
    open_time: AwareDatetime
    close_time: AwareDatetime
    open: PositivePrice
    high: PositivePrice
    low: PositivePrice
    close: PositivePrice
    volume: Annotated[Decimal, Field(ge=0)]
    finality: Literal["closed"] = "closed"

    @model_validator(mode="after")
    def valid_bar(self) -> Self:
        duration = DURATIONS[self.interval]
        if (self.open_time - EPOCH) % duration != timedelta(0):
            raise ContextContractError(field="watch interval alignment")
        if self.close_time != self.open_time + duration - timedelta(milliseconds=1):
            raise ContextContractError(field="watch close time")
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ContextContractError(field="watch OHLC")
        return self


class WatchSeries(FrozenModel):
    interval: Interval
    candles: tuple[WatchCandle, ...] = ()
    failure: SourceFailure | None = None

    @model_validator(mode="after")
    def valid_history(self) -> Self:
        times = tuple(c.open_time for c in self.candles)
        if times != tuple(sorted(set(times))):
            raise ContextContractError(field="watch chronological unique history")
        if any(c.interval != self.interval for c in self.candles):
            raise ContextContractError(field="watch interval identity")
        if self.failure is not None and self.candles:
            raise ContextContractError(field="watch failure with candles")
        return self


class WatchSnapshot(FrozenModel):
    asset: Asset
    symbol: SpotSymbol
    source: Literal["https://fapi.binance.com"] = "https://fapi.binance.com"
    market: Literal["USD-M perpetual"] = "USD-M perpetual"
    endpoint: Literal["/fapi/v1/klines"] = "/fapi/v1/klines"
    cutoff: AwareDatetime | None
    server_time: AwareDatetime | None
    as_of: AwareDatetime | None
    series: tuple[WatchSeries, ...]
    clock_failure: SourceFailure | None = None

    @model_validator(mode="after")
    def consistent_scope(self) -> Self:
        if self.symbol != f"{self.asset}USDT":
            raise ContextContractError(field="watch symbol")
        if tuple(s.interval for s in self.series) != INTERVALS:
            raise ContextContractError(field="watch timeframe coverage")
        if self.as_of is not None:
            if self.cutoff != self.as_of or self.server_time is not None or self.clock_failure:
                raise ContextContractError(field="watch historical cutoff")
        elif self.clock_failure is None:
            if self.server_time is None or self.cutoff != self.server_time - timedelta(seconds=5):
                raise ContextContractError(field="watch exchange cutoff")
        elif self.cutoff is not None or self.server_time is not None:
            raise ContextContractError(field="watch failed clock cutoff")
        if self.cutoff is None and any(s.candles for s in self.series):
            raise ContextContractError(field="watch missing cutoff")
        if self.cutoff is not None and any(
            c.close_time >= self.cutoff for s in self.series for c in s.candles
        ):
            raise ContextContractError(field="watch unfinished candle")
        return self


class WatchCondition(FrozenModel):
    interval: Interval
    name: str
    observed: Decimal
    comparison: Literal[">", "<", "<=", ">="]
    threshold: Decimal
    met: bool
    as_of: AwareDatetime


class WatchDecision(FrozenModel):
    asset: Asset
    symbol: SpotSymbol
    source: Literal["https://fapi.binance.com"] = "https://fapi.binance.com"
    as_of: AwareDatetime | None
    cutoff: AwareDatetime | None
    bias: Direction = "neutral"
    setup: Direction = "neutral"
    trigger: Direction = "neutral"
    action: Literal["consider_long", "consider_short", "wait"] = "wait"
    reasons: tuple[WaitReason, ...] = ()
    latest_closes: tuple[PositivePrice | None, ...] = (None, None, None)
    conditions: tuple[WatchCondition, ...] = ()
    hypothetical: HypotheticalPlan | None = None
    latest_close_times: tuple[AwareDatetime | None, ...] = (None, None, None)
    statuses: tuple[Literal["ok", "unavailable", "insufficient_history", "stale"], ...] = (
        "unavailable",
        "unavailable",
        "unavailable",
    )

    @model_validator(mode="after")
    def aligned_decision(self) -> Self:
        if (
            self.symbol != f"{self.asset}USDT"
            or len(self.latest_close_times) != 3
            or len(self.statuses) != 3
            or len(self.latest_closes) != 3
        ):
            raise ContextContractError(field="watch decision identity/coverage")
        match self.action:
            case "wait":
                if not self.reasons or self.hypothetical is not None:
                    raise ContextContractError(field="watch wait reasons")
                return self
            case "consider_long":
                aligned = self.bias == self.setup == self.trigger == "long"
            case "consider_short":
                aligned = self.bias == self.setup == self.trigger == "short"
            case unreachable:
                assert_never(unreachable)
        if self.reasons or not aligned or any(status != "ok" for status in self.statuses):
            raise ContextContractError(field="watch decision alignment")
        if (
            self.hypothetical is None
            or self.hypothetical.direction != self.bias
            or self.hypothetical.decision_as_of != self.as_of
        ):
            raise ContextContractError(field="watch hypothetical alignment")
        return self
