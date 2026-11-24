"""Canonical Binance USD-M historical evidence and bounded feature summaries."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

Symbol = Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class DerivativesContractError(ValueError):
    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"Invalid derivatives evidence: {field}")


def utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


Timestamp = Annotated[AwareDatetime, AfterValidator(utc)]
Milliseconds = Annotated[StrictInt, Field(gt=0, le=253402300799999)]


class HistoricalModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)


class FundingWire(HistoricalModel):
    symbol: Symbol
    fundingTime: Milliseconds
    fundingRate: StrictStr
    markPrice: StrictStr | None = None
    rateType: StrictStr | None = None


class InterestWire(HistoricalModel):
    symbol: Symbol
    timestamp: Milliseconds
    sumOpenInterest: StrictStr
    sumOpenInterestValue: StrictStr
    CMCCirculatingSupply: StrictStr | None = None


class FundingObservation(HistoricalModel):
    symbol: Symbol
    observed_at: Timestamp
    rate: Decimal


class InterestObservation(HistoricalModel):
    symbol: Symbol
    observed_at: Timestamp
    contracts: Decimal = Field(ge=0)
    notional_usdt: Decimal = Field(ge=0)


class ObservedWindow(HistoricalModel):
    requested_start: Timestamp
    requested_end: Timestamp
    observed_start: Timestamp | None
    observed_end: Timestamp | None
    sample_count: int = Field(ge=0)
    actual_seconds: Decimal | None

    @model_validator(mode="after")
    def consistent_bounds(self) -> Self:
        if self.requested_start >= self.requested_end:
            raise DerivativesContractError("requested bounds")
        if self.sample_count == 0:
            if any(
                v is not None for v in (self.observed_start, self.observed_end, self.actual_seconds)
            ):
                raise DerivativesContractError("empty window bounds")
        elif self.observed_start is None or self.observed_end is None:
            raise DerivativesContractError("missing observed bounds")
        elif (
            not self.requested_start
            <= self.observed_start
            <= self.observed_end
            <= self.requested_end
        ):
            raise DerivativesContractError("observed bounds")
        elif self.actual_seconds != Decimal(
            str((self.observed_end - self.observed_start).total_seconds())
        ):
            raise DerivativesContractError("actual span")
        return self


class HistoricalFeature(HistoricalModel):
    source: Literal["Binance USD-M Futures"] = "Binance USD-M Futures"
    symbol: Symbol
    name: str
    value: Decimal | None
    unit: Literal["rate", "contracts", "percent"]
    requested_horizon_seconds: int = Field(gt=0)
    windows: tuple[ObservedWindow, ...]
    freshness: Literal["fresh", "stale", "unknown"]
    coverage: Literal["sufficient", "insufficient", "unavailable"]
    status: Literal["ok", "insufficient", "unavailable"]
    reason: str | None
    baseline_target: Timestamp | None = None
    baseline_deviation_seconds: Decimal | None = None

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        available = self.status == "ok"
        if available != (self.value is not None) or available != (self.reason is None):
            raise DerivativesContractError("feature value/status/reason")
        if self.coverage != ("sufficient" if available else self.status):
            raise DerivativesContractError("feature coverage")
        if available and self.freshness != "fresh":
            raise DerivativesContractError("available feature freshness")
        return self


class DerivativesSummary(HistoricalModel):
    symbol: Symbol
    retrieved_at: Timestamp
    features: tuple[HistoricalFeature, ...]
    funding_failure: str | None = None
    interest_failure: str | None = None


class DerivativesHistory(DerivativesSummary):
    source: Literal["Binance USD-M Futures"] = "Binance USD-M Futures"
    funding_endpoint: Literal["https://fapi.binance.com/fapi/v1/fundingRate"] = (
        "https://fapi.binance.com/fapi/v1/fundingRate"
    )
    interest_endpoint: Literal["https://fapi.binance.com/futures/data/openInterestHist"] = (
        "https://fapi.binance.com/futures/data/openInterestHist"
    )
    funding: tuple[FundingObservation, ...] = ()
    interest: tuple[InterestObservation, ...] = ()

    @model_validator(mode="after")
    def canonical_history(self) -> Self:
        for rows in (self.funding, self.interest):
            times = tuple(row.observed_at for row in rows)
            if times != tuple(sorted(set(times))):
                raise DerivativesContractError("history ordering/duplicates")
            if any(
                row.symbol != self.symbol or row.observed_at > self.retrieved_at for row in rows
            ):
                raise DerivativesContractError("history symbol/future timestamp")
        if any(feature.symbol != self.symbol for feature in self.features):
            raise DerivativesContractError("feature symbol")
        if (self.funding_failure and self.funding) or (self.interest_failure and self.interest):
            raise DerivativesContractError("failure with observations")
        return self

    def summary(self) -> DerivativesSummary:
        return DerivativesSummary(
            symbol=self.symbol,
            retrieved_at=self.retrieved_at,
            features=self.features,
            funding_failure=self.funding_failure,
            interest_failure=self.interest_failure,
        )
