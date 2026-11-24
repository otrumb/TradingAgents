"""Binance Spot evidence: completed UTC days and separate current price."""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, StrictStr, model_validator

from .crypto_models import (
    Asset,
    ContextContractError,
    CryptoContext,
    FrozenModel,
    SnapshotRequest,
    SourceFailure,
)

SpotSymbol = Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PositivePrice = Annotated[Decimal, Field(gt=0)]
Milliseconds = Annotated[StrictInt, Field(ge=0)]


def utc_milliseconds(value: int) -> datetime:
    """Convert integer exchange milliseconds without float rounding."""
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)


class SpotPriceRequest(SnapshotRequest):
    """Absent date selects current evidence; dated requests end at UTC day close."""

    analysis_date: date | None = None


class SpotObservation(FrozenModel):
    asset: Asset
    symbol: SpotSymbol
    source: Literal["https://data-api.binance.vision"] = "https://data-api.binance.vision"
    provider: Literal["binance"] = "binance"
    market: Literal["spot"] = "spot"
    base_url: Literal["https://data-api.binance.vision"] = "https://data-api.binance.vision"
    venue: Literal["Binance Spot"] = "Binance Spot"
    quote_currency: Literal["USDT"] = "USDT"
    retrieved_at: AwareDatetime
    server_time: AwareDatetime | None

    @model_validator(mode="after")
    def identity_and_utc(self) -> Self:
        if self.symbol != f"{self.asset}USDT":
            raise ContextContractError(field="spot symbol")
        for stamp in (self.retrieved_at, self.server_time):
            if stamp is not None and stamp.utcoffset() != timedelta(0):
                raise ContextContractError(field="spot UTC timestamp")
        return self


class CompletedDailyCandle(SpotObservation):
    endpoint: Literal["/api/v3/klines"] = "/api/v3/klines"
    data_source: Literal["Database"] = "Database"
    interval: Literal["1d"] = "1d"
    timezone: Literal["0"] = "0"
    finality: Literal["closed"] = "closed"
    finality_lag_ms: Literal[0, 5000] = 5000
    analysis_date: date | None = None
    cutoff: AwareDatetime
    open_time: AwareDatetime
    close_time: AwareDatetime
    open: PositivePrice
    high: PositivePrice
    low: PositivePrice
    close: PositivePrice
    volume: Annotated[Decimal, Field(ge=0)]

    @model_validator(mode="after")
    def completed_utc_day(self) -> Self:
        stamps = (self.cutoff, self.open_time, self.close_time)
        if any(stamp.utcoffset() != timedelta(0) for stamp in stamps):
            raise ContextContractError(field="daily UTC timestamps")
        if self.analysis_date is None:
            if (
                self.server_time is None
                or self.finality_lag_ms != 5000
                or self.cutoff != self.server_time - timedelta(milliseconds=5000)
            ):
                raise ContextContractError(field="daily finality cutoff")
        else:
            expected = datetime.combine(self.analysis_date, time.min, UTC) + timedelta(days=1)
            if self.server_time is not None or self.finality_lag_ms != 0 or self.cutoff != expected:
                raise ContextContractError(field="historical daily cutoff")
        if self.close_time >= self.cutoff:
            raise ContextContractError(field="daily candle not eligible")
        if self.open_time != self.open_time.replace(hour=0, minute=0, second=0, microsecond=0):
            raise ContextContractError(field="daily UTC midnight")
        if self.close_time != self.open_time + timedelta(days=1, milliseconds=-1):
            raise ContextContractError(field="daily final interval millisecond")
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ContextContractError(field="daily OHLC range")
        return self


class IntradayPrice(SpotObservation):
    endpoint: Literal["/api/v3/ticker/price"] = "/api/v3/ticker/price"
    data_source: Literal["Memory"] = "Memory"
    finality: Literal["intraday_not_daily_close"] = "intraday_not_daily_close"
    price: PositivePrice
    # Binance ticker/price supplies no source observation timestamp.
    observed_at: None = None


class SpotPriceContext(FrozenModel):
    asset: Asset
    semantics: Literal["current_run_only", "historical_finalized"] = "current_run_only"
    analysis_date: date | None = None
    daily_candles: tuple[CompletedDailyCandle, ...] = ()
    intraday: IntradayPrice | None = None
    daily_failure: SourceFailure | None = None
    intraday_failure: SourceFailure | None = None

    def summary(self) -> "SpotPriceSummary":
        """Keep finality and latest OHLCV, without duplicating candle history."""
        return SpotPriceSummary(
            asset=self.asset,
            semantics=self.semantics,
            analysis_date=self.analysis_date,
            daily_row_count=len(self.daily_candles),
            daily_start=self.daily_candles[0].open_time if self.daily_candles else None,
            daily_end=self.daily_candles[-1].close_time if self.daily_candles else None,
            latest_finalized=self.daily_candles[-1] if self.daily_candles else None,
            intraday=self.intraday,
            daily_failure=self.daily_failure,
            intraday_failure=self.intraday_failure,
        )

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        expected = "current_run_only" if self.analysis_date is None else "historical_finalized"
        if self.semantics != expected:
            raise ContextContractError(field="spot analysis scope")
        if self.analysis_date is not None and (
            self.intraday is not None or self.intraday_failure is not None
        ):
            raise ContextContractError(field="historical intraday evidence")
        if any(c.analysis_date != self.analysis_date for c in self.daily_candles):
            raise ContextContractError(field="daily analysis scope")
        if self.daily_failure is not None and self.daily_candles:
            raise ContextContractError(field="daily failure with evidence")
        if self.intraday_failure is not None and self.intraday is not None:
            raise ContextContractError(field="intraday failure with evidence")
        if any(c.asset != self.asset for c in self.daily_candles):
            raise ContextContractError(field="daily asset")
        if self.intraday is not None and self.intraday.asset != self.asset:
            raise ContextContractError(field="intraday asset")
        times = tuple(c.open_time for c in self.daily_candles)
        if times != tuple(sorted(set(times))):
            raise ContextContractError(field="daily chronological unique history")
        return self


class ServerClock(FrozenModel):
    serverTime: Milliseconds


class SpotPriceSummary(FrozenModel):
    asset: Asset
    semantics: Literal["current_run_only", "historical_finalized"]
    analysis_date: date | None = None
    daily_row_count: int
    daily_start: AwareDatetime | None
    daily_end: AwareDatetime | None
    latest_finalized: CompletedDailyCandle | None
    intraday: IntradayPrice | None
    daily_failure: SourceFailure | None
    intraday_failure: SourceFailure | None


class CryptoAnalystContext(CryptoContext):
    spot_prices: SpotPriceContext


class TickerPrice(FrozenModel):
    symbol: SpotSymbol
    price: StrictStr


# Binance positional wire contract. Trailing trade statistics are parsed but unused.
KlineRow = tuple[
    Milliseconds,
    StrictStr,
    StrictStr,
    StrictStr,
    StrictStr,
    StrictStr,
    Milliseconds,
    StrictStr,
    StrictInt,
    StrictStr,
    StrictStr,
    StrictStr,
]
