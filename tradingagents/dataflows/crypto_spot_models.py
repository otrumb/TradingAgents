"""Binance Spot evidence: completed UTC days and separate current price."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, StrictStr, model_validator

from .crypto_models import Asset, ContextContractError, CryptoContext, FrozenModel, SourceFailure

SpotSymbol = Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PositivePrice = Annotated[Decimal, Field(gt=0)]
Milliseconds = Annotated[StrictInt, Field(ge=0)]


def utc_milliseconds(value: int) -> datetime:
    """Convert integer exchange milliseconds without float rounding."""
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)


class SpotObservation(FrozenModel):
    asset: Asset
    symbol: SpotSymbol
    source: Literal["https://data-api.binance.vision"] = "https://data-api.binance.vision"
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
    interval: Literal["1d"] = "1d"
    timezone: Literal["0"] = "0"
    finality: Literal["closed"] = "closed"
    finality_lag_ms: Literal[5000] = 5000
    server_time: AwareDatetime
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
        if self.cutoff != self.server_time - timedelta(milliseconds=self.finality_lag_ms):
            raise ContextContractError(field="daily finality cutoff")
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
    finality: Literal["intraday_not_daily_close"] = "intraday_not_daily_close"
    price: PositivePrice
    # Binance ticker/price supplies no source observation timestamp.
    observed_at: None = None


class SpotPriceContext(FrozenModel):
    asset: Asset
    semantics: Literal["current_run_only"] = "current_run_only"
    daily_candles: tuple[CompletedDailyCandle, ...] = ()
    intraday: IntradayPrice | None = None
    daily_failure: SourceFailure | None = None
    intraday_failure: SourceFailure | None = None

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
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
