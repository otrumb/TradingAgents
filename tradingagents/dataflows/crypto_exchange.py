"""Frozen wire contracts and parsers for public exchange observations."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .crypto_models import ContextContractError, Metric

Number = Annotated[Decimal, Field(allow_inf_nan=False)]
Millis = Annotated[int, Field(ge=0, le=253402300799000)]


class WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)


class Premium(WireModel):
    symbol: str
    time: Millis
    markPrice: Number | None
    indexPrice: Number | None
    lastFundingRate: Number | None


class OpenInterest(WireModel):
    symbol: str
    time: Millis
    openInterest: Number | None


class LongShort(WireModel):
    symbol: str
    timestamp: Millis
    longShortRatio: Number | None
    longAccount: Number | None
    shortAccount: Number | None


class Book(WireModel):
    symbol: str
    time: Millis
    bidPrice: Number | None
    bidQty: Number | None
    askPrice: Number | None
    askQty: Number | None


def exchange_metrics(row: Premium | OpenInterest | LongShort | Book) -> tuple[Metric, ...]:
    """Preserve vendor units; funding is last rate, L/S is account ratio."""
    from typing import assert_never

    match row:
        case Premium():
            values = (
                ("mark_price", row.markPrice, "USDT"),
                ("index_price", row.indexPrice, "USDT"),
                ("last_funding_rate", row.lastFundingRate, "fraction per funding interval"),
            )
            timestamp, resolution = row.time, "instant / last funding interval"
        case OpenInterest():
            values = (("open_interest", row.openInterest, "base asset"),)
            timestamp, resolution = row.time, "instant"
        case LongShort():
            values = (
                ("long_short_account_ratio", row.longShortRatio, "ratio"),
                ("long_account_share", row.longAccount, "fraction"),
                ("short_account_share", row.shortAccount, "fraction"),
            )
            timestamp, resolution = row.timestamp, "5m"
        case Book():
            values = (
                ("bid_price", row.bidPrice, "USDT"),
                ("bid_quantity", row.bidQty, "base asset"),
                ("ask_price", row.askPrice, "USDT"),
                ("ask_quantity", row.askQty, "base asset"),
            )
            timestamp, resolution = row.time, "instant"
        case unreachable:
            assert_never(unreachable)
    return tuple(
        Metric(
            name=name,
            value=value,
            unit=unit,
            observed_at=datetime.fromtimestamp(timestamp / 1000, UTC),
            resolution=resolution,
            coverage=row.symbol,
        )
        for name, value, unit in values
    )


def parse_premium(raw: str) -> tuple[Metric, ...]:
    return exchange_metrics(Premium.model_validate_json(raw))


def parse_oi(raw: str) -> tuple[Metric, ...]:
    return exchange_metrics(OpenInterest.model_validate_json(raw))


def parse_ls(raw: str) -> tuple[Metric, ...]:
    rows = TypeAdapter(tuple[LongShort, ...]).validate_json(raw)
    if not rows:
        raise ContextContractError(field="empty long/short response")
    return exchange_metrics(max(rows, key=lambda row: row.timestamp))


def parse_book(raw: str) -> tuple[Metric, ...]:
    return exchange_metrics(Book.model_validate_json(raw))


class OptionSummary(WireModel):
    instrument_name: str
    creation_timestamp: Millis
    open_interest: Number | None
    volume: Number | None
    mark_iv: Number | None


class OptionsResponse(WireModel):
    result: tuple[OptionSummary, ...]


def parse_options(raw: str) -> tuple[Metric, ...]:
    rows = OptionsResponse.model_validate_json(raw).result
    if not rows:
        raise ContextContractError(field="empty options summary")
    return tuple(
        Metric(
            name=name,
            value=value,
            unit=unit,
            observed_at=datetime.fromtimestamp(row.creation_timestamp / 1000, UTC),
            resolution="instant / rolling 24h volume",
            coverage=row.instrument_name,
        )
        for row in sorted(rows, key=lambda row: row.instrument_name)
        for name, value, unit in (
            ("option_open_interest", row.open_interest, "base asset"),
            ("option_volume_24h", row.volume, "base asset"),
            ("option_mark_iv", row.mark_iv, "percent annualized"),
        )
    )


class VolatilityResult(WireModel):
    data: tuple[tuple[Millis, Number, Number, Number, Number], ...]


class VolatilityResponse(WireModel):
    result: VolatilityResult


def parse_dvol(raw: str) -> tuple[Metric, ...]:
    rows = VolatilityResponse.model_validate_json(raw).result.data
    if not rows:
        raise ContextContractError(field="empty DVOL response")
    timestamp, _, _, _, close = max(rows, key=lambda row: row[0])
    return (
        Metric(
            name="dvol_close",
            value=close,
            unit="percent annualized",
            observed_at=datetime.fromtimestamp(timestamp / 1000, UTC),
            resolution="60s candle",
            coverage="currency volatility index",
        ),
    )
