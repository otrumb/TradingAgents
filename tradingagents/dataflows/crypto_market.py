from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from typing import Final

import pandas as pd
from pydantic import ValidationError
from stockstats import wrap

from tradingagents.dataflows.crypto_spot_models import SpotPriceContext
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.dataflows.vendors.yahoo.common import price_text
from tradingagents.dataflows.vendors.yahoo.snapshot import DEFAULT_SNAPSHOT_INDICATORS

CRYPTO_INDICATORS: Final = frozenset((*DEFAULT_SNAPSHOT_INDICATORS, "vwma"))
MAX_MISSING_DATES: Final = 10


@dataclass(frozen=True, slots=True)
class DailyCoverage:
    """Interior missing UTC days; explicit dates stay bounded for tool output."""

    missing_day_count: int
    missing_dates: tuple[date, ...]


def daily_coverage(context: SpotPriceContext) -> DailyCoverage:
    """Measure adjacent finalized days without assuming coverage at either boundary."""
    count = 0
    dates: list[date] = []
    for previous, current in pairwise(context.daily_candles):
        missing = (current.open_time - previous.open_time).days - 1
        count += missing
        dates.extend(
            previous.open_time.date() + timedelta(days=offset)
            for offset in range(1, min(missing, MAX_MISSING_DATES - len(dates)) + 1)
        )
    return DailyCoverage(count, tuple(dates))


@dataclass(frozen=True, slots=True)
class UnsupportedCryptoIndicatorError(ValueError):
    indicator: str

    def __str__(self) -> str:
        return (
            f"Unsupported crypto indicator: {self.indicator}. "
            f"Supported indicators: {', '.join(sorted(CRYPTO_INDICATORS))}"
        )


def parse_crypto_market_data(payload: str, symbol: str) -> tuple[SpotPriceContext, pd.DataFrame]:
    try:
        context = SpotPriceContext.model_validate_json(payload)
    except ValidationError as exc:
        raise NoMarketDataError(symbol, symbol, "invalid Binance Spot market evidence") from exc
    if crypto_base(symbol) != context.asset:
        raise NoMarketDataError(
            symbol, symbol, "Binance Spot evidence does not match requested symbol"
        )
    if not context.daily_candles:
        raise NoMarketDataError(symbol, symbol, "no finalized Binance Spot daily candles")
    frame = pd.DataFrame(
        {
            "Date": candle.open_time.replace(tzinfo=None),
            "Open": float(candle.open),
            "High": float(candle.high),
            "Low": float(candle.low),
            "Close": float(candle.close),
            "Volume": float(candle.volume),
        }
        for candle in context.daily_candles
    )
    return context, frame


def provenance(context: SpotPriceContext) -> str:
    latest = context.daily_candles[-1]
    coverage = daily_coverage(context)
    return "\n".join(
        (
            "source: https://data-api.binance.vision",
            "venue: Binance Spot",
            f"symbol: {latest.symbol}",
            "quote_currency: USDT",
            "interval: 1d",
            "timezone: UTC",
            "finality: closed",
            f"finality_lag_ms: {latest.finality_lag_ms}",
            *(
                (f"analysis_date: {latest.analysis_date.isoformat()}", "server_time: null")
                if latest.analysis_date is not None
                else (f"server_time: {latest.server_time.isoformat()}",)
            ),
            f"cutoff: {latest.cutoff.isoformat()}",
            f"rows: {len(context.daily_candles)}",
            f"coverage: {'warning' if coverage.missing_day_count else 'contiguous'}",
            f"missing_day_count: {coverage.missing_day_count}",
            f"missing_dates: {', '.join(day.isoformat() for day in coverage.missing_dates) or 'none'}",
            f"missing_dates_omitted: {coverage.missing_day_count - len(coverage.missing_dates)}",
        )
    )


def crypto_stock_data(payload: str, symbol: str, start_date: str, end_date: str) -> str:
    context, frame = parse_crypto_market_data(payload, symbol)
    selected = frame[
        (frame["Date"] >= pd.to_datetime(start_date)) & (frame["Date"] <= pd.to_datetime(end_date))
    ]
    if selected.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no finalized daily rows from {start_date} to {end_date}"
        )
    return (
        f"# Crypto daily OHLCV for {symbol} from {start_date} to {end_date}\n"
        f"{provenance(context)}\n\n{selected.tail(30).to_csv(index=False)}"
    )


def crypto_indicators(
    payload: str, symbol: str, indicator: str, as_of_date: str, look_back_days: int
) -> str:
    if indicator not in CRYPTO_INDICATORS:
        raise UnsupportedCryptoIndicatorError(indicator)
    context, frame = parse_crypto_market_data(payload, symbol)
    frame = frame[frame["Date"] <= pd.to_datetime(as_of_date)]
    if frame.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no finalized daily rows on or before {as_of_date}"
        )
    stock_frame = wrap(frame.copy())
    stock_frame[indicator]
    values = stock_frame.tail(max(1, min(look_back_days, 30)))[["Date", indicator]]
    rows = "\n".join(
        f"{row['Date']:%Y-%m-%d}: {'N/A' if pd.isna(row[indicator]) else row[indicator]}"
        for _, row in values.iterrows()
    )
    return f"## {indicator} values through {as_of_date}\n\n{provenance(context)}\n\n{rows}"


def crypto_verified_snapshot(
    payload: str,
    symbol: str,
    as_of_date: str,
    look_back_days: int,
    indicators: Iterable[str] = DEFAULT_SNAPSHOT_INDICATORS,
) -> str:
    selected = tuple(indicators)
    for indicator in selected:
        if indicator not in CRYPTO_INDICATORS:
            raise UnsupportedCryptoIndicatorError(indicator)
    context, frame = parse_crypto_market_data(payload, symbol)
    frame = frame[frame["Date"] <= pd.to_datetime(as_of_date)]
    if frame.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no finalized daily rows on or before {as_of_date}"
        )
    stock_frame = wrap(frame.copy())
    latest = frame.iloc[-1]
    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {as_of_date}",
        f"- Latest trading row used: {latest['Date']:%Y-%m-%d}",
        "",
        provenance(context),
        "",
        "### Latest verified OHLCV row",
        "",
    ]
    lines.extend(
        f"- {field}: {price_text(latest[field])}"
        for field in ("Open", "High", "Low", "Close", "Volume")
    )
    lines.extend(("", "### Verified technical indicators (latest row)", ""))
    for indicator in selected:
        try:
            stock_frame[indicator]
            value = stock_frame.iloc[-1][indicator]
            rendered = "N/A" if pd.isna(value) else price_text(value)
        except (KeyError, ValueError, TypeError) as exc:
            rendered = f"N/A ({type(exc).__name__})"
        lines.append(f"- {indicator}: {rendered}")
    recent = frame.tail(max(1, min(look_back_days, 30)))
    lines.extend(("", f"### Recent verified closes (last {len(recent)} rows)", ""))
    lines.extend(
        f"- {row['Date']:%Y-%m-%d}: {price_text(row['Close'])}" for _, row in recent.iterrows()
    )
    return "\n".join(lines)
