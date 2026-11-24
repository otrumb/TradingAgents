from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.crypto_spot_models import SpotPriceContext
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.yahoo.common import price_text
from tradingagents.dataflows.vendors.yahoo.snapshot import DEFAULT_SNAPSHOT_INDICATORS


def parse_crypto_market_data(payload: str, symbol: str) -> tuple[SpotPriceContext, pd.DataFrame]:
    context = SpotPriceContext.model_validate_json(payload)
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
            f"server_time: {latest.server_time.isoformat()}",
            f"cutoff: {latest.cutoff.isoformat()}",
            f"rows: {len(context.daily_candles)}",
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
        f"{provenance(context)}\n\n{selected.to_csv(index=False)}"
    )


def crypto_indicators(
    payload: str, symbol: str, indicator: str, as_of_date: str, look_back_days: int
) -> str:
    context, frame = parse_crypto_market_data(payload, symbol)
    frame = frame[frame["Date"] <= pd.to_datetime(as_of_date)]
    if frame.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no finalized daily rows on or before {as_of_date}"
        )
    stock_frame = wrap(frame.copy())
    stock_frame[indicator]
    values = stock_frame.tail(max(1, look_back_days))[["Date", indicator]]
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
    for indicator in indicators:
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
