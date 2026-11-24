"""GET-only Binance public Spot collection for current and dated daily evidence."""

from datetime import UTC, datetime, time, timedelta
from typing import Final

import requests
from pydantic import TypeAdapter, ValidationError

from .crypto_models import ContextContractError, SnapshotRequest, SourceFailure
from .crypto_spot_models import (
    CompletedDailyCandle,
    IntradayPrice,
    KlineRow,
    ServerClock,
    SpotPriceContext,
    SpotPriceRequest,
    SpotSymbol,
    TickerPrice,
    utc_milliseconds,
)

HOST: Final = "https://data-api.binance.vision"
ROWS: Final = TypeAdapter(tuple[KlineRow, ...])
SYMBOL: Final = TypeAdapter(SpotSymbol)


def public_get(session: requests.Session, path: str, timeout: float) -> str:
    """Use fixed public host, reject redirects, never forward credentials."""
    with session.get(f"{HOST}{path}", timeout=timeout, allow_redirects=False) as response:
        response.raise_for_status()
        if response.is_redirect:
            raise ContextContractError(field="spot endpoint redirect")
        return response.text


def collect_spot_prices(request: SpotPriceRequest | SnapshotRequest) -> SpotPriceContext:
    """Keep clock/candle failure local; ticker remains separate usable evidence."""
    symbol = SYMBOL.validate_python(f"{request.asset}USDT")
    dated = SpotPriceRequest.model_validate(request.model_dump())
    daily: tuple[CompletedDailyCandle, ...] = ()
    intraday = None
    server_time = None
    daily_failure = None
    intraday_failure = None
    with requests.Session() as session:
        session.trust_env = False
        try:
            path = f"/api/v3/klines?symbol={symbol}&interval=1d&timeZone=0&limit=365"
            if dated.analysis_date is None:
                clock = ServerClock.model_validate_json(
                    public_get(session, "/api/v3/time", request.timeout_seconds)
                )
                server_time = utc_milliseconds(clock.serverTime)
                cutoff = server_time - timedelta(milliseconds=5000)
            else:
                cutoff = datetime.combine(dated.analysis_date, time.min, UTC) + timedelta(days=1)
                end_time = (cutoff - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(
                    milliseconds=1
                ) - 1
                path += f"&endTime={end_time}"
            rows = ROWS.validate_json(
                public_get(
                    session,
                    path,
                    request.timeout_seconds,
                )
            )
            retrieved_at = datetime.now(UTC)
            daily = tuple(
                sorted(
                    (
                        CompletedDailyCandle(
                            asset=request.asset,
                            symbol=symbol,
                            retrieved_at=retrieved_at,
                            server_time=server_time,
                            analysis_date=dated.analysis_date,
                            finality_lag_ms=5000 if dated.analysis_date is None else 0,
                            cutoff=cutoff,
                            open_time=utc_milliseconds(row[0]),
                            close_time=utc_milliseconds(row[6]),
                            open=row[1],
                            high=row[2],
                            low=row[3],
                            close=row[4],
                            volume=row[5],
                        )
                        for row in rows
                        if utc_milliseconds(row[6]) < cutoff
                    ),
                    key=lambda candle: candle.open_time,
                )
            )
            if not daily or len({c.open_time for c in daily}) != len(daily):
                raise ContextContractError(field="empty or duplicate eligible daily history")
        except requests.HTTPError as exc:
            daily_failure = SourceFailure(kind="http", detail=f"HTTP {exc.response.status_code}")
        except requests.RequestException as exc:
            daily_failure = SourceFailure(kind="transport", detail=type(exc).__name__)
        except (ValidationError, ContextContractError, OverflowError) as exc:
            daily_failure = SourceFailure(kind="parse", detail=str(exc))
        if daily_failure is not None:
            daily = ()
        if dated.analysis_date is not None:
            return SpotPriceContext(
                asset=request.asset,
                semantics="historical_finalized",
                analysis_date=dated.analysis_date,
                daily_candles=daily,
                daily_failure=daily_failure,
            )
        try:
            ticker = TickerPrice.model_validate_json(
                public_get(
                    session, f"/api/v3/ticker/price?symbol={symbol}", request.timeout_seconds
                )
            )
            intraday = IntradayPrice(
                asset=request.asset,
                symbol=ticker.symbol,
                retrieved_at=datetime.now(UTC),
                server_time=server_time,
                price=ticker.price,
            )
        except requests.HTTPError as exc:
            intraday_failure = SourceFailure(kind="http", detail=f"HTTP {exc.response.status_code}")
        except requests.RequestException as exc:
            intraday_failure = SourceFailure(kind="transport", detail=type(exc).__name__)
        except (ValidationError, ContextContractError) as exc:
            intraday_failure = SourceFailure(kind="parse", detail=str(exc))
    return SpotPriceContext(
        asset=request.asset,
        daily_candles=daily,
        intraday=intraday,
        daily_failure=daily_failure,
        intraday_failure=intraday_failure,
    )
