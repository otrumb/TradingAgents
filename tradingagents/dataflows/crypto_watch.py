"""GET-only fixed-host collector with exchange-clock evidence and observation times."""

from datetime import UTC, datetime, timedelta
from typing import Final

import requests
from pydantic import TypeAdapter, ValidationError

from .crypto_models import ContextContractError, SourceFailure
from .crypto_spot_models import KlineRow, ServerClock, utc_milliseconds
from .crypto_watch_models import (
    EPOCH,
    INTERVALS,
    WATCH_SAFETY_LAG,
    WatchCandle,
    WatchRequest,
    WatchSeries,
    WatchSnapshot,
)

HOST: Final = "https://fapi.binance.com"
ROWS: Final = TypeAdapter(tuple[KlineRow, ...])


def _get(session: requests.Session, path: str, timeout: float) -> str:
    with session.get(f"{HOST}{path}", timeout=timeout, allow_redirects=False) as response:
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ContextContractError(field="watch redirect")
        return response.text


def _clock(
    session: requests.Session, timeout: float, stage: str
) -> tuple[datetime | None, SourceFailure | None]:
    try:
        clock = ServerClock.model_validate_json(_get(session, "/fapi/v1/time", timeout))
        return utc_milliseconds(clock.serverTime), None
    except requests.HTTPError as exc:
        return None, SourceFailure(
            kind="http", detail=f"{stage} clock: HTTP {exc.response.status_code}"
        )
    except requests.RequestException as exc:
        return None, SourceFailure(kind="transport", detail=f"{stage} clock: {type(exc).__name__}")
    except (ValidationError, ContextContractError, OverflowError) as exc:
        return None, SourceFailure(kind="parse", detail=f"{stage} clock: {exc}")


def collect_watch(request: WatchRequest) -> WatchSnapshot:
    symbol = f"{request.asset}USDT"
    server_time = None
    cutoff = request.as_of.astimezone(UTC) if request.as_of is not None else None
    clock_failure = None
    observable_at = cutoff
    series: list[WatchSeries] = []
    with requests.Session() as session:
        session.trust_env = False
        if cutoff is None:
            server_time, clock_failure = _clock(session, request.timeout_seconds, "initial")
            if server_time is not None:
                cutoff = server_time - WATCH_SAFETY_LAG
        for interval in INTERVALS:
            if cutoff is None:
                series.append(WatchSeries(interval=interval, failure=clock_failure))
                continue
            try:
                end_time = (cutoff - EPOCH - timedelta(microseconds=1)) // timedelta(milliseconds=1)
                path = (
                    f"/fapi/v1/klines?symbol={symbol}&interval={interval}"
                    f"&limit={request.history_limit}&endTime={end_time}"
                )
                rows = ROWS.validate_json(_get(session, path, request.timeout_seconds))
                candles = tuple(
                    sorted(
                        (
                            WatchCandle(
                                interval=interval,
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
                series.append(WatchSeries(interval=interval, candles=candles))
            except requests.HTTPError as exc:
                series.append(
                    WatchSeries(
                        interval=interval,
                        failure=SourceFailure(
                            kind="http", detail=f"HTTP {exc.response.status_code}"
                        ),
                    )
                )
            except requests.RequestException as exc:
                series.append(
                    WatchSeries(
                        interval=interval,
                        failure=SourceFailure(kind="transport", detail=type(exc).__name__),
                    )
                )
            except (ValidationError, ContextContractError, OverflowError) as exc:
                series.append(
                    WatchSeries(
                        interval=interval, failure=SourceFailure(kind="parse", detail=str(exc))
                    )
                )
        if request.as_of is None and server_time is not None:
            observable_at, clock_failure = _clock(session, request.timeout_seconds, "final")
            if observable_at is not None and observable_at < server_time:
                observable_at = None
                clock_failure = SourceFailure(
                    kind="parse", detail="final clock: before initial clock"
                )
            if clock_failure is not None:
                cutoff = server_time = None
                series = [WatchSeries(interval=i, failure=clock_failure) for i in INTERVALS]
    return WatchSnapshot(
        asset=request.asset,
        symbol=symbol,
        cutoff=cutoff,
        observable_at=observable_at,
        server_time=server_time,
        as_of=request.as_of,
        series=tuple(series),
        clock_failure=clock_failure,
    )
