"""Settle crypto decisions using consecutive finalized Binance Spot UTC days."""

from datetime import UTC, date, datetime, time, timedelta
from typing import Annotated, Final

import requests
from pydantic import Field, StrictInt, TypeAdapter, ValidationError

from .crypto_models import Asset, ContextContractError, FrozenModel
from .crypto_spot import public_get
from .crypto_spot_models import (
    CompletedDailyCandle,
    KlineRow,
    ServerClock,
    SpotSymbol,
    utc_milliseconds,
)
from .symbols import crypto_base

ROWS: Final = TypeAdapter(tuple[KlineRow, ...])
SYMBOL: Final = TypeAdapter(SpotSymbol)
PENDING: Final = (None, None, None, None)
TIMEOUT: Final = 10.0
DAY: Final = timedelta(days=1)
EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


class SettlementPriceError(ValueError):
    """Compact failure safe to persist in settlement diagnostics."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Binance settlement: {reason}")


class SettlementRequest(FrozenModel):
    asset: Asset
    decision_date: date
    holding_days: Annotated[StrictInt, Field(gt=0)]

    @property
    def entry(self) -> datetime:
        return datetime.combine(self.decision_date, time.min, UTC) + DAY


def _candles(
    session: requests.Session,
    request: SettlementRequest,
    server_time: datetime,
) -> tuple[CompletedDailyCandle, ...]:
    """Validate wire order and exact requested dates without filling gaps."""
    symbol = SYMBOL.validate_python(f"{request.asset}USDT")
    cutoff = server_time - timedelta(seconds=5)
    candles = []
    for offset in range(0, request.holding_days + 1, 1000):
        start = request.entry + offset * DAY
        count = min(1000, request.holding_days + 1 - offset)
        end = start + count * DAY - timedelta(milliseconds=1)
        path = (
            f"/api/v3/klines?symbol={symbol}&interval=1d&timeZone=0"
            f"&startTime={(start - EPOCH) // timedelta(milliseconds=1)}"
            f"&endTime={(end - EPOCH) // timedelta(milliseconds=1)}&limit={count}"
        )
        rows = ROWS.validate_json(public_get(session, path, TIMEOUT))
        previous = None
        for row in rows:
            opened = utc_milliseconds(row[0])
            closed = utc_milliseconds(row[6])
            if opened < start or opened > end or (previous is not None and opened <= previous):
                raise SettlementPriceError("invalid candle dates/order")
            previous = opened
            if closed >= cutoff:
                continue
            candles.append(
                CompletedDailyCandle(
                    asset=request.asset,
                    symbol=symbol,
                    retrieved_at=datetime.now(UTC),
                    server_time=server_time,
                    cutoff=cutoff,
                    open_time=opened,
                    close_time=closed,
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5],
                )
            )
    return tuple(candles)


def fetch_crypto_returns(
    ticker: str,
    trade_date: str,
    holding_days: int = 5,
) -> tuple[float | None, float | None, int | None, str | None]:
    """Return legacy outcome tuple; incomplete exact UTC windows stay pending.

    BTC uses USDT cash (zero return); ETH/SOL use BTCUSDT on identical days.
    One exchange clock snapshot governs both legs, including finality lag.
    """
    try:
        request = SettlementRequest(
            asset=crypto_base(ticker),
            decision_date=trade_date,
            holding_days=holding_days,
        )
        with requests.Session() as session:
            session.trust_env = False
            clock = ServerClock.model_validate_json(public_get(session, "/api/v3/time", TIMEOUT))
            server_time = utc_milliseconds(clock.serverTime)
            exit_time = request.entry + (holding_days + 1) * DAY - timedelta(milliseconds=1)
            if exit_time >= server_time - timedelta(seconds=5):
                return PENDING
            asset = _candles(session, request, server_time)
            expected = tuple(request.entry + i * DAY for i in range(holding_days + 1))
            if tuple(c.open_time for c in asset) != expected:
                return PENDING
            benchmark_return = 0.0
            if request.asset != "BTC":
                benchmark = _candles(
                    session, request.model_copy(update={"asset": "BTC"}), server_time
                )
                if tuple(c.open_time for c in benchmark) != expected:
                    return PENDING
                benchmark_return = float(benchmark[-1].close / benchmark[0].close - 1)
        raw = float(asset[-1].close / asset[0].close - 1)
        return raw, raw - benchmark_return, holding_days, asset[-1].open_time.date().isoformat()
    except requests.HTTPError as exc:
        raise SettlementPriceError(f"HTTP {exc.response.status_code}") from None
    except requests.RequestException as exc:
        raise SettlementPriceError(type(exc).__name__) from None
    except (ValidationError, ContextContractError, OverflowError):
        raise SettlementPriceError("invalid response or request") from None
