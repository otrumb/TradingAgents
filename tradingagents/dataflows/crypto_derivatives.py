"""Credential-free fixed-endpoint Binance USD-M history collection."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final, assert_never

import requests
from pydantic import TypeAdapter, ValidationError

from .crypto_derivatives_features import with_features
from .crypto_derivatives_models import (
    DerivativesContractError,
    DerivativesHistory,
    FundingObservation,
    FundingWire,
    InterestObservation,
    InterestWire,
    Symbol,
)
from .crypto_models import SnapshotRequest

FUNDING_URL: Final = "https://fapi.binance.com/fapi/v1/fundingRate"
INTEREST_URL: Final = "https://fapi.binance.com/futures/data/openInterestHist"
FUNDING_ROWS: Final = TypeAdapter(tuple[FundingWire, ...])
INTEREST_ROWS: Final = TypeAdapter(tuple[InterestWire, ...])
SYMBOL: Final = TypeAdapter(Symbol)


def timestamp(milliseconds: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=milliseconds)


def parse_failure(exc: ValidationError | DerivativesContractError | InvalidOperation) -> str:
    match exc:
        case ValidationError():
            first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
            return f"parse: {exc.error_count()} errors; first {first['loc']}: {first['type']}"[:240]
        case DerivativesContractError():
            return f"parse: {exc.field}"[:240]
        case InvalidOperation():
            return "parse: invalid decimal"
        case unreachable:
            assert_never(unreachable)


def parse_funding(raw: str, symbol: Symbol) -> tuple[FundingObservation, ...]:
    rows = FUNDING_ROWS.validate_json(raw, strict=True)
    result: dict[int, FundingObservation] = {}
    previous = 0
    for row in rows:
        if row.symbol != symbol or row.fundingTime < previous:
            raise DerivativesContractError("funding symbol/order")
        observation = FundingObservation(
            symbol=symbol, observed_at=timestamp(row.fundingTime), rate=Decimal(row.fundingRate)
        )
        if row.fundingTime in result and result[row.fundingTime] != observation:
            raise DerivativesContractError("conflicting funding duplicate")
        result[row.fundingTime] = observation
        previous = row.fundingTime
    return tuple(result.values())


def parse_interest(raw: str, symbol: Symbol) -> tuple[InterestObservation, ...]:
    rows = INTEREST_ROWS.validate_json(raw, strict=True)
    result: dict[int, InterestObservation] = {}
    previous = 0
    for row in rows:
        if row.symbol != symbol or row.timestamp < previous:
            raise DerivativesContractError("OI symbol/order")
        observation = InterestObservation(
            symbol=symbol,
            observed_at=timestamp(row.timestamp),
            contracts=Decimal(row.sumOpenInterest),
            notional_usdt=Decimal(row.sumOpenInterestValue),
        )
        if row.timestamp in result and result[row.timestamp] != observation:
            raise DerivativesContractError("conflicting OI duplicate")
        result[row.timestamp] = observation
        previous = row.timestamp
    return tuple(result.values())


def collect_derivatives(request: SnapshotRequest) -> DerivativesHistory:
    symbol = SYMBOL.validate_python(f"{request.asset}USDT")
    end = datetime.now(UTC)
    start = end - timedelta(days=30)
    end_ms = int(end.timestamp() * 1000)
    cursor = int(start.timestamp() * 1000)
    funding: tuple[FundingObservation, ...] = ()
    interest: tuple[InterestObservation, ...] = ()
    funding_failure = None
    interest_failure = None
    with requests.Session() as session:
        session.trust_env = False
        try:
            collected: dict[datetime, FundingObservation] = {}
            # 30 days at Binance's shortest 1h funding cadence fit within two 1000-row pages.
            for _ in range(2):
                with session.get(
                    FUNDING_URL,
                    params={
                        "symbol": symbol,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                    timeout=request.timeout_seconds,
                    allow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    if response.is_redirect:
                        raise DerivativesContractError("funding redirect")
                    page = parse_funding(response.text, symbol)
                    count = len(FUNDING_ROWS.validate_json(response.text, strict=True))
                    if count > 1000:
                        raise DerivativesContractError("funding response limit")
                if any(
                    not start <= row.observed_at <= end
                    or int(row.observed_at.timestamp() * 1000) < cursor
                    for row in page
                ):
                    raise DerivativesContractError("funding request bounds")
                collected.update((row.observed_at, row) for row in page)
                if count < 1000:
                    break
                if not page:
                    raise DerivativesContractError("funding pagination progress")
                cursor = int(page[-1].observed_at.timestamp() * 1000) + 1
            else:
                raise DerivativesContractError("funding pagination limit")
            funding = tuple(collected.values())
        except requests.HTTPError as exc:
            funding_failure = f"HTTP {exc.response.status_code} at {FUNDING_URL}"
        except requests.RequestException as exc:
            funding_failure = f"{type(exc).__name__} at {FUNDING_URL}"
        except (ValidationError, DerivativesContractError, InvalidOperation) as exc:
            funding_failure = parse_failure(exc)
        try:
            with session.get(
                INTEREST_URL,
                params={
                    "symbol": symbol,
                    "period": "1h",
                    "startTime": int((end - timedelta(hours=26)).timestamp() * 1000),
                    "endTime": end_ms,
                    "limit": 500,
                },
                timeout=request.timeout_seconds,
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                if response.is_redirect:
                    raise DerivativesContractError("OI redirect")
                interest = parse_interest(response.text, symbol)
            if len(interest) > 500 or any(
                not end - timedelta(hours=26) <= row.observed_at <= end for row in interest
            ):
                raise DerivativesContractError("OI request bounds/limit")
        except requests.HTTPError as exc:
            interest = ()
            interest_failure = f"HTTP {exc.response.status_code} at {INTEREST_URL}"
        except requests.RequestException as exc:
            interest = ()
            interest_failure = f"{type(exc).__name__} at {INTEREST_URL}"
        except (ValidationError, DerivativesContractError, InvalidOperation) as exc:
            interest = ()
            interest_failure = parse_failure(exc)
    return with_features(
        DerivativesHistory(
            symbol=symbol,
            retrieved_at=datetime.now(UTC),
            features=(),
            funding=funding,
            interest=interest,
            funding_failure=funding_failure,
            interest_failure=interest_failure,
        )
    )
