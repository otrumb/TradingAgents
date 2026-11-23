"""Fixed public Solana read methods; no caller-controlled RPC method or payload."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Final, Literal

import requests
from pydantic import Field, StrictInt, ValidationError

from .crypto_exchange import WireModel
from .crypto_models import (
    ContextContractError,
    Metric,
    Source,
    SourceFailure,
    SourceSnapshot,
    Venue,
)

RPC_URL: Final = "https://api.mainnet.solana.com"
Unsigned = Annotated[StrictInt, Field(ge=0)]


class PerformanceSample(WireModel):
    slot: Unsigned
    numTransactions: Unsigned
    numNonVoteTransactions: Unsigned | None = None
    samplePeriodSecs: Annotated[StrictInt, Field(gt=0)]


class PerformanceResponse(WireModel):
    jsonrpc: Literal["2.0"]
    id: Literal[1]
    result: Annotated[tuple[PerformanceSample, ...], Field(min_length=1, max_length=1)]


class SupplyContext(WireModel):
    slot: Unsigned


class SupplyValue(WireModel):
    total: Unsigned
    circulating: Unsigned
    nonCirculating: Unsigned


class SupplyResult(WireModel):
    context: SupplyContext
    value: SupplyValue


class SupplyResponse(WireModel):
    jsonrpc: Literal["2.0"]
    id: Literal[1]
    result: SupplyResult


def parse_performance(raw: str) -> tuple[Metric, ...]:
    """Sample TPS includes votes; non-vote TPS is separate, neither measures users."""
    row = PerformanceResponse.model_validate_json(raw).result[0]
    period = Decimal(row.samplePeriodSecs)
    return tuple(
        Metric(
            name=name,
            value=value,
            unit=unit,
            observed_at=None,
            resolution=f"{row.samplePeriodSecs}s sample ending at slot {row.slot}; timestamp absent",
            coverage="Solana mainnet; transactions include votes unless marked non-vote",
        )
        for name, value, unit in (
            ("sample_slot", Decimal(row.slot), "slot"),
            (
                "transactions_per_second_including_votes",
                Decimal(row.numTransactions) / period,
                "transactions/s including votes",
            ),
            (
                "non_vote_transactions_per_second",
                Decimal(row.numNonVoteTransactions) / period
                if row.numNonVoteTransactions is not None
                else None,
                "non-vote transactions/s; not users or token demand",
            ),
        )
    )


def parse_supply(raw: str) -> tuple[Metric, ...]:
    """Supply amounts arrive in integer lamports (1 SOL = 1,000,000,000 lamports)."""
    result = SupplyResponse.model_validate_json(raw).result
    return tuple(
        Metric(
            name=name,
            value=Decimal(value) / Decimal(1_000_000_000) if unit == "SOL" else Decimal(value),
            unit=unit,
            observed_at=None,
            resolution=f"finalized at slot {result.context.slot}; timestamp absent",
            coverage="Solana mainnet",
        )
        for name, value, unit in (
            ("context_slot", result.context.slot, "slot"),
            ("total_supply", result.value.total, "SOL"),
            ("circulating_supply", result.value.circulating, "SOL"),
            ("non_circulating_supply", result.value.nonCirculating, "SOL"),
        )
    )


def collect_solana_context(session: requests.Session, timeout: float) -> tuple[SourceSnapshot, ...]:
    """Only these two fixed POST bodies can reach the fixed public mainnet host."""
    calls: tuple[
        tuple[str, tuple[int | dict[str, str | bool], ...], Callable[[str], tuple[Metric, ...]]],
        ...,
    ] = (
        ("getRecentPerformanceSamples", (1,), parse_performance),
        (
            "getSupply",
            ({"commitment": "finalized", "excludeNonCirculatingAccountsList": True},),
            parse_supply,
        ),
    )
    snapshots = []
    for method, params, parse in calls:
        metrics: tuple[Metric, ...] = ()
        failure = None
        try:
            with session.post(
                RPC_URL,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                if response.is_redirect:
                    raise ContextContractError(field="unexpected public RPC redirect")
                metrics = parse(response.text)
        except requests.HTTPError as exc:
            failure = SourceFailure(kind="http", detail=f"HTTP {exc.response.status_code}")
        except requests.RequestException as exc:
            failure = SourceFailure(kind="transport", detail=type(exc).__name__)
        except (ValidationError, ContextContractError) as exc:
            failure = SourceFailure(kind="parse", detail=str(exc))
        snapshots.append(
            SourceSnapshot(
                source="solana_rpc",
                venue="Solana mainnet RPC",
                endpoint=f"{RPC_URL}#{method}",
                fetched_at=datetime.now(UTC),
                observed_at=None,
                resolution="latest RPC result; provider wall-clock timestamp absent",
                coverage="Solana mainnet",
                metrics=metrics,
                unknowns=(
                    "Collector received time is not observation time; slot is not a wall-clock timestamp.",
                ),
                failure=failure,
            )
        )
    unsupported: tuple[tuple[Source, Venue, str, str], ...] = (
        (
            "deribit",
            "Deribit",
            "https://www.deribit.com/api/v2/public",
            "SOL options and DVOL coverage unsupported/unverified",
        ),
        (
            "growthepie",
            "chain aggregate",
            "https://api.growthepie.com/v1/fundamentals.json",
            "Solana L1 outside growthepie coverage",
        ),
    )
    return tuple(snapshots) + tuple(
        SourceSnapshot(
            source=source,
            venue=venue,
            endpoint=url,
            fetched_at=datetime.now(UTC),
            observed_at=None,
            resolution="unsupported",
            coverage=detail,
            failure=SourceFailure(kind="unavailable", detail=detail),
        )
        for source, venue, url, detail in unsupported
    )
