"""Public read-only current context API for analyst nodes and tools."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import assert_never
from urllib.parse import urlencode

import requests
from pydantic import ValidationError

from .crypto_chain import parse_growth, parse_stables, parse_tvl
from .crypto_exchange import (
    parse_book,
    parse_dvol,
    parse_ls,
    parse_oi,
    parse_options,
    parse_premium,
)
from .crypto_models import (
    Asset,
    ContextContractError,
    CryptoContext,
    Metric,
    SnapshotRequest,
    Source,
    SourceFailure,
    SourceSnapshot,
    Venue,
)
from .crypto_solana import collect_solana_context


@dataclass(frozen=True, slots=True)
class Endpoint:
    source: Source
    venue: Venue
    url: str
    coverage: str
    resolution: str
    max_age: timedelta
    parse: Callable[[str], tuple[Metric, ...]]


def endpoints(asset: Asset, now: datetime) -> tuple[Endpoint, ...]:
    match asset:
        case "ETH":
            chain = "Ethereum"
        case "BTC":
            chain = "Bitcoin"
        case "SOL":
            chain = "Solana"
        case unreachable:
            assert_never(unreachable)
    symbol = f"{asset}USDT"
    binance = "https://fapi.binance.com"
    deribit = "https://www.deribit.com/api/v2/public"
    specs = [
        Endpoint(
            "binance",
            "Binance USD-M Futures",
            f"{binance}/fapi/v1/premiumIndex?symbol={symbol}",
            symbol,
            "instant / last funding interval",
            timedelta(minutes=2),
            parse_premium,
        ),
        Endpoint(
            "binance",
            "Binance USD-M Futures",
            f"{binance}/fapi/v1/openInterest?symbol={symbol}",
            symbol,
            "instant",
            timedelta(minutes=2),
            parse_oi,
        ),
        Endpoint(
            "binance",
            "Binance USD-M Futures",
            f"{binance}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1",
            symbol,
            "5m",
            timedelta(minutes=15),
            parse_ls,
        ),
        Endpoint(
            "binance",
            "Binance USD-M Futures",
            f"{binance}/fapi/v1/ticker/bookTicker?symbol={symbol}",
            symbol,
            "instant; RPI orders excluded",
            timedelta(minutes=2),
            parse_book,
        ),
        Endpoint(
            "deribit",
            "Deribit",
            f"{deribit}/get_book_summary_by_currency?currency={asset}&kind=option",
            f"{asset} options",
            "instant / rolling 24h",
            timedelta(minutes=5),
            parse_options,
        ),
    ]
    params = urlencode(
        {
            "currency": asset,
            "start_timestamp": int((now - timedelta(minutes=5)).timestamp() * 1000),
            "end_timestamp": int(now.timestamp() * 1000),
            "resolution": "60",
        }
    )
    specs.append(
        Endpoint(
            "deribit",
            "Deribit",
            f"{deribit}/get_volatility_index_data?{params}",
            f"{asset} DVOL",
            "60s",
            timedelta(minutes=5),
            parse_dvol,
        )
    )
    if asset == "SOL":
        specs = specs[:4]  # SOL options and DVOL coverage are unsupported; never probe Deribit.
    specs.extend(
        (
            Endpoint(
                "defillama",
                "chain aggregate",
                "https://api.llama.fi/v2/chains",
                chain,
                "current; timestamp absent",
                timedelta(days=2),
                partial(parse_tvl, chain=chain),
            ),
            Endpoint(
                "defillama",
                "chain aggregate",
                "https://stablecoins.llama.fi/stablecoinchains",
                chain,
                "current; timestamp absent",
                timedelta(days=2),
                partial(parse_stables, chain=chain),
            ),
        )
    )
    if asset == "ETH":
        for export, key in (
            ("daa", "daa"),
            ("fees", "fees_paid_usd"),
            ("stables_mcap", "stables_mcap"),
        ):
            specs.append(
                Endpoint(
                    "growthepie",
                    "chain aggregate",
                    f"https://api.growthepie.com/v1/export/{export}.json",
                    "Ethereum L1",
                    "1d",
                    timedelta(days=3),
                    partial(parse_growth, metric_key=key),
                )
            )
    return tuple(specs)


def fetch_snapshot(session: requests.Session, endpoint: Endpoint, timeout: float) -> SourceSnapshot:
    metrics: tuple[Metric, ...] = ()
    failure = None
    try:
        with session.get(endpoint.url, timeout=timeout, allow_redirects=False) as response:
            response.raise_for_status()
            if response.is_redirect:
                raise ContextContractError(field="unexpected public endpoint redirect")
            metrics = endpoint.parse(response.text)
    except requests.HTTPError as exc:
        failure = SourceFailure(kind="http", detail=f"HTTP {exc.response.status_code}")
    except requests.RequestException as exc:
        failure = SourceFailure(kind="transport", detail=type(exc).__name__)
    except (ValidationError, ContextContractError) as exc:
        failure = SourceFailure(kind="parse", detail=str(exc))
    fetched_at = datetime.now(UTC)
    if endpoint.source == "binance" and any(m.coverage != endpoint.coverage for m in metrics):
        metrics = ()
        failure = SourceFailure(kind="parse", detail="Unexpected Binance symbol")
    if endpoint.source == "deribit" and endpoint.coverage.endswith("options"):
        asset = endpoint.coverage.split()[0]
        if any(not m.coverage.startswith(f"{asset}-") for m in metrics):
            metrics = ()
            failure = SourceFailure(kind="parse", detail="Unexpected Deribit currency")
    unknowns = tuple(
        f"{m.name} [{m.coverage}]: {m.freshness(fetched_at, endpoint.max_age)}"
        for m in metrics
        if m.freshness(fetched_at, endpoint.max_age) != "fresh"
    )
    observed = tuple(m.observed_at for m in metrics if m.observed_at is not None)
    return SourceSnapshot(
        source=endpoint.source,
        venue=endpoint.venue,
        endpoint=endpoint.url,
        fetched_at=fetched_at,
        observed_at=min(observed) if observed else None,
        resolution=endpoint.resolution,
        coverage=endpoint.coverage,
        metrics=metrics,
        unknowns=unknowns,
        failure=failure,
    )


def collect_crypto_context(request: SnapshotRequest) -> CryptoContext:
    """Collect latest ETH/BTC/SOL context. Call `.render()` for deterministic analyst text.

    No historical input, credentials, retries, cache, or execution methods. Failures
    belong to individual endpoints, so one unavailable provider retains other data.
    """
    request = SnapshotRequest.model_validate(request)
    with requests.Session() as session:
        session.trust_env = False
        snapshots = tuple(
            fetch_snapshot(session, endpoint, request.timeout_seconds)
            for endpoint in endpoints(request.asset, datetime.now(UTC))
        )
        if request.asset == "SOL":
            snapshots += collect_solana_context(session, request.timeout_seconds)
    if request.asset == "BTC":
        snapshots += (
            SourceSnapshot(
                source="growthepie",
                venue="chain aggregate",
                endpoint="https://api.growthepie.com/v1/fundamentals.json",
                fetched_at=datetime.now(UTC),
                observed_at=None,
                resolution="1d",
                coverage="Bitcoin L1 unsupported",
                unknowns=(
                    "Ethereum ecosystem chain metrics cannot stand in for Bitcoin activity.",
                ),
                failure=SourceFailure(
                    kind="unavailable", detail="Bitcoin L1 outside growthepie coverage"
                ),
            ),
        )
    return CryptoContext(asset=request.asset, snapshots=snapshots)
