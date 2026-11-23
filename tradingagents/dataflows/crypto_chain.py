"""Parse chain context without treating network aggregates as token demand."""

from datetime import UTC, date, datetime

from pydantic import TypeAdapter

from .crypto_exchange import Number, WireModel
from .crypto_models import ContextContractError, Metric


class ChainTVL(WireModel):
    name: str
    tvl: Number | None


class Circulating(WireModel):
    peggedUSD: Number | None = None


class StablecoinChain(WireModel):
    name: str
    totalCirculatingUSD: Circulating | None = None
    totalCirculating: Circulating | None = None


def parse_tvl(raw: str, chain: str) -> tuple[Metric, ...]:
    rows = TypeAdapter(tuple[ChainTVL, ...]).validate_json(raw)
    selected = tuple(row for row in rows if row.name == chain)
    if len(selected) != 1:
        raise ContextContractError(field=f"TVL chain coverage: {chain}")
    return (
        Metric(
            name="chain_tvl",
            value=selected[0].tvl,
            unit="USD",
            observed_at=None,
            resolution="current, provider timestamp absent",
            coverage=chain,
        ),
    )


def parse_stables(raw: str, chain: str) -> tuple[Metric, ...]:
    rows = TypeAdapter(tuple[StablecoinChain, ...]).validate_json(raw)
    selected = tuple(row for row in rows if row.name == chain)
    if len(selected) != 1:
        raise ContextContractError(field=f"stablecoin chain coverage: {chain}")
    total = selected[0].totalCirculatingUSD
    value = total.peggedUSD if total is not None else None
    return (
        Metric(
            name="usd_pegged_stablecoin_supply",
            value=value,
            unit="USD",
            observed_at=None,
            resolution="current, provider timestamp absent",
            coverage=chain,
        ),
    )


class GrowthRow(WireModel):
    metric_key: str
    origin_key: str
    date: date
    value: Number | None


def parse_growth(raw: str, metric_key: str) -> tuple[Metric, ...]:
    """Use latest Ethereum day, preserving null instead of backfilling older values."""
    rows = TypeAdapter(tuple[GrowthRow, ...]).validate_json(raw)
    selected = tuple(
        row for row in rows if row.origin_key == "ethereum" and row.metric_key == metric_key
    )
    if not selected:
        raise ContextContractError(field=f"growthepie Ethereum metric: {metric_key}")
    latest = max(selected, key=lambda row: row.date)
    if sum(row.date == latest.date for row in selected) != 1:
        raise ContextContractError(field="duplicate growthepie observation")
    units = {"daa": "active addresses", "fees_paid_usd": "USD", "stables_mcap": "USD"}
    return (
        Metric(
            name=metric_key,
            value=latest.value,
            unit=units[metric_key],
            observed_at=datetime.combine(latest.date, datetime.min.time(), UTC),
            resolution="1d; timestamp marks UTC day start",
            coverage="Ethereum L1",
        ),
    )
