"""Immutable public crypto context; latest observations, never point-in-time history."""

from datetime import timedelta
from decimal import Decimal
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Asset = Literal["ETH", "BTC", "SOL"]
Source = Literal["binance", "deribit", "defillama", "growthepie", "solana_rpc"]
Venue = Literal["Binance USD-M Futures", "Deribit", "chain aggregate", "Solana mainnet RPC"]


class FrozenModel(BaseModel):
    """Tuple-only containers keep frozen models deeply immutable."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Metric(FrozenModel):
    """Null means unavailable, zero means measured zero. Units stay explicit."""

    name: str
    value: Decimal | None
    unit: str
    observed_at: AwareDatetime | None
    resolution: str
    coverage: str

    def freshness(self, fetched_at: AwareDatetime, max_age: timedelta) -> str:
        """Future timestamps and absent observations cannot count as fresh."""
        if self.observed_at is None or self.value is None:
            return "unknown"
        age = fetched_at - self.observed_at
        if age < timedelta(0):
            return "future"
        return "fresh" if age <= max_age else "stale"


class SourceFailure(FrozenModel):
    kind: Literal["http", "transport", "parse", "unavailable"]
    detail: str


class SourceSnapshot(FrozenModel):
    source: Source
    venue: Venue
    endpoint: str
    fetched_at: AwareDatetime
    observed_at: AwareDatetime | None
    resolution: str
    coverage: str
    metrics: tuple[Metric, ...] = ()
    unknowns: tuple[str, ...] = ()
    failure: SourceFailure | None = None

    @model_validator(mode="after")
    def consistent_venue(self) -> Self:
        """Prevent exchange-specific data being presented as market-wide."""
        expected: dict[Source, Venue] = {
            "binance": "Binance USD-M Futures",
            "deribit": "Deribit",
            "defillama": "chain aggregate",
            "growthepie": "chain aggregate",
            "solana_rpc": "Solana mainnet RPC",
        }
        if self.venue != expected[self.source]:
            raise ContextContractError(field="venue")
        if self.failure is not None and self.metrics:
            raise ContextContractError(field="failure with metrics")
        return self


class ContextContractError(ValueError):
    """Invalid public context invariant."""

    def __init__(self, *, field: str) -> None:
        self.field = field
        super().__init__(f"Invalid crypto context: {field}")


class CryptoContext(FrozenModel):
    asset: Asset
    semantics: Literal["current_snapshot"] = "current_snapshot"
    snapshots: tuple[SourceSnapshot, ...]
    unknowns: tuple[str, ...] = (
        "Current public snapshots only; historical point-in-time reconstruction unsupported.",
        "Chain activity describes networks, not token-holder demand or exchange-wide positioning.",
    )

    def render(self) -> str:
        """Canonical JSON for tools and analyst nodes; no wall-clock-dependent text."""
        import json

        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2)

    def summary(self) -> "CryptoContextSummary":
        """Project source evidence without instrument-sized options arrays."""
        return CryptoContextSummary(
            asset=self.asset,
            semantics=self.semantics,
            sources=tuple(SourceSummary.from_snapshot(row) for row in self.snapshots),
            unknowns=self.unknowns,
        )


class OptionFamilySummary(FrozenModel):
    name: str
    unit: str
    resolution: str
    count: int
    available_count: int
    missing_count: int
    minimum: Decimal | None
    maximum: Decimal | None
    total: Decimal | None
    observed_start: AwareDatetime | None
    observed_end: AwareDatetime | None


class SourceSummary(FrozenModel):
    source: Source
    venue: Venue
    endpoint: str
    fetched_at: AwareDatetime
    observed_at: AwareDatetime | None
    resolution: str
    coverage: str
    metrics: tuple[Metric, ...]
    option_instrument_count: int
    option_families: tuple[OptionFamilySummary, ...]
    freshness_counts: dict[str, int]
    unknowns: tuple[str, ...]
    failure: SourceFailure | None

    @classmethod
    def from_snapshot(cls, row: SourceSnapshot) -> "SourceSummary":
        """Keep provenance; aggregate option families and collector freshness flags."""
        options = tuple(m for m in row.metrics if m.name.startswith("option_"))
        families = []
        for name, unit, resolution in sorted({(m.name, m.unit, m.resolution) for m in options}):
            members = tuple(
                m for m in options if (m.name, m.unit, m.resolution) == (name, unit, resolution)
            )
            values = sorted(m.value for m in members if m.value is not None)
            observed = sorted(m.observed_at for m in members if m.observed_at is not None)
            families.append(
                OptionFamilySummary(
                    name=name,
                    unit=unit,
                    resolution=resolution,
                    count=len(members),
                    available_count=len(values),
                    missing_count=len(members) - len(values),
                    minimum=min(values) if values else None,
                    maximum=max(values) if values else None,
                    total=sum(values, Decimal(0))
                    if values and name in {"option_open_interest", "option_volume_24h"}
                    else None,
                    observed_start=min(observed) if observed else None,
                    observed_end=max(observed) if observed else None,
                )
            )
        flags = {f"{m.name} [{m.coverage}]": m for m in options}
        freshness_counts: dict[str, int] = {}
        unknowns = []
        for note in row.unknowns:
            prefix, _, status = note.rpartition(": ")
            if prefix in flags and status in {"unknown", "future", "stale", "fresh"}:
                freshness_counts[status] = freshness_counts.get(status, 0) + 1
            else:
                unknowns.append(note)
        return cls(
            source=row.source,
            venue=row.venue,
            endpoint=row.endpoint,
            fetched_at=row.fetched_at,
            observed_at=row.observed_at,
            resolution=row.resolution,
            coverage=row.coverage,
            metrics=tuple(m for m in row.metrics if not m.name.startswith("option_")),
            option_instrument_count=len({m.coverage for m in options}),
            option_families=tuple(families),
            freshness_counts=freshness_counts,
            unknowns=tuple(unknowns),
            failure=row.failure,
        )


class CryptoContextSummary(FrozenModel):
    asset: Asset
    semantics: Literal["current_snapshot"]
    sources: tuple[SourceSummary, ...]
    unknowns: tuple[str, ...]


class SnapshotRequest(FrozenModel):
    """No date/as-of input: this collector cannot satisfy historical requests."""

    asset: Asset
    timeout_seconds: float = Field(default=10, gt=0, le=60)
