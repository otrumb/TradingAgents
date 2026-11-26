"""Immutable pattern evidence contracts; no detector or candle-history dependency."""

from datetime import date
from typing import Annotated, Final, Literal, Self, assert_never

from pydantic import Field, StrictInt, ValidationInfo, model_validator

from .crypto_derivatives_models import Timestamp
from .crypto_models import Asset, ContextContractError, FrozenModel, Source, SourceFailure

PATTERN_SCHEMA_VERSION: Final = "1"
PATTERN_CATALOG_VERSION: Final = "classic-crypto-v1"
PatternStatus = Literal["forming", "confirmed", "failed"]
StreamStatus = Literal[
    "ok", "unavailable", "insufficient_history", "missing_bars", "execution_error"
]
Count = Annotated[StrictInt, Field(ge=0)]
Identifier = Annotated[str, Field(min_length=1)]


class PatternStreamKey(FrozenModel):
    asset: Asset
    market: Literal["spot", "perpetual"]
    timeframe: Literal["1d", "4h", "1h"]

    @model_validator(mode="after")
    def supported_stream(self) -> Self:
        if (self.market, self.timeframe) not in {
            ("spot", "1d"),
            ("perpetual", "1d"),
            ("perpetual", "4h"),
            ("perpetual", "1h"),
        }:
            raise ContextContractError(field="pattern market/timeframe")
        return self


class PatternCatalogStatus(FrozenModel):
    """Loader injection seam: pass tuple of these as context['catalog_statuses']."""

    pattern_id: Identifier
    statuses: tuple[PatternStatus, ...] = Field(min_length=1)


class PatternEvent(PatternStreamKey):
    pattern_id: Identifier
    status: PatternStatus
    anchor_status: Literal["not_applicable", "available", "provider_not_exposed"]
    direction: Literal["bullish", "bearish", "non_directional"] = "non_directional"
    raw_signal: Literal[-1, 1] = 1
    pivot_at: Timestamp | None = None
    extreme_at: Timestamp | None = None
    confirmation_at: Timestamp | None = None
    signal_at: Timestamp
    observable_at: Timestamp

    @property
    def dedupe_key(self) -> tuple:
        """Identity excludes observation delay; distinct lifecycle states survive."""
        return (
            self.asset,
            self.market,
            self.timeframe,
            self.pattern_id,
            self.status,
            self.pivot_at,
            self.extreme_at,
            self.signal_at,
        )

    @model_validator(mode="after")
    def consistent_event(self, info: ValidationInfo) -> Self:
        anchors = (self.pivot_at, self.extreme_at)
        expected = {"available": 2, "not_applicable": 0, "provider_not_exposed": 0}
        if sum(stamp is not None for stamp in anchors) != expected[self.anchor_status]:
            raise ContextContractError(field="pattern anchor_status")
        if (
            self.pivot_at is not None
            and self.extreme_at is not None
            and self.pivot_at > self.extreme_at
        ):
            raise ContextContractError(field="pattern anchor order")
        if any(stamp is not None and stamp > self.signal_at for stamp in anchors):
            raise ContextContractError(field="pattern anchor after signal")
        match self.status:
            case "forming":
                if self.confirmation_at is not None:
                    raise ContextContractError(field="forming confirmation")
            case "confirmed":
                if self.confirmation_at != self.signal_at:
                    raise ContextContractError(field="missing confirmation")
            case "failed":
                pass
            case unreachable:
                assert_never(unreachable)
        if self.confirmation_at is not None and (
            self.confirmation_at > self.signal_at
            or any(stamp is not None and stamp > self.confirmation_at for stamp in anchors)
        ):
            raise ContextContractError(field="pattern confirmation order")
        if self.signal_at > self.observable_at:
            raise ContextContractError(field="pattern observable before signal")
        if info.context is not None and "catalog_statuses" in info.context:
            policies = tuple(
                PatternCatalogStatus.model_validate(row) for row in info.context["catalog_statuses"]
            )
            matches = tuple(row for row in policies if row.pattern_id == self.pattern_id)
            if len(matches) != 1 or self.status not in matches[0].statuses:
                raise ContextContractError(field="pattern catalog status")
        return self


class PatternProvenance(FrozenModel):
    source: Source = "binance"
    endpoint: Identifier
    retrieved_at: Timestamp
    runner_version: Identifier
    dependency_version: Identifier
    catalog_digest: Identifier


class PatternCatalogDisposition(FrozenModel):
    disposition: Literal["available", "unsupported", "quarantined"]
    count: Count


class PatternCatalogEvidence(FrozenModel):
    catalog_version: Literal["classic-crypto-v1"] = PATTERN_CATALOG_VERSION
    total_count: Count
    dispositions: tuple[PatternCatalogDisposition, ...]

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        names = tuple(row.disposition for row in self.dispositions)
        if (
            len(set(names)) != len(names)
            or sum(row.count for row in self.dispositions) != self.total_count
        ):
            raise ContextContractError(field="pattern catalog counts/dispositions")
        return self


class PatternStreamEvidence(PatternStreamKey):
    status: StreamStatus
    provenance: PatternProvenance
    bar_count: Count
    required_bars: Annotated[StrictInt, Field(gt=0)]
    first_bar_at: Timestamp | None = None
    last_bar_at: Timestamp | None = None
    events: tuple[PatternEvent, ...] = ()
    failure: SourceFailure | None = None
    detector_errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent_stream(self) -> Self:
        if self.bar_count == 0:
            if self.first_bar_at is not None or self.last_bar_at is not None:
                raise ContextContractError(field="empty pattern bar bounds")
        elif (
            self.first_bar_at is None
            or self.last_bar_at is None
            or not (self.first_bar_at <= self.last_bar_at <= self.provenance.retrieved_at)
        ):
            raise ContextContractError(field="pattern bar bounds")
        match self.status:
            case "ok":
                if self.bar_count < self.required_bars or self.failure is not None:
                    raise ContextContractError(field="pattern ok bars/failure")
            case "missing_bars" | "insufficient_history":
                if self.events or self.failure is not None:
                    raise ContextContractError(field="pattern missing bars")
                if self.status == "insufficient_history" and self.bar_count >= self.required_bars:
                    raise ContextContractError(field="pattern insufficient history")
            case "unavailable":
                if self.events:
                    raise ContextContractError(field="pattern failed evidence")
            case "execution_error":
                if not self.detector_errors:
                    raise ContextContractError(field="pattern detector errors")
            case unreachable:
                assert_never(unreachable)
        if len({event.dedupe_key for event in self.events}) != len(self.events):
            raise ContextContractError(field="duplicate pattern event")
        for event in self.events:
            if (event.asset, event.market, event.timeframe) != (
                self.asset,
                self.market,
                self.timeframe,
            ):
                raise ContextContractError(field="pattern event stream")
            if (
                self.last_bar_at is None
                or event.signal_at > self.last_bar_at
                or event.observable_at > self.provenance.retrieved_at
            ):
                raise ContextContractError(field="pattern event observation bounds")
        return self


class PatternStreamSummary(PatternStreamKey):
    status: StreamStatus
    provenance: PatternProvenance
    bar_count: Count
    required_bars: Annotated[StrictInt, Field(gt=0)]
    first_bar_at: Timestamp | None
    last_bar_at: Timestamp | None
    event_count: Count
    latest_events: tuple[PatternEvent, ...]
    failure: SourceFailure | None
    detector_errors: tuple[str, ...] = ()


class CryptoPatternSummary(FrozenModel):
    schema_version: Literal["1"] = PATTERN_SCHEMA_VERSION
    asset: Asset
    analysis_date: date | None = None
    catalog: PatternCatalogEvidence
    streams: tuple[PatternStreamSummary, ...]


class CryptoPatternEvidence(FrozenModel):
    schema_version: Literal["1"] = PATTERN_SCHEMA_VERSION
    asset: Asset
    analysis_date: date | None = None
    catalog: PatternCatalogEvidence
    streams: tuple[PatternStreamEvidence, ...]

    @model_validator(mode="after")
    def consistent_streams(self) -> Self:
        keys = tuple((row.market, row.timeframe) for row in self.streams)
        if len(set(keys)) != len(keys):
            raise ContextContractError(field="duplicate pattern stream")
        if any(row.asset != self.asset for row in self.streams):
            raise ContextContractError(field="pattern evidence asset")
        if self.analysis_date is not None and any(
            row.last_bar_at is not None and row.last_bar_at.date() > self.analysis_date
            for row in self.streams
        ):
            raise ContextContractError(field="pattern analysis date")
        return self

    def summary(self) -> CryptoPatternSummary:
        """Keep latest event per pattern/status, plus counts and source provenance."""
        streams = []
        for row in self.streams:
            latest: dict[tuple[str, PatternStatus], PatternEvent] = {}
            for event in sorted(
                row.events, key=lambda event: (event.signal_at, event.observable_at)
            ):
                latest[event.pattern_id, event.status] = event
            streams.append(
                PatternStreamSummary(
                    **row.model_dump(exclude={"events"}),
                    event_count=len(row.events),
                    latest_events=tuple(latest[key] for key in sorted(latest)),
                )
            )
        return CryptoPatternSummary(
            schema_version=self.schema_version,
            asset=self.asset,
            analysis_date=self.analysis_date,
            catalog=self.catalog,
            streams=tuple(streams),
        )
