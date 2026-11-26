"""Classic evidence from already finalized public streams; no fetching or trading actions."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import assert_never

import numpy as np

from .crypto_models import Asset, ContextContractError, SourceFailure
from .crypto_pattern_catalog import CallableDescriptor, load_pattern_catalog
from .crypto_pattern_certification import load_certification
from .crypto_pattern_inputs import PatternInputs, execute, structural_reason
from .crypto_pattern_models import (
    CryptoPatternEvidence,
    PatternCatalogDisposition,
    PatternCatalogEvidence,
    PatternEvent,
    PatternProvenance,
    PatternStatus,
    PatternStreamEvidence,
    PatternStreamKey,
)
from .crypto_spot_models import CompletedDailyCandle, SpotPriceContext
from .crypto_watch_models import DURATIONS, WatchCandle, WatchSeries, WatchSnapshot


@dataclass(frozen=True, slots=True)
class ClosedPatternStream:
    key: PatternStreamKey
    candles: tuple[CompletedDailyCandle, ...] | tuple[WatchCandle, ...]
    observed_at: datetime
    endpoint: str
    failure: SourceFailure | None = None


def _events(
    descriptor: CallableDescriptor, stream: ClosedPatternStream
) -> tuple[PatternEvent, ...]:
    candles = stream.candles
    inputs = PatternInputs(
        *(
            np.array([float(getattr(c, field)) for c in candles])
            for field in ("open", "high", "low", "close", "volume")
        )
    )
    latest: dict[tuple[datetime, str], PatternEvent] = {}
    for mode in descriptor.entry.modes:
        output = execute(descriptor, inputs, mode)
        reason = structural_reason(output, len(candles))
        if reason:
            raise ContextContractError(field=reason)
        if not output[-1]:
            continue
        index = len(candles) - 1
        raw = int(output[index])
        match descriptor.entry.direction:
            case "bullish":
                if raw != 1:
                    raise ContextContractError(field="bullish detector sign")
                direction = "bullish"
            case "bearish":
                if raw != -1:
                    raise ContextContractError(field="bearish detector sign")
                direction = "bearish"
            case "bidirectional":
                direction = "bullish" if raw > 0 else "bearish"
            case "non_directional":
                direction = "non_directional"
            case unreachable:
                assert_never(unreachable)
        status: PatternStatus = "forming" if mode == "forming" else "confirmed"
        if descriptor.entry.name.startswith("busted_"):
            status = "failed"
        signal = candles[index].close_time
        event = PatternEvent(
            **stream.key.model_dump(),
            pattern_id=descriptor.entry.id,
            status=status,
            direction=direction,
            raw_signal=raw,
            anchor_status="provider_not_exposed",
            signal_at=signal,
            observable_at=stream.observed_at,
            confirmation_at=signal if status == "confirmed" else None,
        )
        key = (signal, descriptor.entry.id)
        previous = latest.get(key)
        precedence = {"forming": 0, "confirmed": 1, "failed": 2}
        if previous is None or precedence[event.status] > precedence[previous.status]:
            latest[key] = event
    if not latest:
        return ()
    newest = max(event.signal_at for event in latest.values())
    return tuple(event for event in latest.values() if event.signal_at == newest)


def evaluate_stream(stream: ClosedPatternStream) -> PatternStreamEvidence:
    """Isolate dependency failures per detector, retaining successful sibling evidence."""
    report = load_certification()
    descriptors = load_pattern_catalog().callable_descriptors()
    required = max(
        (row.required_bars for row in report.entries if row.status == "available"), default=160
    )
    candles = stream.candles
    if candles and stream.observed_at < candles[-1].close_time:
        raise ContextContractError(field="pattern observation before close")
    provenance = PatternProvenance(
        endpoint=stream.endpoint,
        retrieved_at=stream.observed_at,
        runner_version="bounded-causal-v1",
        dependency_version=report.package_version,
        catalog_digest=report.digest,
    )
    status = "ok"
    events: list[PatternEvent] = []
    errors: list[str] = []
    if stream.failure is not None or not candles:
        status = "unavailable"
    elif any(
        right.open_time - left.open_time != DURATIONS[stream.key.timeframe]
        for left, right in zip(candles[:-1], candles[1:], strict=True)
    ):
        status = "missing_bars"
    elif len(candles) < required:
        status = "insufficient_history"
    else:
        for descriptor in descriptors:
            try:
                events.extend(_events(descriptor, stream))
            except Exception as exc:
                # Dependency boundary: keep failure local and visible, never abort sibling streams.
                errors.append(f"{descriptor.entry.id}:{type(exc).__name__}:{exc}")
        if errors:
            status = "execution_error"
    return PatternStreamEvidence(
        **stream.key.model_dump(),
        status=status,
        provenance=provenance,
        bar_count=len(candles),
        required_bars=required,
        first_bar_at=candles[0].close_time if candles else None,
        last_bar_at=candles[-1].close_time if candles else None,
        events=tuple(events),
        failure=stream.failure,
        detector_errors=tuple(errors),
    )


def spot_stream(context: SpotPriceContext, observed_at: datetime) -> ClosedPatternStream:
    return ClosedPatternStream(
        PatternStreamKey(asset=context.asset, market="spot", timeframe="1d"),
        context.daily_candles,
        observed_at,
        "/api/v3/klines",
        context.daily_failure,
    )


def watch_stream(asset: Asset, series: WatchSeries, observed_at: datetime) -> ClosedPatternStream:
    return ClosedPatternStream(
        PatternStreamKey(asset=asset, market="perpetual", timeframe=series.interval),
        series.candles,
        observed_at,
        "/fapi/v1/klines",
        series.failure,
    )


def classic_pattern_evidence(
    spot: SpotPriceContext,
    watch: WatchSnapshot | None,
    observed_at: datetime,
    *,
    run_date: date,
) -> CryptoPatternEvidence:
    """Use supplied retrieval time for Spot and snapshot observation time for perpetual."""
    streams = [spot_stream(spot, observed_at)]
    if watch is not None:
        if watch.asset != spot.asset:
            raise ContextContractError(field="pattern watch asset")
        streams.extend(
            watch_stream(watch.asset, series, watch.observable_at or observed_at)
            for series in watch.series
        )
    counts = load_certification().counts()
    return CryptoPatternEvidence(
        asset=spot.asset,
        run_date=run_date,
        analysis_date=spot.analysis_date,
        catalog=PatternCatalogEvidence(
            total_count=sum(counts.values()),
            dispositions=tuple(
                PatternCatalogDisposition(disposition=status, count=counts[status])
                for status in ("available", "quarantined", "unsupported")
            ),
        ),
        streams=tuple(evaluate_stream(stream) for stream in streams),
    )
