from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.crypto_models import SourceFailure
from tradingagents.dataflows.crypto_pattern_models import (
    CryptoPatternEvidence,
    CryptoPatternSummary,
    PatternCatalogEvidence,
    PatternCatalogStatus,
    PatternEvent,
    PatternProvenance,
    PatternStreamEvidence,
)

NOW = datetime(2026, 1, 10, tzinfo=UTC)


@pytest.fixture
def event() -> PatternEvent:
    return PatternEvent(
        asset="BTC",
        market="spot",
        timeframe="1d",
        pattern_id="double_bottom",
        status="confirmed",
        anchor_status="available",
        pivot_at=NOW - timedelta(days=3),
        extreme_at=NOW - timedelta(days=2),
        confirmation_at=NOW - timedelta(days=1),
        signal_at=NOW - timedelta(days=1),
        observable_at=NOW,
    )


@pytest.fixture
def stream(event: PatternEvent) -> PatternStreamEvidence:
    return PatternStreamEvidence(
        asset="BTC",
        market="spot",
        timeframe="1d",
        status="ok",
        provenance=PatternProvenance(
            endpoint="/api/v3/klines",
            retrieved_at=NOW,
            runner_version="1",
            dependency_version="0.1",
            catalog_digest="sha256:fixture",
        ),
        bar_count=30,
        required_bars=20,
        first_bar_at=NOW - timedelta(days=30),
        last_bar_at=NOW - timedelta(days=1),
        events=(event,),
    )


@pytest.fixture
def evidence(stream: PatternStreamEvidence) -> CryptoPatternEvidence:
    return CryptoPatternEvidence(
        asset="BTC",
        streams=(stream,),
        run_date=NOW.date(),
        catalog=PatternCatalogEvidence.model_validate(
            {
                "total_count": 3,
                "dispositions": [
                    {"disposition": "available", "count": 2},
                    {"disposition": "quarantined", "count": 1},
                ],
            }
        ),
    )


def test_full_and_compact_json_roundtrip(evidence: CryptoPatternEvidence) -> None:
    restored = CryptoPatternEvidence.model_validate_json(evidence.model_dump_json())
    compact = restored.summary()
    assert restored == evidence
    assert CryptoPatternSummary.model_validate_json(compact.model_dump_json()) == compact
    assert compact.streams[0].provenance.catalog_digest == "sha256:fixture"
    assert compact.catalog.total_count == 3
    assert compact.run_date == NOW.date()


def test_nested_evidence_is_frozen(evidence: CryptoPatternEvidence) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        evidence.streams[0].provenance.runner_version = "changed"


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize(
    "market,timeframe",
    [("spot", "1d"), ("perpetual", "1d"), ("perpetual", "4h"), ("perpetual", "1h")],
)
def test_supported_streams_roundtrip(
    event: PatternEvent, asset: str, market: str, timeframe: str
) -> None:
    raw = event.model_dump() | {"asset": asset, "market": market, "timeframe": timeframe}
    parsed = PatternEvent.model_validate(raw)
    assert PatternEvent.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(
    "patch",
    [
        {"asset": "XRP"},
        {"market": "futures"},
        {"timeframe": "5m"},
        {"market": "spot", "timeframe": "4h"},
    ],
)
def test_unsupported_stream_rejected(event: PatternEvent, patch: dict) -> None:
    with pytest.raises(ValidationError):
        PatternEvent.model_validate(event.model_dump() | patch)


def test_aware_offsets_normalize_to_utc_and_naive_rejected(event: PatternEvent) -> None:
    parsed = PatternEvent.model_validate(
        event.model_dump() | {"observable_at": "2026-01-10T07:00:00+07:00"}
    )
    assert parsed.observable_at == NOW
    assert parsed.observable_at.utcoffset() == timedelta(0)
    with pytest.raises(ValidationError, match="timezone"):
        PatternEvent.model_validate(event.model_dump() | {"signal_at": "2026-01-09T00:00:00"})


@pytest.mark.parametrize(
    "patch",
    [
        {"confirmation_at": None},
        {"status": "forming"},
        {"confirmation_at": NOW},
        {"observable_at": NOW - timedelta(days=2)},
        {"pivot_at": NOW},
        {"extreme_at": NOW - timedelta(days=4)},
        {"extreme_at": None},
        {"anchor_status": "provider_not_exposed"},
    ],
)
def test_invalid_event_timing_or_anchors_rejected(event: PatternEvent, patch: dict) -> None:
    with pytest.raises(ValidationError):
        PatternEvent.model_validate(event.model_dump() | patch)


@pytest.mark.parametrize("status", ["forming", "failed"])
@pytest.mark.parametrize(
    "anchor_status,pivot", [("not_applicable", None), ("provider_not_exposed", None)]
)
def test_nullable_anchors_and_unconfirmed_lifecycle(
    event: PatternEvent, status: str, anchor_status: str, pivot: datetime | None
) -> None:
    parsed = PatternEvent.model_validate(
        event.model_dump()
        | {
            "status": status,
            "anchor_status": anchor_status,
            "pivot_at": pivot,
            "extreme_at": None,
            "confirmation_at": None,
        }
    )
    assert parsed.pivot_at == pivot
    assert parsed.confirmation_at is None


def test_catalog_status_injection_rejects_unknown_or_unsupported_status(
    event: PatternEvent,
) -> None:
    policy = PatternCatalogStatus(pattern_id=event.pattern_id, statuses=("forming",))
    with pytest.raises(ValidationError, match="catalog status"):
        PatternEvent.model_validate(event.model_dump(), context={"catalog_statuses": (policy,)})
    with pytest.raises(ValidationError, match="catalog status"):
        PatternEvent.model_validate(event.model_dump(), context={"catalog_statuses": ()})
    accepted = PatternCatalogStatus(pattern_id=event.pattern_id, statuses=("confirmed",))
    assert (
        PatternEvent.model_validate_json(
            event.model_dump_json(), context={"catalog_statuses": (accepted,)}
        )
        == event
    )


def test_duplicate_events_rejected_despite_observation_delay(
    stream: PatternStreamEvidence, event: PatternEvent
) -> None:
    delayed = PatternEvent.model_validate(
        event.model_dump() | {"observable_at": NOW - timedelta(hours=1)}
    )
    with pytest.raises(ValidationError, match="duplicate pattern event"):
        PatternStreamEvidence.model_validate(stream.model_dump() | {"events": (event, delayed)})


def test_duplicate_streams_rejected(evidence: CryptoPatternEvidence) -> None:
    with pytest.raises(ValidationError, match="duplicate pattern stream"):
        CryptoPatternEvidence.model_validate(
            evidence.model_dump() | {"streams": evidence.streams * 2}
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"bar_count": 0},
        {"bar_count": 1},
        {"last_bar_at": None},
        {"last_bar_at": NOW + timedelta(days=1)},
        {"asset": "ETH"},
        {"failure": {"kind": "parse", "detail": "bad bars"}},
        {"status": "missing_bars"},
        {"status": "execution_error"},
    ],
)
def test_inconsistent_stream_rejected(stream: PatternStreamEvidence, patch: dict) -> None:
    with pytest.raises(ValidationError):
        PatternStreamEvidence.model_validate(stream.model_dump() | patch)


@pytest.mark.parametrize("status", ["missing_bars", "unavailable"])
def test_empty_stream_preserves_unavailability(stream: PatternStreamEvidence, status: str) -> None:
    failure = (
        SourceFailure(kind="unavailable", detail="provider down")
        if status == "unavailable"
        else None
    )
    parsed = PatternStreamEvidence.model_validate(
        stream.model_dump()
        | {
            "status": status,
            "bar_count": 0,
            "first_bar_at": None,
            "last_bar_at": None,
            "events": (),
            "failure": failure,
        }
    )
    assert parsed.events == ()
    assert parsed.failure == failure


def test_compact_keeps_latest_per_pattern_status(
    evidence: CryptoPatternEvidence, event: PatternEvent
) -> None:
    older = PatternEvent.model_validate(
        event.model_dump()
        | {
            "signal_at": NOW - timedelta(days=2),
            "confirmation_at": NOW - timedelta(days=2),
        }
    )
    stream = evidence.streams[0]
    expanded = CryptoPatternEvidence.model_validate(
        evidence.model_dump()
        | {
            "streams": (stream.model_dump() | {"events": (event, older)},),
        }
    )
    compact = expanded.summary()
    assert compact.streams[0].event_count == 2
    assert compact.streams[0].latest_events == (event,)
    assert compact.catalog == evidence.catalog


@pytest.mark.parametrize("field", ["action", "rating", "order", "size", "forecast", "candles"])
def test_evidence_only_boundary_rejects_extras(evidence: CryptoPatternEvidence, field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        CryptoPatternEvidence.model_validate(evidence.model_dump() | {field: "forbidden"})


@pytest.mark.parametrize(
    "dispositions",
    [
        [{"disposition": "available", "count": 2}],
        [{"disposition": "available", "count": 1}, {"disposition": "available", "count": 2}],
    ],
)
def test_catalog_counts_and_duplicate_dispositions_rejected(dispositions: list) -> None:
    with pytest.raises(ValidationError, match="counts/dispositions"):
        PatternCatalogEvidence.model_validate({"total_count": 3, "dispositions": dispositions})
