from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.crypto_pattern_models import (
    CryptoPatternEvidence,
    CryptoPatternSummary,
)


@pytest.fixture
def payload():
    return {
        "asset": "BTC",
        "run_date": date(2026, 1, 10),
        "catalog": {"total_count": 0, "dispositions": []},
        "streams": (),
    }


@pytest.mark.parametrize("model", [CryptoPatternEvidence, CryptoPatternSummary])
@pytest.mark.parametrize("analysis_date", [None, date(2026, 1, 10)])
def test_current_and_historical_run_dates_accepted(payload, model, analysis_date):
    evidence = model.model_validate(payload | {"analysis_date": analysis_date})

    assert evidence.run_date == date(2026, 1, 10)
    assert evidence.analysis_date == analysis_date


@pytest.mark.parametrize("model", [CryptoPatternEvidence, CryptoPatternSummary])
def test_historical_analysis_date_must_equal_run_date(payload, model):
    with pytest.raises(ValidationError, match="pattern analysis/run date"):
        model.model_validate(payload | {"analysis_date": date(2026, 1, 9)})


@pytest.mark.parametrize("model", [CryptoPatternEvidence, CryptoPatternSummary])
@pytest.mark.parametrize("patch", [{}, {"run_date": None}, {"run_date": "2026-01-10"}])
def test_run_date_required_non_nullable_and_strict(payload, model, patch):
    raw = {key: value for key, value in payload.items() if key != "run_date"} | patch

    with pytest.raises(ValidationError, match="run_date"):
        model.model_validate(raw)


def test_bars_after_historical_analysis_date_rejected(payload):
    stamp = datetime(2026, 1, 11, tzinfo=UTC)
    stream = {
        "asset": "BTC",
        "market": "spot",
        "timeframe": "1d",
        "status": "insufficient_history",
        "bar_count": 1,
        "required_bars": 160,
        "first_bar_at": stamp,
        "last_bar_at": stamp,
        "provenance": {
            "endpoint": "/api/v3/klines",
            "retrieved_at": stamp,
            "runner_version": "1",
            "dependency_version": "1.2.2",
            "catalog_digest": "sha256:fixture",
        },
    }

    with pytest.raises(ValidationError, match="pattern analysis date"):
        CryptoPatternEvidence.model_validate(
            payload | {"analysis_date": date(2026, 1, 10), "streams": (stream,)}
        )
