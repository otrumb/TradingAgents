from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.crypto_derivatives_features import with_features
from tradingagents.dataflows.crypto_derivatives_models import (
    DerivativesHistory,
    FundingObservation,
    InterestObservation,
)

NOW = datetime(2026, 11, 24, 12, tzinfo=UTC)


def history(*, funding=(), interest=(), **kwargs) -> DerivativesHistory:
    return with_features(
        DerivativesHistory(
            symbol="BTCUSDT",
            retrieved_at=NOW,
            features=(),
            funding=funding,
            interest=interest,
            **kwargs,
        )
    )


def oi(hours: float, value: str) -> InterestObservation:
    return InterestObservation(
        symbol="BTCUSDT",
        observed_at=NOW - timedelta(hours=hours),
        contracts=Decimal(value),
        notional_usdt=Decimal("1000"),
    )


def test_arithmetic_funding_mean_and_observed_utc_trend():
    rows = tuple(
        FundingObservation(
            symbol="BTCUSDT",
            observed_at=NOW - timedelta(hours=hours),
            rate=Decimal("0.003") if hours <= 168 else Decimal("-0.001"),
        )
        for hours in range(720, 0, -8)
    )

    result = history(funding=rows)

    mean, trend = result.features[:2]
    assert mean.value == Decimal("-0.00006666666666666666666666666666666667")
    assert trend.value == Decimal("0.004")
    assert [w.sample_count for w in trend.windows] == [21, 21]
    assert trend.windows[0].actual_seconds == Decimal(160 * 3600)


def test_funding_zero_is_measured_and_sparse_windows_are_null():
    rows = tuple(
        FundingObservation(symbol="BTCUSDT", observed_at=NOW - timedelta(hours=h), rate=Decimal(0))
        for h in range(720, 0, -8)
    )
    full = history(funding=rows)
    sparse = history(funding=rows[-1:])
    assert full.features[0].value == 0
    assert full.features[1].value == 0
    assert sparse.features[0].value is None
    assert sparse.features[1].status == "insufficient"
    assert sparse.features[1].reason


@pytest.mark.parametrize(
    "deviation,available", [(30, True), (-30, True), (30.001, False), (-30.001, False)]
)
def test_baseline_tolerance_is_inclusive(deviation, available):
    result = history(interest=(oi(4 + deviation / 60, "100"), oi(0, "125")))
    feature = result.features[4]
    assert feature.value == (Decimal(25) if available else None)
    assert result.features[5].value == (Decimal(25) if available else None)


@pytest.mark.parametrize("age,available", [(90, True), (90.001, False)])
def test_hourly_latest_freshness_allows_30_minute_publication_delay(age, available):
    result = history(interest=(oi(age / 60 + 1, "100"), oi(age / 60, "110")))
    assert result.features[2].value == (Decimal(10) if available else None)
    assert result.features[2].freshness == ("fresh" if available else "stale")


@pytest.mark.parametrize(
    "base,latest,absolute,percent",
    [("0", "100", None, None), ("100", "0", "-100", "-100"), ("100", "100", "0", "0")],
)
def test_positive_baseline_zero_latest_and_zero_delta(base, latest, absolute, percent):
    result = history(interest=(oi(1, base), oi(0, latest)))
    assert result.features[2].value == (Decimal(absolute) if absolute is not None else None)
    assert result.features[3].value == (Decimal(percent) if percent is not None else None)


def test_hourly_deltas_use_nearest_timestamp_not_row_offset():
    result = history(
        interest=(oi(24, "50"), oi(4.1, "80"), oi(3.8, "85"), oi(1, "90"), oi(0, "100"))
    )
    assert [f.value for f in result.features[2::2]] == [Decimal(10), Decimal(20), Decimal(50)]
    assert result.features[4].windows[0].actual_seconds == Decimal(14760)
    assert result.features[4].windows[0].sample_count == 4


def test_empty_failed_and_stale_funding_have_null_values():
    empty = history()
    failed = history(funding_failure="HTTP 451")
    stale = history(
        funding=(
            FundingObservation(
                symbol="BTCUSDT", observed_at=NOW - timedelta(hours=13), rate=Decimal(0)
            ),
        )
    )
    assert all(f.value is None and f.reason for f in empty.features)
    assert failed.features[0].status == "unavailable"
    assert stale.features[0].freshness == "stale"
    assert stale.features[0].value is None


def test_sidecar_rejects_duplicate_future_wrong_symbol_and_nonfinite():
    for rows in ((oi(1, "1"), oi(1, "1")), (oi(-1, "1"),), (oi(0, "1"), oi(1, "1"))):
        with pytest.raises(ValidationError):
            history(interest=rows)
    with pytest.raises(ValidationError):
        oi(0, "NaN")
    with pytest.raises(ValidationError):
        DerivativesHistory(symbol="ETHUSDT", retrieved_at=NOW, features=(), interest=(oi(0, "1"),))
