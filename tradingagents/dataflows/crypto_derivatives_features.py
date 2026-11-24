"""Decimal features from validated observations, anchored to UTC retrieval time."""

from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from typing import Final

from .crypto_derivatives_models import (
    DerivativesHistory,
    FundingObservation,
    HistoricalFeature,
    InterestObservation,
    ObservedWindow,
)

FUNDING_MAX_AGE: Final = timedelta(hours=12)
OI_MAX_AGE: Final = timedelta(minutes=90)
BASELINE_TOLERANCE: Final = timedelta(minutes=30)


def observed_window(
    bounds: tuple[datetime, datetime], times: tuple[datetime, ...]
) -> ObservedWindow:
    return ObservedWindow(
        requested_start=bounds[0],
        requested_end=bounds[1],
        observed_start=min(times) if times else None,
        observed_end=max(times) if times else None,
        sample_count=len(times),
        actual_seconds=Decimal(str((max(times) - min(times)).total_seconds())) if times else None,
    )


def funding_features(history: DerivativesHistory) -> tuple[HistoricalFeature, ...]:
    now = history.retrieved_at
    results = []
    for name, days in (("funding_mean_30d", 30), ("funding_trend_7d_minus_previous_7d", 7)):
        bounds = ((now - timedelta(days=days), now),)
        if days == 7:
            bounds += ((now - timedelta(days=14), now - timedelta(days=7)),)
        groups = tuple(
            tuple(row for row in history.funding if start <= row.observed_at < end)
            for start, end in bounds
        )
        windows = tuple(
            observed_window(bound, tuple(row.observed_at for row in rows))
            for bound, rows in zip(bounds, groups, strict=True)
        )
        reason = history.funding_failure
        freshness = "unknown"
        value = None
        if not reason:
            if history.funding:
                freshness = (
                    "fresh" if now - history.funding[-1].observed_at <= FUNDING_MAX_AGE else "stale"
                )
            if freshness == "stale":
                reason = "latest funding older than 12 hours"
            elif any(
                not funding_coverage(rows, bound)
                for rows, bound in zip(groups, bounds, strict=True)
            ):
                reason = "require >=2 events/day, both edges and all gaps <=12 hours per window"
            else:
                with localcontext() as ctx:
                    ctx.prec = 34
                    means = tuple(
                        sum((row.rate for row in rows), Decimal(0)) / len(rows) for rows in groups
                    )
                    value = means[0] if days == 30 else means[0] - means[1]
        status = "unavailable" if history.funding_failure else ("insufficient" if reason else "ok")
        results.append(
            HistoricalFeature(
                symbol=history.symbol,
                name=name,
                value=value,
                unit="rate",
                requested_horizon_seconds=days * 86400,
                windows=windows,
                freshness=freshness,
                coverage="sufficient" if status == "ok" else status,
                status=status,
                reason=reason,
            )
        )
    return tuple(results)


def funding_coverage(
    rows: tuple[FundingObservation, ...], bounds: tuple[datetime, datetime]
) -> bool:
    if len(rows) < (bounds[1] - bounds[0]).days * 2:
        return False
    times = (bounds[0], *(row.observed_at for row in rows), bounds[1])
    return all(
        right - left <= FUNDING_MAX_AGE for left, right in zip(times, times[1:], strict=False)
    )


def interest_features(history: DerivativesHistory) -> tuple[HistoricalFeature, ...]:
    results = []
    latest = history.interest[-1] if history.interest else None
    for hours in (1, 4, 24):
        reason = history.interest_failure
        freshness = "unknown"
        baseline: InterestObservation | None = None
        end = latest.observed_at if latest else history.retrieved_at
        target = end - timedelta(hours=hours)
        if latest:
            freshness = "fresh" if history.retrieved_at - end <= OI_MAX_AGE else "stale"
            candidates = tuple(row for row in history.interest if row.observed_at < end)
            if candidates:
                baseline = min(
                    candidates, key=lambda row: (abs(row.observed_at - target), row.observed_at)
                )
        if not reason:
            if latest is None:
                reason = "no hourly OI observations"
            elif freshness == "stale":
                reason = "latest hourly OI older than 90 minutes (1h cadence plus 30m publication allowance)"
            elif baseline is None or abs(baseline.observed_at - target) > BASELINE_TOLERANCE:
                reason = "nearest baseline outside inclusive 30-minute tolerance"
            elif baseline.contracts <= 0:
                reason = "baseline OI must be positive"
        selected = tuple(
            row.observed_at
            for row in history.interest
            if target - BASELINE_TOLERANCE <= row.observed_at <= end
        )
        # Observed bounds measure the selected baseline/latest pair; counts retain intervening evidence.
        if baseline and abs(baseline.observed_at - target) <= BASELINE_TOLERANCE:
            selected = tuple(
                row.observed_at
                for row in history.interest
                if baseline.observed_at <= row.observed_at <= end
            )
        window = observed_window((target - BASELINE_TOLERANCE, end), selected)
        status = "unavailable" if history.interest_failure else ("insufficient" if reason else "ok")
        for unit in ("contracts", "percent"):
            value = None
            if reason is None and latest is not None and baseline is not None:
                with localcontext() as ctx:
                    ctx.prec = 34
                    delta = latest.contracts - baseline.contracts
                    value = delta if unit == "contracts" else delta / baseline.contracts * 100
            results.append(
                HistoricalFeature(
                    symbol=history.symbol,
                    name=f"oi_delta_{hours}h_{unit}",
                    baseline_target=target,
                    baseline_deviation_seconds=(
                        Decimal(str((baseline.observed_at - target).total_seconds()))
                        if baseline
                        else None
                    ),
                    value=value,
                    unit=unit,
                    requested_horizon_seconds=hours * 3600,
                    windows=(window,),
                    freshness=freshness,
                    coverage="sufficient" if status == "ok" else status,
                    status=status,
                    reason=reason,
                )
            )
    return tuple(results)


def with_features(history: DerivativesHistory) -> DerivativesHistory:
    return DerivativesHistory(
        **history.model_dump(exclude={"features"}),
        features=funding_features(history) + interest_features(history),
    )
