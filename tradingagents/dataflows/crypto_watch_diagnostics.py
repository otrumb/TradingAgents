"""Exact Decimal thresholds for both directions, using closed evidence only."""

from decimal import Context, Decimal, localcontext
from typing import Literal

from .crypto_watch_models import Interval, WatchCondition, WatchSnapshot


def ema(closes: tuple[Decimal, ...], period: int) -> tuple[Decimal, ...]:
    """First-close seed, alpha 2/(period+1), fixed precision 28."""
    with localcontext(Context(prec=28)):
        alpha = Decimal(2) / Decimal(period + 1)
        values = [closes[0]]
        for close in closes[1:]:
            values.append(values[-1] + alpha * (close - values[-1]))
        return tuple(values)


def diagnose_watch(snapshot: WatchSnapshot) -> tuple[WatchCondition, ...]:
    """Emit conjunctions per stage/direction; pullback difference is an OR over three bars."""
    with localcontext(Context(prec=28)):
        return _conditions(snapshot)


def _conditions(snapshot: WatchSnapshot) -> tuple[WatchCondition, ...]:
    daily, four, hour = (s.candles for s in snapshot.series)
    d20 = ema(tuple(c.close for c in daily), 20)[-1]
    d50 = ema(tuple(c.close for c in daily), 50)[-1]
    f20 = ema(tuple(c.close for c in four), 20)
    prior = tuple(zip(four[-4:-1], f20[-4:-1], strict=True))
    rows: tuple[tuple[Interval, str, Decimal, Decimal, Literal[">", "<", "<=", ">="]], ...] = (
        ("1d", "long:close_above_ema20", daily[-1].close, d20, ">"),
        ("1d", "long:ema20_above_ema50", d20, d50, ">"),
        ("1d", "short:close_below_ema20", daily[-1].close, d20, "<"),
        ("1d", "short:ema20_below_ema50", d20, d50, "<"),
        ("4h", "long:close_above_ema20", four[-1].close, f20[-1], ">"),
        ("4h", "long:close_above_previous_high", four[-1].close, four[-2].high, ">"),
        (
            "4h",
            "long:min_pullback_low_minus_own_ema20",
            min(c.low - e for c, e in prior),
            Decimal(0),
            "<=",
        ),
        ("4h", "short:close_below_ema20", four[-1].close, f20[-1], "<"),
        ("4h", "short:close_below_previous_low", four[-1].close, four[-2].low, "<"),
        (
            "4h",
            "short:max_pullback_high_minus_own_ema20",
            max(c.high - e for c, e in prior),
            Decimal(0),
            ">=",
        ),
        (
            "1h",
            "long:close_above_prior_20_high",
            hour[-1].close,
            max(c.high for c in hour[-21:-1]),
            ">",
        ),
        (
            "1h",
            "short:close_below_prior_20_low",
            hour[-1].close,
            min(c.low for c in hour[-21:-1]),
            "<",
        ),
    )
    stamps = {s.interval: s.candles[-1].close_time for s in snapshot.series}
    return tuple(
        WatchCondition(
            interval=interval,
            name=name,
            observed=observed,
            threshold=threshold,
            comparison=comparison,
            as_of=stamps[interval],
            met={
                ">": observed > threshold,
                "<": observed < threshold,
                "<=": observed <= threshold,
                ">=": observed >= threshold,
            }[comparison],
        )
        for interval, name, observed, threshold, comparison in rows
    )
