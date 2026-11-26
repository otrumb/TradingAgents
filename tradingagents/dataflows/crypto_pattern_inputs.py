"""Owned, bounded OHLCV fixtures and package invocation shared by certification/runtime."""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .crypto_pattern_catalog import CallableDescriptor, Mode


@dataclass(frozen=True, slots=True)
class PatternInputs:
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]

    def prefix(self, length: int) -> "PatternInputs":
        return PatternInputs(*(getattr(self, key)[:length].copy() for key in FIELDS))


FIELDS: Final = ("open", "high", "low", "close", "volume")
# Explicit owned witnesses: bar index and signed expected detection, never searched.
POSITIVE: Final = {
    "candles.black_candle": (1, -1),
    "candles.white_candle": (0, 1),
    "candles.doji": (2, 1),
    "candles.engulfing_bullish": (4, 1),
    "chart.inside_day": (2, 1),
    "chart.three_day_compression": (8, 1),
}


def owned_fixture() -> PatternInputs:
    """Hand-authored shape witnesses followed by independent oscillating future bars."""
    rng = np.random.default_rng(422)
    close = 100 + np.cumsum(rng.uniform(-5, 5, 160))
    opening = close + rng.uniform(-2, 2, 160)
    high = np.maximum(opening, close) + rng.uniform(0.1, 4, 160)
    low = np.minimum(opening, close) - rng.uniform(0.1, 4, 160)
    for i, bar in enumerate(
        (
            (100, 104, 99, 103),
            (104, 105, 98, 100),
            (102, 103, 101, 102),
            (103, 104, 99, 100),
            (99, 105, 98, 104),
            (100, 120, 80, 101),
            (100, 103, 98, 101),
            (100, 102, 99, 101),
            (100, 102, 99, 101),
        )
    ):
        opening[i], high[i], low[i], close[i] = bar
    return PatternInputs(opening, high, low, close, rng.uniform(1, 1000, 160))


def structural_fixtures() -> tuple[tuple[str, PatternInputs], ...]:
    """Valid positive OHLC, including zero volume and degenerate ranges."""
    base = owned_fixture()
    flat = np.full(160, 100.0)
    volatile = np.where(np.arange(160) % 2, 1000.0, 1.0)
    return (
        ("owned", base),
        ("flat", PatternInputs(flat.copy(), flat.copy(), flat.copy(), flat.copy(), flat.copy())),
        ("zero_volume", PatternInputs(base.open, base.high, base.low, base.close, np.zeros(160))),
        ("high_volatility", PatternInputs(volatile, volatile + 1, volatile * 0.5, volatile, flat)),
        *((f"short_{n}", base.prefix(n)) for n in (0, 1, 2, 5)),
    )


def execute(descriptor: CallableDescriptor, inputs: PatternInputs, mode: Mode) -> NDArray[np.int8]:
    """Use frozen signature metadata; never forward scanner-wide mode arguments."""
    args = tuple(getattr(inputs, field).copy() for field in descriptor.entry.required_inputs)
    match mode:
        case "default":
            return descriptor.detector(*args)
        case "confirmed" | "forming":
            return descriptor.detector(*args, mode=mode)
        case unreachable:
            from typing import assert_never

            assert_never(unreachable)


def structural_reason(output: NDArray[np.int8], length: int) -> str | None:
    """Validate dependency output without silently casting or padding it."""
    if not isinstance(output, np.ndarray):
        return "output_not_array"
    if output.shape != (length,):
        return "output_length"
    if output.dtype != np.int8:
        return "output_dtype"
    if not np.isin(output, (-1, 0, 1)).all():
        return "output_domain"
    return None
