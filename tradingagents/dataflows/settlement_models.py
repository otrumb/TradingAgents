"""Immutable facts used by settlement adapters and the decision journal."""

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal


class SettlementContractError(ValueError):
    """Invalid settlement fact at the adapter or journal boundary."""

    def __init__(self, field: str):
        self.field = field
        super().__init__(f"Invalid settlement: {field}")


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementLeg:
    provider: Literal["yahoo", "binance_spot", "cash"]
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float

    def __post_init__(self) -> None:
        if self.provider not in {"yahoo", "binance_spot", "cash"} or not self.symbol.strip():
            raise SettlementContractError("provider/symbol")
        if date.fromisoformat(self.entry_date) > date.fromisoformat(self.exit_date):
            raise SettlementContractError("leg dates")
        if any(not math.isfinite(p) or p <= 0 for p in (self.entry_price, self.exit_price)):
            raise SettlementContractError("leg prices")
        if self.provider == "cash" and (
            self.symbol != "USDT" or self.entry_price != 1.0 or self.exit_price != 1.0
        ):
            raise SettlementContractError("cash leg")


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementOutcome:
    asset: SettlementLeg
    benchmark: SettlementLeg
    benchmark_label: str
    window_kind: Literal["trading_sessions", "utc_days"]
    window_count: int
    raw_return: float
    benchmark_return: float
    alpha_return: float

    def __post_init__(self) -> None:
        if self.window_kind not in {"trading_sessions", "utc_days"}:
            raise SettlementContractError("window kind")
        if type(self.window_count) is not int or self.window_count <= 0:
            raise SettlementContractError("window count")
        if not self.benchmark_label.strip():
            raise SettlementContractError("benchmark label")
        if not all(
            math.isfinite(v) for v in (self.raw_return, self.benchmark_return, self.alpha_return)
        ):
            raise SettlementContractError("returns")
        if self.alpha_return != self.raw_return - self.benchmark_return:
            raise SettlementContractError("alpha return")
        for leg, value in ((self.asset, self.raw_return), (self.benchmark, self.benchmark_return)):
            if not math.isclose(value, leg.exit_price / leg.entry_price - 1, abs_tol=1e-15):
                raise SettlementContractError("price return")
        if (
            self.benchmark.entry_date > self.asset.entry_date
            or self.benchmark.exit_date > self.asset.exit_date
        ):
            raise SettlementContractError("benchmark dates")
        if self.window_kind == "utc_days" and (
            self.asset.entry_date != self.benchmark.entry_date
            or self.asset.exit_date != self.benchmark.exit_date
            or (
                date.fromisoformat(self.asset.exit_date) - date.fromisoformat(self.asset.entry_date)
            ).days
            != self.window_count
        ):
            raise SettlementContractError("UTC window")

    @property
    def resolution_date(self) -> str:
        return self.asset.exit_date

    def legacy_tuple(self) -> tuple[float, float, int, str]:
        return self.raw_return, self.alpha_return, self.window_count, self.resolution_date
