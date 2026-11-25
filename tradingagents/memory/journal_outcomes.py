from dataclasses import asdict
from datetime import date
from math import isfinite
from typing import NotRequired, TypeAlias, TypedDict

from tradingagents.agents.rating import action_adjusted_return
from tradingagents.dataflows.settlement_models import (
    SettlementContractError,
    SettlementLeg,
    SettlementOutcome,
)

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


def valid_outcome(outcome: JSONObject, rating: str, trade_date: str) -> bool:
    """Recognize known settlement facts without discarding unknown extensions."""
    raw, alpha = outcome.get("raw_return"), outcome.get("alpha_return")
    days, resolved = outcome.get("holding_days"), outcome.get("resolution_date")
    if (
        type(raw) not in (int, float)
        or type(alpha) not in (int, float)
        or type(days) is not int
        or days <= 0
        or "resolution_date" not in outcome
    ):
        return False
    try:
        if not isfinite(raw) or not isfinite(alpha):
            return False
        if resolved is not None and (
            not isinstance(resolved, str)
            or date.fromisoformat(resolved).isoformat() != resolved
            or resolved < trade_date
        ):
            return False
        if any(
            key in outcome for key in ("asset", "window", "benchmark_return", "benchmark_label")
        ):
            legs = []
            for key in ("asset", "benchmark"):
                facts = outcome.get(key)
                if not isinstance(facts, dict):
                    return False
                names = ("provider", "symbol", "entry_date", "exit_date")
                prices = ("entry_price", "exit_price")
                if not all(isinstance(facts.get(name), str) for name in names):
                    return False
                if not all(type(facts.get(name)) in (int, float) for name in prices):
                    return False
                if any(
                    date.fromisoformat(facts[name]).isoformat() != facts[name]
                    for name in ("entry_date", "exit_date")
                ):
                    return False
                legs.append(SettlementLeg(**{name: facts[name] for name in (*names, *prices)}))
            window = outcome.get("window")
            label, benchmark_return = (
                outcome.get("benchmark_label"),
                outcome.get("benchmark_return"),
            )
            if (
                not isinstance(window, dict)
                or not isinstance(window.get("kind"), str)
                or not isinstance(label, str)
                or type(benchmark_return) not in (int, float)
            ):
                return False
            detailed = SettlementOutcome(
                asset=legs[0],
                benchmark=legs[1],
                benchmark_label=label,
                window_kind=window.get("kind"),
                window_count=window.get("count"),
                raw_return=raw,
                benchmark_return=benchmark_return,
                alpha_return=alpha,
            )
            if (
                days != detailed.window_count
                or resolved != detailed.resolution_date
                or detailed.asset.entry_date <= trade_date
            ):
                return False
        if "scoring" in outcome:
            scoring = outcome["scoring"]
            value = action_adjusted_return(rating, alpha)
            expected: JSONObject = {
                "method": "rating-direction",
                "version": 1,
                "basis": "alpha",
                "direction": action_adjusted_return(rating, 1.0),
                "value": value,
                "hit": None if value is None else value > 0,
            }
            if not isinstance(scoring, dict):
                return False
            for key, expected_value in expected.items():
                actual = scoring.get(key)
                if key not in scoring or type(actual) is not type(expected_value):
                    return False
                if isinstance(expected_value, float):
                    if actual.hex() != expected_value.hex():
                        return False
                elif actual != expected_value:
                    return False
    except (ValueError, OverflowError):
        return False
    return True


class OutcomeUpdate(TypedDict):
    ticker: str
    trade_date: str
    raw_return: float
    alpha_return: float
    holding_days: int
    reflection: str
    resolution_date: NotRequired[str | None]
    detailed_outcome: NotRequired[SettlementOutcome | None]


def outcome_document(update: OutcomeUpdate, rating: str) -> JSONObject:
    outcome: JSONObject = {
        "raw_return": update["raw_return"],
        "alpha_return": update["alpha_return"],
        "holding_days": update["holding_days"],
        "resolution_date": update.get("resolution_date"),
    }
    detailed = update.get("detailed_outcome")
    if detailed is not None:
        if (
            update["raw_return"] != detailed.raw_return
            or update["alpha_return"] != detailed.alpha_return
            or update["holding_days"] != detailed.window_count
            or update.get("resolution_date") not in (None, detailed.resolution_date)
        ):
            raise SettlementContractError("update metadata")
        outcome.update(
            {
                "asset": asdict(detailed.asset),
                "benchmark": asdict(detailed.benchmark),
                "benchmark_label": detailed.benchmark_label,
                "benchmark_return": detailed.benchmark_return,
                "window": {"kind": detailed.window_kind, "count": detailed.window_count},
                "resolution_date": detailed.resolution_date,
            }
        )
    value = action_adjusted_return(rating, update["alpha_return"])
    outcome["scoring"] = {
        "method": "rating-direction",
        "version": 1,
        "basis": "alpha",
        "direction": action_adjusted_return(rating, 1.0),
        "value": value,
        "hit": None if value is None else value > 0,
    }
    return outcome
