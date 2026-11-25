"""Read-only journal inventory and deterministic closed-interval cohort audit."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from tradingagents.agents.rating import RATING_REVIEW, action_adjusted_return
from tradingagents.backtest_summary import OutcomeGroup, _alpha, _score, summarize_entries
from tradingagents.memory import journal_records as codec

SCHEMA: Final = "tradingagents.journal-audit"


class AuditInputError(ValueError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"Invalid journal audit: {field}")


@dataclass(frozen=True, slots=True)
class AuditQuery:
    assets: tuple[str, ...] = ("ETHUSDT", "BTCUSDT")
    window: str = "utc_days:7"
    rating: str = "Underweight"
    min_sample: int = 5

    def __post_init__(self) -> None:
        kind, _, count = self.window.partition(":")
        if kind not in {"utc_days", "trading_sessions"} or not count.isdecimal() or int(count) < 1:
            raise AuditInputError("window")
        if type(self.min_sample) is not int or self.min_sample < 1:
            raise AuditInputError("min_sample")
        if not self.assets or any(not asset.strip() for asset in self.assets):
            raise AuditInputError("assets")
        if action_adjusted_return(self.rating, 1.0) is None:
            raise AuditInputError("directional rating")


@dataclass(frozen=True, slots=True)
class Inventory:
    physical: int
    opaque: int
    recognized: int
    pending: int
    resolved: int
    scoreable: int
    unscored: int
    complete_provenance: int


@dataclass(frozen=True, slots=True, order=True)
class CohortKey:
    asset: str
    asset_provider: str
    benchmark: str
    window: str


@dataclass(frozen=True, slots=True)
class Sample:
    physical_index: int
    analysis_date: str
    entry_date: str
    exit_date: str
    rating: str
    alpha: float
    alpha_hex: str


@dataclass(frozen=True, slots=True)
class CohortAudit:
    key: CohortKey
    rating: str
    candidates: int
    selected: tuple[Sample, ...]
    excluded_overlap: int
    count: int
    status: str
    hit_rate: float | None
    mean_alpha: float | None
    mean_alpha_hex: str | None
    negative_alpha_count: int
    negative_alpha_fraction: float | None
    favorable: bool | None

    def aggregate(self) -> codec.JSONObject:
        return {
            "key": asdict(self.key),
            "rating": self.rating,
            "candidates": self.candidates,
            "excluded_overlap": self.excluded_overlap,
            "count": self.count,
            "status": self.status,
            "hit_rate": self.hit_rate,
            "mean_alpha": self.mean_alpha,
            "mean_alpha_hex": self.mean_alpha_hex,
            "negative_alpha_count": self.negative_alpha_count,
            "negative_alpha_fraction": self.negative_alpha_fraction,
            "favorable": self.favorable,
        }


@dataclass(frozen=True, slots=True)
class JournalAudit:
    inventory: Inventory
    query: AuditQuery
    cohorts: tuple[CohortAudit, ...]

    def render_json(self) -> str:
        primary = []
        for asset in sorted(set(self.query.assets)):
            matches = [
                c.aggregate()
                for c in self.cohorts
                if c.key.asset == asset
                and c.key.window == self.query.window
                and c.rating == self.query.rating
            ]
            primary.extend(
                matches
                or [
                    {
                        "asset": asset,
                        "window": self.query.window,
                        "rating": self.query.rating,
                        "count": 0,
                        "candidates": 0,
                        "excluded_overlap": 0,
                        "status": "insufficient",
                        "favorable": None,
                        "mean_alpha": None,
                        "mean_alpha_hex": None,
                        "hit_rate": None,
                        "negative_alpha_count": 0,
                        "negative_alpha_fraction": None,
                    }
                ]
            )
        return json.dumps(
            {
                "schema": SCHEMA,
                "version": 1,
                "inventory": asdict(self.inventory),
                "query": asdict(self.query),
                "cohorts": [c.aggregate() for c in self.cohorts],
                "primary": primary,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def render(self) -> str:
        lines = [
            "Journal inventory: " + ", ".join(f"{k}={v}" for k, v in asdict(self.inventory).items())
        ]
        lines.append(
            f"Advisory min_sample={self.query.min_sample}; closed intervals; earliest finish selection."
        )
        for asset in sorted(set(self.query.assets)):
            matches = [
                c
                for c in self.cohorts
                if c.key.asset == asset
                and c.key.window == self.query.window
                and c.rating == self.query.rating
            ]
            if not matches:
                lines.append(
                    f"{asset} | {self.query.window} | {self.query.rating}: insufficient (n=0)"
                )
            for cohort in matches:
                lines.append(
                    f"{asset} | {cohort.key.asset_provider} | {cohort.key.benchmark} | {cohort.key.window} | {cohort.rating}: {cohort.status} (n={cohort.count}), negative_alpha_count={cohort.negative_alpha_count}, negative_alpha_fraction={cohort.negative_alpha_fraction}, mean_alpha={cohort.mean_alpha}, hit_rate={cohort.hit_rate}, favorable={cohort.favorable}"
                )
        return "\n".join(lines)


def _sample(record: codec.Record, index: int) -> tuple[CohortKey, Sample] | None:
    entry = record.entry
    if (
        record.document is None
        or entry is None
        or entry["pending"]
        or entry["rating"] == RATING_REVIEW
    ):
        return None
    alpha = _alpha(entry)
    if alpha is None:
        return None
    match entry["outcome"]:
        case {
            "asset": {
                "provider": str(provider),
                "symbol": str(asset),
                "entry_date": str(start),
                "exit_date": str(end),
            },
            "benchmark": {"provider": str(), "symbol": str()},
            "window": {"kind": str(), "count": int()},
        }:
            group = OutcomeGroup.from_entry(entry)
            return (
                CohortKey(asset, provider, group.benchmark, group.window),
                Sample(index, entry["date"], start, end, group.rating, alpha, alpha.hex()),
            )
    return None


def audit_journal(path: Path, query: AuditQuery = AuditQuery()) -> JournalAudit:
    records = codec.decode(path.read_bytes())
    entries = [record.entry for record in records if record.entry is not None]
    summary = summarize_entries(entries)
    groups: dict[CohortKey, list[Sample]] = {}
    complete = 0
    for index, record in enumerate(records):
        candidate = _sample(record, index)
        if candidate is not None:
            key, sample = candidate
            groups.setdefault(key, []).append(sample)
            complete += 1
    audits = []
    for key, candidates in sorted(groups.items()):
        selected = []
        last_exit = ""
        for sample in sorted(
            candidates, key=lambda s: (s.exit_date, s.entry_date, s.analysis_date, s.physical_index)
        ):
            if sample.entry_date > last_exit:
                selected.append(sample)
                last_exit = sample.exit_date
        for rating in sorted({s.rating for s in candidates}):
            kept = tuple(s for s in selected if s.rating == rating)
            rows = [
                (entries_record, sample.alpha)
                for sample in kept
                if (entries_record := records[sample.physical_index].entry) is not None
            ]
            score = _score(rows) if rows else None
            enough = len(kept) >= query.min_sample
            mean = score.mean_alpha if enough and score is not None else None
            negative_count = sum(s.alpha < 0 for s in kept)
            fraction = negative_count / len(kept) if enough else None
            audits.append(
                CohortAudit(
                    key,
                    rating,
                    sum(s.rating == rating for s in candidates),
                    kept,
                    sum(s.rating == rating for s in candidates) - len(kept),
                    len(kept),
                    "sufficient" if enough else "insufficient",
                    score.hit_rate if enough and score is not None else None,
                    mean,
                    mean.hex() if mean is not None else None,
                    negative_count,
                    fraction,
                    fraction > 0.5
                    if fraction is not None and action_adjusted_return(rating, 1.0) is not None
                    else None,
                )
            )
    inventory = Inventory(
        len(records),
        len(records) - len(entries),
        len(entries),
        summary.pending,
        len(entries) - summary.pending,
        summary.resolved,
        summary.unscored,
        complete,
    )
    return JournalAudit(inventory, query, tuple(audits))
