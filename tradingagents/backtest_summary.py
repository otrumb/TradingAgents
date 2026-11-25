"""Exact decision statistics grouped by recorded outcome provenance."""

from dataclasses import dataclass, field
from math import fsum, isfinite

from tradingagents.agents.rating import RATING_REVIEW, action_adjusted_return
from tradingagents.memory.journal_records import Entry


def _alpha(entry: Entry) -> float | None:
    """Prefer finite exact alpha; older entries expose only rounded percent text."""
    exact = entry.get("alpha_return")
    if exact is not None and isfinite(exact):
        return exact
    try:
        value = float((entry.get("alpha") or "").strip().removesuffix("%")) / 100
    except ValueError:
        return None
    return value if isfinite(value) else None


@dataclass(frozen=True, slots=True, order=True)
class OutcomeGroup:
    """Comparable identity of outcomes with compatible measurement provenance."""

    asset: str
    benchmark: str
    window: str
    rating: str

    @classmethod
    def from_entry(cls, entry: Entry) -> "OutcomeGroup":
        asset = benchmark = "unknown"
        window = f"unknown:{entry.get('holding') or 'unknown'}"
        provenance = entry.get("provenance")
        facts = entry.get("outcome") or {}
        if not facts.get("asset") and isinstance(provenance, dict):
            facts = provenance
        match facts.get("asset"):
            case {"symbol": str(symbol)} if symbol:
                asset = symbol
        match facts.get("benchmark"):
            case {"provider": str(provider), "symbol": str(symbol)} if provider and symbol:
                benchmark = f"{provider}:{symbol}"
            case {"provider": str(provider)} if provider:
                label = facts.get("benchmark_label")
                if isinstance(label, str) and label:
                    benchmark = f"{provider}:{label}"
        match facts.get("window"):
            case {"kind": str(kind), "count": int(count)} if kind and count > 0:
                window = f"{kind}:{count}"
        return cls(asset, benchmark, window, entry["rating"])


def _hit(entry: Entry, alpha: float) -> bool | None:
    """Derive hits from the stored call and exact factual alpha."""
    adjusted = action_adjusted_return(entry["rating"], alpha)
    if adjusted is None:
        return None
    return adjusted > 0


@dataclass
class RatingScore:
    count: int
    hit_rate: float | None
    mean_alpha: float


def _score(rows: list[tuple[Entry, float]]) -> RatingScore:
    hits = [hit for entry, alpha in rows if (hit := _hit(entry, alpha)) is not None]
    return RatingScore(
        count=len(rows),
        hit_rate=sum(hits) / len(hits) if hits else None,
        mean_alpha=fsum(alpha for _, alpha in rows) / len(rows),
    )


@dataclass
class BacktestSummary:
    resolved: int
    pending: int
    by_rating: dict[str, RatingScore]
    unscored: int = 0
    holding: str = ""
    by_group: dict[OutcomeGroup, RatingScore] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"Resolved cells: {self.resolved} · pending: {self.pending}"
            + (f" · unscored: {self.unscored}" if self.unscored else "")
        ]
        for group, score in sorted(self.by_group.items()):
            called = (
                f"called the direction {score.hit_rate:.0%}"
                if score.hit_rate is not None
                else "no direction claimed"
            )
            lines.append(
                f"- {group.asset} | {group.benchmark} | {group.window} | {group.rating}: "
                f"n={score.count}, {called}, mean alpha {score.mean_alpha:+.2%}"
            )
        lines.append("")
        if self.pending:
            lines.append("Pending cells are not scored above; re-run to settle them.")
        lines.append(
            f"Alpha is measured over {self.holding} from the first close after each analysis date. "
            "One model sampling per cell, and text feeds are not archived, so "
            "these figures are indicative rather than repeatable."
        )
        return "\n".join(lines)


def summarize_entries(entries: list[Entry]) -> BacktestSummary:
    """Keep rating aggregates compatible while separating outcome measurements."""
    resolved = [
        (entry, alpha)
        for entry in entries
        if not entry["pending"]
        and entry["rating"] != RATING_REVIEW
        and (alpha := _alpha(entry)) is not None
    ]
    ratings: dict[str, list[tuple[Entry, float]]] = {}
    groups: dict[OutcomeGroup, list[tuple[Entry, float]]] = {}
    for entry, alpha in resolved:
        ratings.setdefault(entry["rating"], []).append((entry, alpha))
        groups.setdefault(OutcomeGroup.from_entry(entry), []).append((entry, alpha))
    pending = sum(entry["pending"] for entry in entries)
    unscored = len(entries) - pending - len(resolved)
    windows = set()
    for group in groups:
        kind, _, count = group.window.partition(":")
        if kind == "unknown" and count.endswith("d"):
            windows.add(f"{count[:-1]} trading days or UTC days (legacy convention unknown)")
        else:
            windows.add(group.window)
    return BacktestSummary(
        resolved=len(resolved),
        pending=pending,
        by_rating={rating: _score(rows) for rating, rows in ratings.items()},
        unscored=unscored,
        holding=", ".join(sorted(windows)) or "the configured window",
        by_group={group: _score(groups[group]) for group in sorted(groups)},
    )
