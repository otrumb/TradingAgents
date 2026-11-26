"""Validated classic pattern artifacts and bounded evidence-only report text."""

from collections.abc import Mapping
from datetime import date
from html import escape
from typing import Final

from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_pattern_models import (
    CryptoPatternEvidence,
    CryptoPatternSummary,
)
from tradingagents.dataflows.symbols import crypto_base

MAX_EVENTS: Final = 18


def validate_patterns(state: Mapping[str, str]) -> CryptoPatternEvidence | None:
    """Reject incomplete, malformed, inconsistent or wrong-run pattern state."""
    raw = state.get("crypto_pattern_evidence")
    compact = state.get("crypto_pattern_report")
    if raw in (None, "") and compact in (None, ""):
        return None
    if raw in (None, "") or compact in (None, ""):
        raise ContextContractError(field="pattern evidence/summary pair")
    evidence = CryptoPatternEvidence.model_validate_json(raw)
    summary = CryptoPatternSummary.model_validate_json(compact)
    if summary != evidence.summary():
        raise ContextContractError(field="pattern evidence/summary consistency")
    if (
        state.get("asset_type") != "crypto"
        or evidence.asset != crypto_base(state.get("company_of_interest", ""))
        or not state.get("trade_date")
        or date.fromisoformat(state["trade_date"]) != evidence.run_date
    ):
        raise ContextContractError(field="pattern run identity/date")
    return evidence


def pattern_sources(evidence: CryptoPatternEvidence | None) -> tuple[str, ...]:
    """Only observed Binance streams establish Binance source labels."""
    if evidence is None:
        return ()
    return tuple(
        sorted(
            {
                {"spot": "Binance Spot", "perpetual": "Binance USD-M"}[row.market]
                for row in evidence.streams
                if row.bar_count and row.provenance.source == "binance"
            }
        )
    )


def _text(value: str) -> str:
    """Bound provenance/identifiers and prevent Markdown structure injection."""
    return (
        escape(" ".join(value.split())[:96])
        .replace("|", "&#124;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("`", "&#96;")
        .replace("*", "&#42;")
        .replace("_", "&#95;")
    )


def render_patterns(summary: CryptoPatternSummary) -> str:
    """Render compact latest-bar evidence, never detector arrays or error payloads."""
    counts = {row.disposition: row.count for row in summary.catalog.dispositions}
    lines = [
        "Evidence only: certified historical observations; no predictive or execution meaning.",
        f"- Catalog: {summary.catalog.total_count} total; "
        f"available {counts.get('available', 0)}; "
        f"quarantined {counts.get('quarantined', 0)}; "
        f"unsupported {counts.get('unsupported', 0)}.",
    ]
    conditions = {
        "ok": "none",
        "unavailable": "source unavailable",
        "insufficient_history": "required history unavailable",
        "missing_bars": "continuous bar coverage missing",
        "execution_error": "detector execution incomplete",
    }
    for row in sorted(summary.streams, key=lambda row: (row.asset, row.market, row.timeframe)):
        provenance = row.provenance
        lines.extend(
            [
                f"\n#### {row.asset} / {row.market} / {row.timeframe}",
                f"- Status: {row.status}; bars {row.bar_count}; required {row.required_bars}.",
                f"- Window: {row.first_bar_at} to {row.last_bar_at}.",
                f"- Source: {provenance.source}; endpoint {_text(provenance.endpoint)}; "
                f"retrieved {provenance.retrieved_at}.",
                f"- Provenance: runner {_text(provenance.runner_version)}; "
                f"dependency {_text(provenance.dependency_version)}; "
                f"catalog {_text(provenance.catalog_digest)}.",
                f"- Missing conditions: {conditions[row.status]}.",
            ]
        )
        if row.failure is not None:
            lines.append(f"- Source failure: {row.failure.kind}; detail retained in sidecar.")
        if row.detector_errors:
            lines.append(
                f"- Detector errors: {min(len(row.detector_errors), 999)}"
                f"{'+' if len(row.detector_errors) > 999 else ''}; details retained in sidecar."
            )
        latest = sorted(
            (event for event in row.latest_events if event.signal_at == row.last_bar_at),
            key=lambda event: (
                event.pattern_id,
                event.status,
                event.signal_at,
                event.observable_at,
            ),
        )
        if not latest:
            lines.append("- No certified latest-bar pattern evidence.")
        for event in latest[:MAX_EVENTS]:
            lines.append(
                f"- Pattern {_text(event.pattern_id)}; status {event.status}; "
                f"direction {event.direction}; signal {event.signal_at}; "
                f"observable {event.observable_at}; anchor status {event.anchor_status}."
            )
        if len(latest) > MAX_EVENTS:
            lines.append(f"- Additional latest events omitted: {len(latest) - MAX_EVENTS}.")
    return "\n".join(lines)
