"""Decision journal wire codec; unknown records retain their original bytes."""

import json
import math
import re
from dataclasses import dataclass
from datetime import date as iso_date
from typing import Final, TypedDict

from tradingagents.memory.journal_outcomes import (
    JSONObject as JSONObject,
    JSONValue as JSONValue,
    OutcomeUpdate as OutcomeUpdate,
    outcome_document as outcome_document,
    valid_outcome,
)

SEPARATOR: Final = "\n\n<!-- ENTRY_END -->\n\n"
SCHEMA: Final = "tradingagents.decision-journal"
_DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
_REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)
_SEPARATOR_RE = re.compile(rb"\r?\n\r?\n<!-- ENTRY_END -->\r?\n\r?\n")


class Entry(TypedDict):
    """Compatible display fields plus exact values and explicit provenance."""

    date: str
    ticker: str
    rating: str
    decision: str
    reflection: str
    pending: bool
    raw: str | None
    alpha: str | None
    holding: str | None
    resolved: str | None
    raw_return: float | None
    alpha_return: float | None
    holding_days: int | None
    provenance: JSONValue
    outcome: JSONObject | None


@dataclass(frozen=True, slots=True)
class Record:
    """Physical span, parsed view, and lossless v2 document when recognized."""

    raw: bytes
    entry: Entry | None
    document: JSONObject | None = None


def encode(document: JSONObject) -> bytes:
    """Frame one compact JSON object, rejecting nonfinite numbers recursively."""
    return ("\x1e" + json.dumps(document, allow_nan=False, separators=(",", ":")) + "\n").encode()


def pending_document(date: str, ticker: str, rating: str, decision: str) -> JSONObject:
    """Build the v2 pending contract; prose is always JSON string data."""
    return {
        "schema": SCHEMA,
        "version": 2,
        "date": date,
        "ticker": ticker,
        "rating": rating,
        "decision": decision,
        "outcome": None,
        "reflection": None,
        "provenance": None,
    }


def _view(document: JSONObject) -> Entry | None:
    """Parse recognized fields at the storage boundary; retain extension metadata."""
    match document:
        case {
            "date": str(date),
            "ticker": str(ticker),
            "rating": str(rating),
            "decision": str(decision),
            "outcome": _,
            "reflection": _,
        }:
            pass
        case _:
            return None
    outcome = document.get("outcome")
    try:
        if iso_date.fromisoformat(date).isoformat() != date:
            return None
    except ValueError:
        return None
    reflection = document.get("reflection")
    raw = alpha = None
    days = resolved = None
    if outcome is None:
        if reflection is not None:
            return None
    else:
        if not isinstance(outcome, dict) or not isinstance(reflection, str):
            return None
        if not valid_outcome(outcome, rating, date):
            return None
        raw, alpha = outcome.get("raw_return"), outcome.get("alpha_return")
        days, resolved = outcome.get("holding_days"), outcome.get("resolution_date")
        raw, alpha = float(raw), float(alpha)
    return Entry(
        date=date,
        ticker=ticker,
        rating=rating,
        decision=decision,
        reflection=reflection or "",
        pending=outcome is None,
        raw=f"{raw:+.1%}" if raw is not None else None,
        alpha=f"{alpha:+.1%}" if alpha is not None else None,
        holding=f"{days}d" if days is not None else None,
        resolved=resolved,
        raw_return=raw,
        alpha_return=alpha,
        holding_days=days,
        provenance=document.get("provenance"),
        outcome=outcome,
    )


def _legacy(raw: bytes) -> Entry | None:
    lines = raw.decode("utf-8", errors="surrogateescape").strip().splitlines()
    if not lines or not (lines[0].startswith("[") and lines[0].endswith("]")):
        return None
    fields = [field.strip() for field in lines[0][1:-1].split("|")]
    if len(fields) not in (4, 6, 7):
        return None
    pending = fields[3] == "pending"
    if pending != (len(fields) == 4):
        return None
    raw_return = alpha_return = days = None
    resolved = None
    try:
        if iso_date.fromisoformat(fields[0]).isoformat() != fields[0]:
            return None
        if not pending:
            if not (
                fields[3].endswith("%") and fields[4].endswith("%") and fields[5].endswith("d")
            ):
                return None
            raw_return = float(fields[3][:-1]) / 100
            alpha_return = float(fields[4][:-1]) / 100
            days = int(fields[5][:-1])
            if not math.isfinite(raw_return) or not math.isfinite(alpha_return) or days <= 0:
                return None
        if len(fields) == 7:
            if not fields[6].startswith("resolved:"):
                return None
            resolved = fields[6][len("resolved:") :].strip()
            if iso_date.fromisoformat(resolved).isoformat() != resolved or resolved < fields[0]:
                return None
    except ValueError:
        return None
    body = "\n".join(lines[1:]).strip()
    decision, reflection = _DECISION_RE.search(body), _REFLECTION_RE.search(body)
    return Entry(
        date=fields[0],
        ticker=fields[1],
        rating=fields[2],
        pending=pending,
        raw=None if pending else fields[3],
        alpha=None if pending else fields[4],
        holding=None if pending else fields[5],
        resolved=resolved,
        decision=decision.group(1).strip() if decision else "",
        reflection=reflection.group(1).strip() if reflection else "",
        raw_return=raw_return,
        alpha_return=alpha_return,
        holding_days=days,
        provenance=None,
        outcome=None
        if pending
        else {
            "raw_return": raw_return,
            "alpha_return": alpha_return,
            "holding_days": days,
            "resolution_date": resolved,
        },
    )


def decode(data: bytes) -> list[Record]:
    """Scan physical spans without splitting JSON prose or normalizing opaque bytes."""
    records = []
    offset = 0
    while offset < len(data):
        if data[offset : offset + 1] == b"\x1e":
            end = data.find(b"\n", offset)
            end = len(data) if end < 0 else end + 1
            next_frame = data.find(b"\x1e", offset + 1)
            if next_frame >= 0:
                end = min(end, next_frame)
            raw = data[offset:end]
            document = entry = None
            try:
                value = json.loads(raw[1:])
                if (
                    isinstance(value, dict)
                    and value.get("schema") == SCHEMA
                    and type(value.get("version")) is int
                    and value["version"] == 2
                ):
                    # Re-encoding validates finite numbers, including extension metadata.
                    encode(value)
                    entry = _view(value)
                    document = value if entry is not None else None
            except (ValueError, UnicodeDecodeError):
                entry = None
            records.append(Record(raw, entry, document))
        else:
            boundary = _SEPARATOR_RE.search(data, offset)
            frame = data.find(b"\x1e", offset)
            end = len(data) if boundary is None else boundary.end()
            if frame >= 0:
                end = min(end, frame)
            raw = data[offset:end]
            body = raw if boundary is None else data[offset : min(end, boundary.start())]
            records.append(Record(raw, _legacy(body)))
        offset = end
    return records


def resolve(record: Record, outcome: JSONObject, reflection: str) -> bytes:
    """Upgrade a recognized pending record, retaining all v2 extension fields."""
    entry = record.entry
    assert entry is not None
    document = record.document or pending_document(
        entry["date"], entry["ticker"], entry["rating"], entry["decision"]
    )
    return encode({**document, "outcome": outcome, "reflection": reflection})


def rotate(records: list[Record], maximum: int | None) -> list[Record]:
    """Cap recognized resolved records only; keep opaque and pending spans intact."""
    if not maximum or maximum <= 0:
        return records
    resolved = [
        (record.entry["date"], i)
        for i, record in enumerate(records)
        if record.entry is not None and not record.entry["pending"]
    ]
    dropped = {i for _, i in sorted(resolved)[: max(0, len(resolved) - maximum)]}
    return [record for i, record in enumerate(records) if i not in dropped]
