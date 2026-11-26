"""Frozen registry metadata; availability requires independent causal certification.

JSON rows follow ``columns``. Root status/reasons/provenance apply to every row;
per-ID overrides let certification promote entries without changing registry data.
IDs use family.name, avoiding candle/chart name collisions. Modes describe direct
call signatures, not scanner-wide mode forwarding. No detectors run during loading.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal, assert_never

import numpy as np
import ta_patterns as tap
import ta_patterns.chart_patterns as cp
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

CATALOG_VERSION: Final = "classic-crypto-v1"
PACKAGE_VERSION: Final = "1.2.2"
Family = Literal["candles", "chart"]
Direction = Literal["bullish", "bearish", "bidirectional", "non_directional"]
Status = Literal["available", "quarantined", "unsupported"]
Mode = Literal["default", "confirmed", "forming"]
Row = tuple[str, Family, str, Direction, Literal["c", "oc", "ohlc", "ohlcv"], bool]
Detector = Callable[..., NDArray[np.int8]]


class CatalogError(ValueError):
    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"Invalid classic pattern catalog: {field}")


class CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Certification(CatalogModel):
    status: Status
    reasons: tuple[str, ...]


class CatalogWire(CatalogModel):
    catalog_version: Literal["classic-crypto-v1"]
    package_version: Literal["1.2.2"]
    provenance: str = Field(min_length=1)
    status: Status
    reasons: tuple[str, ...]
    aliases: dict[str, str]
    columns: tuple[str, ...]
    entries: tuple[Row, ...]
    overrides: dict[str, Certification]


@dataclass(frozen=True, slots=True)
class PatternEntry:
    id: str
    name: str
    display_name: str
    family: Family
    subfamily: str
    direction: Direction
    required_inputs: tuple[str, ...]
    modes: tuple[Mode, ...]
    canonical_alias: str | None
    status: Status
    reasons: tuple[str, ...]
    provenance: str


@dataclass(frozen=True, slots=True)
class CallableDescriptor:
    entry: PatternEntry
    detector: Detector


@dataclass(frozen=True, slots=True)
class PatternCatalog:
    catalog_version: str
    package_version: str
    entries: tuple[PatternEntry, ...]

    def callable_descriptors(self) -> tuple[CallableDescriptor, ...]:
        """Return only available entries whose entire alias chain is available."""
        by_id = {entry.id: entry for entry in self.entries}
        descriptors = []
        seen: set[str] = set()
        for entry in self.entries:
            current = entry
            while current.status == "available" and current.canonical_alias is not None:
                current = by_id[current.canonical_alias]
            match current.status:
                case "available":
                    if current.id in seen:
                        continue
                    seen.add(current.id)
                    match current.family:
                        case "candles":
                            detector = getattr(tap, current.name)
                        case "chart":
                            detector = getattr(cp, current.name)
                        case unreachable:
                            assert_never(unreachable)
                    if not callable(detector):
                        raise CatalogError(f"callable {current.id}")
                    descriptors.append(CallableDescriptor(current, detector))
                case "quarantined" | "unsupported":
                    continue
                case unreachable:
                    assert_never(unreachable)
        return tuple(descriptors)


def load_pattern_catalog(path: Path | None = None, *, certified: bool = True) -> PatternCatalog:
    """Load packaged catalog or certification revision, rejecting version/alias drift."""
    if version("ta-patterns") != PACKAGE_VERSION or tap.__version__ != PACKAGE_VERSION:
        raise CatalogError("installed package version")
    source = (
        path if path is not None else files(__package__).joinpath("catalogs/classic-crypto-v1.json")
    )
    wire = CatalogWire.model_validate_json(source.read_text(encoding="utf-8"))
    overrides = wire.overrides
    if path is None and certified:
        from .crypto_pattern_certification import load_certification

        overrides = {
            row.pattern_id: Certification(status=row.status, reasons=row.reasons)
            for row in load_certification().entries
        }
    if wire.columns != (
        "name",
        "family",
        "subfamily",
        "direction",
        "required_inputs",
        "supports_mode",
    ):
        raise CatalogError("columns")
    counts = Counter(row[1] for row in wire.entries)
    if counts != {"candles": 106, "chart": 194}:
        raise CatalogError("family counts")
    ids = tuple(f"{row[1]}.{row[0]}" for row in wire.entries)
    if len(set(ids)) != 300 or ids != tuple(sorted(ids)):
        raise CatalogError("unique sorted IDs")
    if (wire.aliases.keys() | wire.overrides.keys()) - set(ids):
        raise CatalogError("unknown alias/override ID")
    for alias in wire.aliases:
        visited = {alias}
        target = wire.aliases[alias]
        while True:
            if target not in ids or target in visited:
                raise CatalogError("missing or cyclic alias target")
            visited.add(target)
            if target not in wire.aliases:
                break
            target = wire.aliases[target]
    entries = []
    for row, entry_id in zip(wire.entries, ids, strict=True):
        name, family, subfamily, direction, inputs, supports_mode = row
        certification = overrides.get(
            entry_id, Certification(status=wire.status, reasons=wire.reasons)
        )
        if certification.status != "available" and not certification.reasons:
            raise CatalogError(f"missing status reasons {entry_id}")
        if certification.status == "available" and (
            not certification.reasons or "causal_certification_pending" in certification.reasons
        ):
            raise CatalogError(f"missing certification {entry_id}")
        entries.append(
            PatternEntry(
                id=entry_id,
                name=name,
                display_name=name.replace("_", " ").title(),
                family=family,
                subfamily=subfamily,
                direction=direction,
                required_inputs=tuple(
                    {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}[key]
                    for key in inputs
                ),
                modes=("confirmed", "forming") if supports_mode else ("default",),
                canonical_alias=wire.aliases.get(entry_id),
                status=certification.status,
                reasons=certification.reasons,
                provenance=wire.provenance,
            )
        )
    return PatternCatalog(wire.catalog_version, wire.package_version, tuple(entries))
