"""Finite empirical causal certification; no detector algorithm is reimplemented."""

import hashlib
from collections import Counter
from importlib.resources import files
from typing import Final, Literal

import numpy as np
from pydantic import Field

from .crypto_pattern_catalog import (
    PACKAGE_VERSION,
    CallableDescriptor,
    CatalogError,
    CatalogModel,
    Mode,
    PatternCatalog,
    Status,
    load_pattern_catalog,
)
from .crypto_pattern_inputs import (
    FIELDS,
    POSITIVE,
    PatternInputs,
    execute,
    owned_fixture,
    structural_fixtures,
    structural_reason,
)

PREFIXES: Final = (1, 2, 5, 9, 16, 32, 64, 96, 128, 159)
ARTIFACT: Final = "catalogs/classic-crypto-v1-certification.json"


class ModeCertification(CatalogModel):
    mode: Mode
    reasons: tuple[str, ...]
    positive_fixture: str | None


class EntryCertification(CatalogModel):
    pattern_id: str
    canonical_id: str
    status: Status
    reasons: tuple[str, ...]
    modes: tuple[ModeCertification, ...]
    required_bars: int = Field(gt=0)


class CertificationReport(CatalogModel):
    catalog_version: Literal["classic-crypto-v1"] = "classic-crypto-v1"
    package_version: Literal["1.2.2"] = "1.2.2"
    protocol: Literal["bounded-causal-v1"] = "bounded-causal-v1"
    entries: tuple[EntryCertification, ...]

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def counts(self) -> Counter:
        return Counter(row.status for row in self.entries)


class CertificationDigest(CatalogModel):
    algorithm: Literal["sha256"]
    canonicalization: Literal["CertificationReport.model_dump_json UTF-8"]
    digest: str
    package_version: Literal["1.2.2"]


def certify_mode(descriptor: CallableDescriptor, mode: Mode) -> ModeCertification:
    """Exercise structure, meaningful prefixes, changed future and owned positive witness."""
    reasons: set[str] = set()
    for name, inputs in structural_fixtures():
        try:
            output = execute(descriptor, inputs, mode)
            reason = structural_reason(output, len(inputs.close))
            if reason:
                reasons.add(f"{reason}:{name}")
                continue
            for length in PREFIXES:
                if length >= len(inputs.close):
                    continue
                prefix = execute(descriptor, inputs.prefix(length), mode)
                reason = structural_reason(prefix, length)
                if reason:
                    reasons.add(f"{reason}:prefix:{name}")
                    continue
                if not np.array_equal(output[:length], prefix):
                    reasons.add("prefix_invariance")
                # Change every future price and volume while retaining valid OHLC.
                mutated = PatternInputs(
                    *(
                        np.concatenate(
                            (
                                getattr(inputs, field)[:length],
                                getattr(inputs, field)[length:] * (7 if field != "volume" else 0),
                            )
                        )
                        for field in FIELDS
                    )
                )
                future = execute(descriptor, mutated, mode)
                reason = structural_reason(future, len(inputs.close))
                if reason:
                    reasons.add(f"{reason}:suffix:{name}")
                elif not np.array_equal(output[:length], future[:length]):
                    reasons.add("future_suffix_invariance")
        except Exception as exc:
            # Dependency execution boundary: record failure per mode, continue catalog.
            reasons.add(f"execution_error:{name}:{type(exc).__name__}")
    positive = POSITIVE.get(descriptor.entry.id)
    witness = None
    if positive is None:
        reasons.add("no_positive_fixture")
    else:
        index, sign = positive
        try:
            result = execute(descriptor, owned_fixture().prefix(index + 1), mode)
            if structural_reason(result, index + 1) or result[index] != sign:
                reasons.add("positive_fixture_failed")
            else:
                witness = f"owned_shapes_v1:{index}:{sign}"
        except Exception as exc:
            reasons.add(f"positive_fixture_error:{type(exc).__name__}")
    return ModeCertification(mode=mode, reasons=tuple(sorted(reasons)), positive_fixture=witness)


def certify_catalog(catalog: PatternCatalog | None = None) -> CertificationReport:
    """All canonical detectors/modes run once; aliases inherit identical certification."""
    catalog = catalog or load_pattern_catalog(certified=False)
    by_id = {row.id: row for row in catalog.entries}
    results: dict[str, EntryCertification] = {}
    for entry in catalog.entries:
        if entry.canonical_alias is not None:
            continue
        import ta_patterns as tap
        import ta_patterns.chart_patterns as cp

        module = {"candles": tap, "chart": cp}[entry.family]
        descriptor = CallableDescriptor(entry, getattr(module, entry.name))
        modes = tuple(certify_mode(descriptor, mode) for mode in entry.modes)
        reasons = tuple(sorted({reason for mode in modes for reason in mode.reasons}))
        structural = any(reason.startswith(("output_", "execution_error:")) for reason in reasons)
        status: Status = "unsupported" if structural else "quarantined" if reasons else "available"
        results[entry.id] = EntryCertification(
            pattern_id=entry.id,
            canonical_id=entry.id,
            status=status,
            reasons=reasons or ("bounded_causal_certified",),
            modes=modes,
            required_bars=160,
        )
    for entry in catalog.entries:
        if entry.canonical_alias is None:
            continue
        target = by_id[entry.canonical_alias]
        while target.canonical_alias is not None:
            target = by_id[target.canonical_alias]
        results[entry.id] = results[target.id].model_copy(update={"pattern_id": entry.id})
    return CertificationReport(entries=tuple(results[key] for key in sorted(results)))


def load_certification() -> CertificationReport:
    """Parse packaged report and enforce complete coverage before using overrides."""
    report = CertificationReport.model_validate_json(
        files(__package__).joinpath(ARTIFACT).read_text(encoding="utf-8")
    )
    catalog = load_pattern_catalog(certified=False)
    if tuple(row.pattern_id for row in report.entries) != tuple(row.id for row in catalog.entries):
        raise CatalogError("certification coverage")
    if report.package_version != PACKAGE_VERSION:
        raise CatalogError("certification package version")
    digest = CertificationDigest.model_validate_json(
        files(__package__)
        .joinpath("catalogs/classic-crypto-v1-certification-digest.json")
        .read_text(encoding="utf-8")
    )
    if report.digest != "sha256:" + digest.digest:
        raise CatalogError("certification digest")
    return report
