import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import ta_patterns.chart_patterns as cp

from tradingagents.dataflows.crypto_pattern_catalog import CallableDescriptor, load_pattern_catalog
from tradingagents.dataflows.crypto_pattern_certification import (
    certify_catalog,
    certify_mode,
    load_certification,
)
from tradingagents.dataflows.crypto_pattern_inputs import POSITIVE, execute, owned_fixture


@pytest.fixture(scope="module")
def report():
    return certify_catalog()


def test_full_catalog_artifact_is_reproducible(report):
    path = Path(__file__).resolve().parents[1] / (
        "tradingagents/dataflows/catalogs/classic-crypto-v1-certification.json"
    )
    if os.environ.get("REGENERATE_PATTERN_CERTIFICATION") == "1":
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    packaged = load_certification()
    assert packaged == report
    assert len(report.entries) == 300
    assert report.counts() == Counter(available=6, quarantined=210, unsupported=84)
    assert all(row.reasons and row.modes for row in report.entries)
    assert all(
        mode.positive_fixture
        for row in report.entries
        if row.status == "available"
        for mode in row.modes
    )
    print("certification", report.counts(), report.digest)


def test_missing_owned_positive_quarantines_even_invariant_zero_detector():
    entry = next(
        row for row in load_pattern_catalog(certified=False).entries if row.id == "candles.hammer"
    )

    result = certify_mode(
        CallableDescriptor(entry, lambda *args: np.zeros(len(args[0]), np.int8)), "default"
    )

    assert result.reasons == ("no_positive_fixture",)


def test_known_package_prefix_mutation_is_quarantined(report):
    row = next(row for row in report.entries if row.pattern_id == "chart.measured_move_up")
    assert row.status == "quarantined"
    assert "prefix_invariance" in row.reasons


def test_prefix_gate_catches_retroactive_owned_positive():
    entry = next(
        row for row in load_pattern_catalog(certified=False).entries if row.id == "chart.inside_day"
    )

    def retroactive(opening, high, low, close):
        output = cp.inside_day(opening, high, low, close)
        if len(close) > 32:
            output[0] = 1
        return output

    result = certify_mode(CallableDescriptor(entry, retroactive), "default")
    assert "prefix_invariance" in result.reasons
    assert result.positive_fixture is not None


@pytest.mark.parametrize("pattern_id", tuple(POSITIVE))
def test_owned_witness_has_exact_signed_detection(pattern_id):
    entry = next(
        row for row in load_pattern_catalog(certified=False).entries if row.id == pattern_id
    )
    import ta_patterns as tap

    module = {"candles": tap, "chart": cp}[entry.family]
    index, sign = POSITIVE[pattern_id]
    output = execute(
        CallableDescriptor(entry, getattr(module, entry.name)),
        owned_fixture().prefix(index + 1),
        "default",
    )
    assert output[index] == sign


def test_alias_certification_runs_canonical_once(monkeypatch, report):
    catalog = load_pattern_catalog(certified=False)
    canonical = next(row for row in catalog.entries if row.id == "chart.three_day_compression")
    alias = next(row for row in catalog.entries if row.id == "chart.three_dc")
    import tradingagents.dataflows.crypto_pattern_certification as certification

    calls = []
    original = certification.certify_mode

    def tracked(descriptor, mode):
        calls.append(descriptor.entry.id)
        return original(descriptor, mode)

    monkeypatch.setattr(certification, "certify_mode", tracked)
    result = certify_catalog(replace(catalog, entries=(canonical, alias)))
    assert calls == [canonical.id]
    assert result.entries[0].modes == result.entries[1].modes
    assert len({d.entry.id for d in load_pattern_catalog().callable_descriptors()}) == 5
