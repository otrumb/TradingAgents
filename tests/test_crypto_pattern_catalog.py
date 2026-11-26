import inspect
import json
from collections import Counter
from importlib.metadata import distribution
from pathlib import Path

import pytest
import ta_patterns as tap
import ta_patterns.chart_patterns as cp
from pydantic import ValidationError

from tradingagents.dataflows.crypto_pattern_catalog import (
    CatalogError,
    load_pattern_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tradingagents/dataflows/catalogs/classic-crypto-v1.json"


def test_frozen_catalog_matches_installed_public_registry():
    catalog = load_pattern_catalog()
    assert catalog.catalog_version == "classic-crypto-v1"
    assert catalog.package_version == distribution("ta-patterns").version == "1.2.2"
    assert Counter(entry.family for entry in catalog.entries) == {"candles": 106, "chart": 194}
    for family, module, names in (
        ("candles", tap, tap.list_patterns()),
        ("chart", cp, cp.chart_list_patterns()),
    ):
        entries = [entry for entry in catalog.entries if entry.family == family]
        assert [entry.name for entry in entries] == names
        for entry in entries:
            detector = getattr(module, entry.name)
            parameters = inspect.signature(detector).parameters
            assert entry.subfamily == detector.__module__.split(".")[-1]
            assert entry.name in getattr(module, entry.direction.upper())
            assert entry.required_inputs == tuple(
                field
                for key, field in (
                    ("o", "open"),
                    ("h", "high"),
                    ("l", "low"),
                    ("c", "close"),
                    ("v", "volume"),
                )
                if key in parameters
            )
            assert entry.modes == (
                ("confirmed", "forming") if "mode" in parameters else ("default",)
            )
            assert entry.provenance


def test_volume_patterns_require_volume():
    catalog = load_pattern_catalog()
    assert {entry.name for entry in catalog.entries if "volume" in entry.required_inputs} == {
        "dome_shaped_volume",
        "u_shaped_volume",
        "rising_volume_trend",
        "falling_volume_trend",
        "volume_breakout_day",
    }


def test_pending_catalog_exposes_no_callables():
    catalog = load_pattern_catalog(certified=False)
    assert {(entry.status, entry.reasons) for entry in catalog.entries} == {
        ("quarantined", ("causal_certification_pending",))
    }
    assert catalog.callable_descriptors() == ()


@pytest.mark.parametrize(
    "corruption",
    [
        "count",
        "duplicate",
        "catalog_version",
        "package_version",
        "cycle",
        "target",
        "override",
        "status",
        "reasons",
        "pending_available",
        "columns",
    ],
)
def test_invalid_catalog_is_rejected(tmp_path: Path, corruption: str):
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    match corruption:
        case "count":
            payload["entries"].pop()
        case "duplicate":
            payload["entries"][1] = payload["entries"][0]
        case "catalog_version" | "package_version":
            payload[corruption] = "wrong"
        case "cycle":
            payload["aliases"]["chart.three_day_compression"] = "chart.three_dc"
        case "target":
            payload["aliases"]["chart.three_dc"] = "chart.missing"
        case "override":
            payload["overrides"]["chart.missing"] = {"status": "unsupported", "reasons": ["x"]}
        case "status":
            payload["status"] = "pending"
        case "reasons":
            payload["reasons"] = []
        case "pending_available":
            payload["status"] = "available"
        case "columns":
            payload["columns"].reverse()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((CatalogError, ValidationError)):
        load_pattern_catalog(path)


def test_certification_exposes_only_available_entries_and_blocks_alias_to_quarantine(
    tmp_path: Path,
):
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["overrides"] = {
        "candles.hammer": {"status": "available", "reasons": ["causal_certified"]},
        "chart.three_dc": {"status": "available", "reasons": ["causal_certified"]},
        "chart.inside_day": {"status": "unsupported", "reasons": ["certification_failed"]},
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    descriptors = load_pattern_catalog(path).callable_descriptors()
    assert [descriptor.entry.id for descriptor in descriptors] == ["candles.hammer"]
    assert descriptors[0].detector is tap.hammer


def test_certified_alias_resolves_canonical_callable(tmp_path: Path):
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["overrides"] = {
        entry_id: {"status": "available", "reasons": ["causal_certified"]}
        for entry_id in ("chart.three_dc", "chart.three_day_compression")
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    descriptors = load_pattern_catalog(path).callable_descriptors()
    assert [descriptor.entry.id for descriptor in descriptors] == [
        "chart.three_day_compression",
    ]
    assert all(descriptor.detector is cp.three_day_compression for descriptor in descriptors)


def test_installed_package_version_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tap, "__version__", "1.2.3")
    with pytest.raises(CatalogError, match="installed package version"):
        load_pattern_catalog()


def test_catalog_and_loader_stay_under_python_size_limit():
    for path in (
        ROOT / "tradingagents/dataflows/crypto_pattern_catalog.py",
        ROOT / "tradingagents/dataflows/crypto_pattern_models.py",
        ROOT / "tradingagents/dataflows/crypto_pattern_inputs.py",
        ROOT / "tradingagents/dataflows/crypto_pattern_certification.py",
        ROOT / "tradingagents/dataflows/crypto_pattern_runtime.py",
        ROOT / "tests/test_crypto_pattern_models.py",
        ROOT / "tests/test_crypto_pattern_certification.py",
        ROOT / "tests/test_crypto_pattern_runtime.py",
        Path(__file__),
    ):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert (
            sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines) <= 250
        )
