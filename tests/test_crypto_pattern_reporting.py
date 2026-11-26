import json
import re
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_pattern_models import CryptoPatternEvidence
from tradingagents.portfolio import PortfolioContext
from tradingagents.report_decision import decision_summary, observed_sources
from tradingagents.reporting import write_report_tree
from tradingagents.spot_decision import calculate_spot_allocation

NOW = datetime(2026, 1, 10, tzinfo=UTC)


@pytest.fixture
def evidence() -> CryptoPatternEvidence:
    return CryptoPatternEvidence.model_validate(
        {
            "asset": "BTC",
            "run_date": NOW.date(),
            "analysis_date": "2026-01-10",
            "catalog": {
                "total_count": 300,
                "dispositions": [
                    {"disposition": "available", "count": 6},
                    {"disposition": "quarantined", "count": 210},
                    {"disposition": "unsupported", "count": 84},
                ],
            },
            "streams": [
                {
                    "asset": "BTC",
                    "market": "spot",
                    "timeframe": "1d",
                    "status": "ok",
                    "bar_count": 30,
                    "required_bars": 20,
                    "first_bar_at": NOW - timedelta(days=30),
                    "last_bar_at": NOW,
                    "provenance": {
                        "source": "binance",
                        "endpoint": "/api/v3/klines",
                        "retrieved_at": NOW,
                        "runner_version": "1",
                        "dependency_version": "1.2.2",
                        "catalog_digest": "sha256:fixture",
                    },
                    "events": [
                        {
                            "asset": "BTC",
                            "market": "spot",
                            "timeframe": "1d",
                            "pattern_id": "double_bottom",
                            "status": "confirmed",
                            "anchor_status": "provider_not_exposed",
                            "direction": "bullish",
                            "confirmation_at": NOW,
                            "signal_at": NOW,
                            "observable_at": NOW,
                        }
                    ],
                }
            ],
        }
    )


def state_for(evidence: CryptoPatternEvidence) -> dict[str, str]:
    return {
        "company_of_interest": "BTC-USD",
        "asset_type": "crypto",
        "trade_date": "2026-01-10",
        "crypto_pattern_evidence": evidence.model_dump_json(),
        "crypto_pattern_report": evidence.summary().model_dump_json(),
    }


def test_sidecar_roundtrip_and_stale_removal(evidence, tmp_path):
    state = state_for(evidence)
    write_report_tree(state, "BTC-USD", tmp_path)
    sidecar = tmp_path / "evidence/crypto_patterns.json"
    saved = sidecar.read_text(encoding="utf-8")
    assert CryptoPatternEvidence.model_validate_json(saved) == evidence
    assert saved == json.dumps(
        json.loads(state["crypto_pattern_evidence"]), sort_keys=True, indent=2
    )

    write_report_tree({}, "BTC-USD", tmp_path)
    assert not sidecar.exists()
    assert not (tmp_path / "6_patterns/patterns.md").exists()
    assert "Classic Pattern Evidence" not in (tmp_path / "complete_report.html").read_text()


@pytest.mark.parametrize("field", ["crypto_pattern_evidence", "crypto_pattern_report"])
@pytest.mark.parametrize("raw", ["{", '{"unexpected":1}', "null", "", None])
def test_malformed_or_unpaired_state_rejected(evidence, tmp_path, field, raw):
    state = state_for(evidence) | {field: raw}
    with pytest.raises((ValidationError, ContextContractError)):
        write_report_tree(state, "BTC-USD", tmp_path)
    assert not (tmp_path / "complete_report.md").exists()


@pytest.mark.parametrize(
    "patch",
    [
        {"asset_type": "stock"},
        {"company_of_interest": "ETH-USD"},
        {"trade_date": "2026-01-11"},
    ],
)
def test_wrong_run_rejected(evidence, tmp_path, patch):
    with pytest.raises(ContextContractError):
        write_report_tree(state_for(evidence) | patch, "BTC-USD", tmp_path)


def test_valid_but_mismatched_summary_rejected(evidence, tmp_path):
    summary = evidence.summary().model_dump(mode="json")
    summary["streams"][0]["bar_count"] = 31
    state = state_for(evidence) | {"crypto_pattern_report": json.dumps(summary)}
    with pytest.raises(ContextContractError, match="consistency"):
        write_report_tree(state, "BTC-USD", tmp_path)


def test_latest_only_compact_section_and_unchanged_decision(evidence, tmp_path):
    raw = evidence.model_dump()
    older = raw["streams"][0]["events"][0] | {
        "confirmation_at": NOW - timedelta(days=1),
        "signal_at": NOW - timedelta(days=1),
    }
    raw["streams"][0]["events"] += (older,)
    expanded = CryptoPatternEvidence.model_validate(raw)
    state = state_for(expanded)
    state["spot_allocation_data"] = calculate_spot_allocation(
        PortfolioContext(), "BTC-USD"
    ).model_dump_json()
    baseline = decision_summary(
        {k: v for k, v in state.items() if not k.startswith("crypto_pattern")}
    )

    path = write_report_tree(state, "BTC-USD", tmp_path)
    section = (tmp_path / "6_patterns/patterns.md").read_text()
    assert decision_summary(state) == baseline
    assert (tmp_path / "0_summary/decision.md").read_text() == baseline
    assert section.count("- Pattern ") == 1
    assert "available 6; quarantined 210; unsupported 84" in section
    assert "anchor status provider_not_exposed" in section
    assert "Evidence only" in section and "Missing conditions: none" in section
    assert not re.search(
        r"\b(buy|sell|hold|order|forecast|probability|score|rating|portfolio|journal)\b",
        section,
        re.I,
    )
    for forbidden in ('"events"', '"latest_events"', '"candles"', '"open"', '"close"'):
        assert forbidden not in path.read_text()
        assert forbidden not in (tmp_path / "complete_report.html").read_text()


@pytest.mark.parametrize(
    "status",
    [
        "unavailable",
        "insufficient_history",
        "missing_bars",
        "execution_error",
    ],
)
def test_failures_exposed_without_exception_payload(evidence, tmp_path, status):
    raw = evidence.model_dump()
    row = raw["streams"][0]
    row.update(status=status, events=(), bar_count=0, first_bar_at=None, last_bar_at=None)
    row["detector_errors"] = ("SECRET " * 1000,) * 1200
    if status == "unavailable":
        row["failure"] = {"kind": "transport", "detail": "SECRET " * 1000}
    state = state_for(CryptoPatternEvidence.model_validate(raw))

    write_report_tree(state, "BTC-USD", tmp_path)
    section = (tmp_path / "6_patterns/patterns.md").read_text()
    assert f"Status: {status}" in section
    assert "No certified latest-bar pattern evidence" in section
    assert "Detector errors: 999+" in section
    assert "SECRET" not in section
    assert len(section) < 1500
    assert observed_sources(state) == ""


def test_bounded_event_count_and_provenance(evidence, tmp_path):
    raw = evidence.model_dump()
    row = raw["streams"][0]
    row["events"] = [row["events"][0] | {"pattern_id": f"pattern{i:03d}"} for i in range(100)]
    row["provenance"]["endpoint"] = "<script>\n" * 1000
    write_report_tree(state_for(CryptoPatternEvidence.model_validate(raw)), "BTC-USD", tmp_path)
    section = (tmp_path / "6_patterns/patterns.md").read_text()
    assert section.count("- Pattern ") == 18
    assert "Additional latest events omitted: 82" in section
    assert len(section) < 6500
    assert "<script>" not in section


@pytest.mark.parametrize(
    "source,market,expected",
    [
        ("binance", "spot", "Binance Spot"),
        ("binance", "perpetual", "Binance USD-M"),
        ("deribit", "perpetual", ""),
    ],
)
def test_observed_source_uses_validated_provenance(evidence, tmp_path, source, market, expected):
    raw = evidence.model_dump()
    row = raw["streams"][0]
    row["market"] = market
    row["events"] = ()
    row["provenance"]["source"] = source
    state = state_for(CryptoPatternEvidence.model_validate(raw))
    write_report_tree(state, "BTC-USD", tmp_path)
    assert observed_sources(state) == expected
    for name in ("complete_report.md", "complete_report.html"):
        text = (tmp_path / name).read_text()
        assert ("Observed evidence" in text) == bool(expected)
        if expected:
            assert expected in text


def test_markdown_html_deterministic_with_fixed_clock(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.reporting.datetime", type("Clock", (), {"now": lambda: NOW}))
    monkeypatch.setattr(
        "tradingagents.report_html.datetime", type("Clock", (), {"now": lambda: NOW})
    )
    state = state_for(evidence)
    write_report_tree(state, "BTC-USD", tmp_path)
    before = {
        name: (tmp_path / name).read_bytes()
        for name in ("complete_report.md", "complete_report.html")
    }
    write_report_tree(state, "BTC-USD", tmp_path)
    assert before == {name: (tmp_path / name).read_bytes() for name in before}
    assert b"Classic Pattern Evidence" in before["complete_report.html"]


def test_historical_events_do_not_claim_latest_bar_evidence(evidence, tmp_path):
    raw = evidence.model_dump()
    raw["streams"][0]["events"][0].update(
        confirmation_at=NOW - timedelta(days=1), signal_at=NOW - timedelta(days=1)
    )
    write_report_tree(state_for(CryptoPatternEvidence.model_validate(raw)), "BTC-USD", tmp_path)
    section = (tmp_path / "6_patterns/patterns.md").read_text()
    assert "No certified latest-bar pattern evidence" in section
    assert "- Pattern " not in section
