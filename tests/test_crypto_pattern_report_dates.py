from datetime import UTC, datetime

import pytest

from tradingagents.crypto_pattern_report import validate_patterns
from tradingagents.dataflows.crypto_models import ContextContractError
from tradingagents.dataflows.crypto_pattern_models import CryptoPatternEvidence
from tradingagents.reporting import write_report_tree


@pytest.fixture
def current_state(monkeypatch) -> dict[str, str]:
    class Clock:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return datetime(2026, 1, 11, 0, 30, tzinfo=UTC)

    monkeypatch.setattr("tradingagents.crypto_pattern_report.datetime", Clock)
    evidence = CryptoPatternEvidence.model_validate(
        {
            "asset": "BTC",
            "analysis_date": None,
            "streams": [],
            "catalog": {"total_count": 0, "dispositions": []},
        }
    )
    return {
        "asset_type": "crypto",
        "company_of_interest": "BTC-USD",
        "trade_date": "2026-01-11",
        "crypto_pattern_evidence": evidence.model_dump_json(),
        "crypto_pattern_report": evidence.summary().model_dump_json(),
    }


@pytest.mark.parametrize(
    "analysis_date,trade_date,accepted",
    [
        (None, "2026-01-11", True),
        (None, "2026-01-10", False),
        (None, "2026-01-12", False),
        ("2026-01-10", "2026-01-10", True),
        ("2026-01-10", "2026-01-11", False),
        ("2026-01-10", "2026-01-12", False),
        ("2026-01-11", "2026-01-11", True),
        (None, "", False),
        ("2026-01-10", "", False),
    ],
)
def test_date_identity_requires_utc_today_or_exact_historical_date(
    current_state, analysis_date, trade_date, accepted
):
    evidence = CryptoPatternEvidence.model_validate_json(current_state["crypto_pattern_evidence"])
    evidence = CryptoPatternEvidence.model_validate(
        evidence.model_dump() | {"analysis_date": analysis_date}
    )
    state = current_state | {
        "trade_date": trade_date,
        "crypto_pattern_evidence": evidence.model_dump_json(),
        "crypto_pattern_report": evidence.summary().model_dump_json(),
    }

    if accepted:
        assert validate_patterns(state) == evidence
    else:
        with pytest.raises(ContextContractError, match="pattern run identity/date"):
            validate_patterns(state)


def test_current_evidence_writes_markdown_html_and_sidecar(current_state, tmp_path):
    path = write_report_tree(current_state, "BTC-USD", tmp_path)

    assert "Classic Pattern Evidence" in path.read_text(encoding="utf-8")
    assert "Classic Pattern Evidence" in (tmp_path / "complete_report.html").read_text(
        encoding="utf-8"
    )
    saved = CryptoPatternEvidence.model_validate_json(
        (tmp_path / "evidence/crypto_patterns.json").read_text(encoding="utf-8")
    )
    assert saved.analysis_date is None
