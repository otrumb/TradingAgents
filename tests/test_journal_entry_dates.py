import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.settlement_fixtures import outcome
from tradingagents.memory import TradingMemoryLog, journal_records as codec


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("kind", ["trading_sessions", "utc_days"])
@pytest.mark.parametrize("decision_date", ["2026-01-05", "2026-01-06", "2026-01-07"])
def test_asset_entry_after_decision_controls_recognition_and_rotation(
    tmp_path: Path, batch: bool, kind: str, decision_date: str
) -> None:
    detailed = outcome(resolved="2026-01-11")
    if kind == "utc_days":
        detailed = replace(detailed, window_kind=kind)
    else:
        detailed = replace(detailed, benchmark=replace(detailed.benchmark, entry_date="2026-01-02"))
    document = codec.pending_document(decision_date, "CHECK", "Buy", "decision")
    document.update(
        outcome=codec.outcome_document(
            codec.OutcomeUpdate(
                ticker="CHECK",
                trade_date=decision_date,
                raw_return=detailed.raw_return,
                alpha_return=detailed.alpha_return,
                holding_days=5,
                reflection="settled",
                detailed_outcome=detailed,
            ),
            "Buy",
        ),
        reflection="settled",
    )
    raw = b"\x1e " + json.dumps(document).encode() + b"\r\n"
    record = codec.decode(raw)[0]
    valid = decision_date == "2026-01-05"
    assert (record.entry is not None) == valid
    assert (record.document is not None) == valid
    path = tmp_path / "journal.md"
    path.write_bytes(raw)
    journal = TradingMemoryLog({"memory_log_path": str(path), "memory_log_max_entries": 1})
    journal.store_decision("KEEP", "2026-02-01", "Rating: Buy")
    update = codec.OutcomeUpdate(
        ticker="KEEP",
        trade_date="2026-02-01",
        raw_return=0.1,
        alpha_return=0.1,
        holding_days=5,
        reflection="settled",
    )

    if batch:
        journal.batch_update_with_outcomes([update])
    else:
        journal.update_with_outcome(**update)

    assert [entry["ticker"] for entry in journal.load_entries()] == ["KEEP"]
    if valid:
        assert not path.read_bytes().startswith(raw)
    else:
        assert path.read_bytes().startswith(raw)
        retained = codec.decode(path.read_bytes())[0]
        assert retained.raw == raw
        assert retained.entry is None and retained.document is None
