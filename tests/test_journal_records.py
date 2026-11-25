"""Storage contracts for mixed legacy and versioned decision journals."""

import json
from pathlib import Path

import pytest

from tests.settlement_fixtures import outcome as detailed_outcome
from tradingagents.memory import TradingMemoryLog, journal_records as codec


@pytest.fixture
def journal(tmp_path: Path) -> TradingMemoryLog:
    return TradingMemoryLog({"memory_log_path": str(tmp_path / "journal.md")})


def test_prose_cannot_inject_records_or_sections(journal: TradingMemoryLog) -> None:
    text = (
        " \nREFLECTION:\nfalse"
        + codec.SEPARATOR
        + ("[2026-01-02 | FAKE | Buy | pending]\nDECISION:\n\x1e{}\n\u2603\r\n ")
    )
    journal.store_decision(text, "2026-01-01", text, rating=text)
    pending = journal.get_pending_entries()
    assert len(pending) == 1
    assert (pending[0]["ticker"], pending[0]["rating"], pending[0]["decision"]) == (text,) * 3
    assert journal.update_with_outcome(text, "2026-01-01", 0.1, -0.0, 5, text)
    entry = journal.load_entries()[0]
    assert entry["decision"] == text and entry["reflection"] == text
    assert journal.get_pending_entries() == []


@pytest.mark.parametrize("value", [5e-324, 1.2345678901234567e-19, -0.0])
@pytest.mark.parametrize("batch", [False, True])
def test_exact_float_roundtrip(journal: TradingMemoryLog, value: float, batch: bool) -> None:
    journal.store_decision("BTC", "2026-01-01", "Rating: Buy")
    update = codec.OutcomeUpdate(
        ticker="BTC",
        trade_date="2026-01-01",
        raw_return=value,
        alpha_return=value,
        holding_days=5,
        reflection="exact",
    )
    if batch:
        journal.batch_update_with_outcomes([update])
    else:
        journal.update_with_outcome(**update)
    entry = journal.load_entries()[0]
    assert entry["raw_return"].hex() == value.hex()
    assert entry["alpha_return"].hex() == value.hex()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("batch", [False, True])
def test_nonfinite_update_keeps_original_file(
    tmp_path: Path, journal: TradingMemoryLog, value: float, batch: bool
) -> None:
    journal.store_decision("BTC", "2026-01-01", "Rating: Buy")
    path = tmp_path / "journal.md"
    before = path.read_bytes()
    updates = [
        codec.OutcomeUpdate(
            ticker="BTC",
            trade_date="2026-01-01",
            raw_return=value,
            alpha_return=0.0,
            holding_days=5,
            reflection="invalid",
        )
    ]
    with pytest.raises(ValueError, match="Out of range float"):
        if batch:
            journal.batch_update_with_outcomes(updates)
        else:
            journal.update_with_outcome(**updates[0])
    assert path.read_bytes() == before
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("suffix", ["", " | resolved:2026-01-07"])
def test_legacy_numeric_fallback_has_unknown_provenance(suffix: str) -> None:
    data = (
        f"[2026-01-01 | BTC | Buy | +1.2% | -0.0% | 5d{suffix}]\n\n"
        "DECISION:\nold\n\nREFLECTION:\nlesson" + codec.SEPARATOR
    ).encode()
    entry = codec.decode(data)[0].entry
    assert entry["raw_return"].hex() == (0.012).hex()
    assert entry["alpha_return"].hex() == (-0.0).hex()
    assert entry["holding_days"] == 5
    assert entry["provenance"] is None
    assert entry["resolved"] == ("2026-01-07" if suffix else None)
    assert "source" not in entry["outcome"] and "benchmark" not in entry["outcome"]


def test_mixed_order_duplicates_and_legacy_upgrade(
    tmp_path: Path, journal: TradingMemoryLog
) -> None:
    legacy = "[2026-01-03 | OLD | Buy | pending]\n\nDECISION:\nold" + codec.SEPARATOR
    path = tmp_path / "journal.md"
    path.write_bytes(legacy.replace("\n", "\r\n").encode())
    journal.store_decision("NEW", "2026-01-02", "Rating: Hold")
    with path.open("ab") as file:
        file.write(
            (
                "[2026-01-01 | LAST | Sell | +1.0% | +0.5% | 5d]\n\n"
                "DECISION:\nlast" + codec.SEPARATOR
            ).encode()
        )
    before = path.read_bytes()
    for ticker, date in (("OLD", "2026-01-03"), ("NEW", "2026-01-02"), ("LAST", "2026-01-01")):
        journal.store_decision(ticker, date, "duplicate")
    assert path.read_bytes() == before
    journal.update_with_outcome("OLD", "2026-01-03", 5e-324, -0.0, 5, "settled")
    entries = journal.load_entries()
    assert [entry["ticker"] for entry in entries] == ["OLD", "NEW", "LAST"]
    assert entries[0]["raw_return"].hex() == (5e-324).hex()
    assert entries[0]["provenance"] is None


@pytest.mark.parametrize("batch", [False, True])
def test_opaque_bytes_survive_updates_and_rotation(tmp_path: Path, batch: bool) -> None:
    path = tmp_path / "journal.md"
    journal = TradingMemoryLog({"memory_log_path": str(path), "memory_log_max_entries": 1})
    opaque = (
        b'\x1e {"schema":"tradingagents.decision-journal","version":99,"x":-0.0}\r\n'
        b"\x1e{broken\r\n\xff\n"
        b'\x1e{"schema":"tradingagents.decision-journal","version":2,"date":5}\n'
    )
    path.write_bytes(opaque)
    journal.store_decision("OLD", "2026-01-01", "Rating: Buy")
    journal.update_with_outcome("OLD", "2026-01-01", 0.1, 0.0, 5, "old")
    journal.store_decision("KEEP", "2026-02-01", "Rating: Hold")
    journal.store_decision("PENDING", "2026-03-01", "Rating: Hold")
    update = codec.OutcomeUpdate(
        ticker="KEEP",
        trade_date="2026-02-01",
        raw_return=0.2,
        alpha_return=0.1,
        holding_days=5,
        reflection="keep",
    )
    if batch:
        journal.batch_update_with_outcomes([update])
    else:
        journal.update_with_outcome(**update)
    assert path.read_bytes().startswith(opaque)
    assert [entry["ticker"] for entry in journal.load_entries()] == ["KEEP", "PENDING"]


def test_resolved_extensions_roundtrip_without_loss() -> None:
    document = codec.pending_document("2026-01-01", "BTC", "Buy", "decision")
    document["provenance"] = {"source": "binance", "benchmark": None, "price": 5e-324}
    document["extension"] = ["keep", -0.0]
    record = codec.decode(codec.encode(document))[0]
    outcome: codec.JSONObject = {
        "raw_return": 5e-324,
        "alpha_return": -0.0,
        "holding_days": 5,
        "resolution_date": "2026-01-07",
        "entry_price": 123.45678901234567,
        "benchmark": {"source": "exact", "price": -0.0},
    }
    resolved = codec.resolve(record, outcome, "reflection")
    result = codec.decode(resolved)[0]
    assert result.document == {**document, "outcome": outcome, "reflection": "reflection"}
    assert json.loads(resolved[1:])["outcome"]["raw_return"].hex() == (5e-324).hex()
    assert codec.encode(result.document) == resolved


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "section,field,value",
    [
        ("outcome", "holding_days", -5),
        ("outcome", "holding_days", 0),
        ("document", "date", "2026-02-30"),
        ("outcome", "resolution_date", "2026-13-12"),
        ("outcome", "resolution_date", "2026-01-11"),
        ("outcome", "holding_days", 6),
        ("outcome", "alpha_return", 0.4),
        ("outcome", "raw_return", float("inf")),
        ("outcome", "alpha_return", float("nan")),
        ("outcome", "resolution_date", "2026-01-04"),
        ("outcome", "benchmark_return", 0.4),
        ("window", "count", 6),
        ("window", "kind", "other"),
        ("window", "kind", "utc_days"),
        ("asset", "exit_price", 120.0),
        ("asset", "entry_date", "2026-02-30"),
        ("asset", "entry_price", True),
        ("benchmark", "provider", "other"),
        ("scoring", "value", -0.02),
        ("scoring", "direction", -1.0),
        ("scoring", "hit", False),
        ("scoring", "version", True),
        ("scoring", "value", 0.020000000000000004),
    ],
)
def test_malformed_v2_stays_opaque_through_update_and_rotation(
    tmp_path: Path, batch: bool, section: str, field: str, value: codec.JSONValue
) -> None:
    detailed = detailed_outcome()
    document = codec.pending_document("2026-01-05", "BAD", "Buy", "bad")
    facts = codec.outcome_document(
        codec.OutcomeUpdate(
            ticker="BAD",
            trade_date="2026-01-05",
            raw_return=detailed.raw_return,
            alpha_return=detailed.alpha_return,
            holding_days=5,
            reflection="bad",
            detailed_outcome=detailed,
        ),
        "Buy",
    )
    document.update(outcome=facts, reflection="bad")
    target = (
        document if section == "document" else facts if section == "outcome" else facts[section]
    )
    assert isinstance(target, dict)
    target[field] = value
    opaque = b"\x1e " + json.dumps(document, indent=None).encode() + b"\r\n"
    path = tmp_path / "journal.md"
    path.write_bytes(opaque)
    journal = TradingMemoryLog({"memory_log_path": str(path), "memory_log_max_entries": 1})
    for ticker, day in (("OLD", "2026-02-01"), ("KEEP", "2026-03-01")):
        journal.store_decision(ticker, day, "Rating: Buy")
        update = codec.OutcomeUpdate(
            ticker=ticker,
            trade_date=day,
            raw_return=0.1,
            alpha_return=0.1,
            holding_days=5,
            reflection="settled",
        )
        if batch:
            journal.batch_update_with_outcomes([update])
        else:
            journal.update_with_outcome(**update)

    assert path.read_bytes().startswith(opaque)
    record = codec.decode(path.read_bytes())[0]
    assert record.raw == opaque
    assert record.entry is None and record.document is None
    assert [entry["ticker"] for entry in journal.load_entries()] == ["KEEP"]
