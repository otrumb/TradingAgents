import math
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from tests.settlement_fixtures import outcome
from tradingagents.dataflows.settlement_models import SettlementContractError
from tradingagents.memory import TradingMemoryLog, journal_records as codec, settlement


def test_stock_calendar_legs_and_resolved_symbols(monkeypatch, tmp_path):
    prices = {
        "XAUUSD+": pd.Series(
            [100.125, 110.25, 120.375],
            index=pd.to_datetime(
                [
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ]
            ),
        ),
        "SPX500": pd.Series(
            [200.5, 220.75],
            index=pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-07",
                ]
            ),
        ),
    }
    monkeypatch.setattr(settlement, "get_closes", lambda symbol, *args: prices[symbol])
    monkeypatch.setattr(settlement, "_now", lambda: pd.Timestamp("2026-02-01", tz="UTC"))
    detailed = settlement.fetch_outcome("XAUUSD+", "2026-01-03", 2, "SPX500")
    assert detailed.asset.symbol == "GC=F"
    assert detailed.asset.provider == detailed.benchmark.provider == "yahoo"
    assert (detailed.asset.entry_date, detailed.asset.exit_date) == ("2026-01-04", "2026-01-06")
    assert (detailed.asset.entry_price, detailed.asset.exit_price) == (100.125, 120.375)
    assert detailed.benchmark.symbol == "^GSPC"
    assert (detailed.benchmark.entry_date, detailed.benchmark.exit_date) == ("2026-01-02",) * 2
    assert (detailed.benchmark.entry_price, detailed.benchmark.exit_price) == (200.5,) * 2
    assert detailed.benchmark_return == 0.0
    assert detailed.resolution_date == "2026-01-06"
    assert (detailed.window_kind, detailed.window_count) == ("trading_sessions", 2)
    assert settlement.fetch_returns("XAUUSD+", "2026-01-03", 2, "SPX500") == detailed.legacy_tuple()


@pytest.mark.parametrize(
    "rating,direction",
    [
        ("Buy", 1),
        ("Overweight", 1),
        ("Sell", -1),
        ("Underweight", -1),
        ("Hold", None),
        ("REVIEW", None),
        ("unknown", None),
    ],
)
@pytest.mark.parametrize("alpha", [5e-324, -0.0, -0.01])
def test_scoring_uses_stored_rating_with_exact_alpha(tmp_path, rating, direction, alpha):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    log.store_decision("NVDA", "2026-01-05", "Rating: Buy", rating=rating)
    log.update_with_outcome("NVDA", "2026-01-05", alpha, alpha, 5, "lesson")
    entry = log.load_entries()[0]
    scoring = entry["outcome"]["scoring"]
    value = None if direction is None else alpha * direction
    assert scoring == {
        "method": "rating-direction",
        "version": 1,
        "basis": "alpha",
        "direction": direction,
        "value": value,
        "hit": None if value is None else value > 0,
    }
    assert entry["alpha_return"].hex() == alpha.hex()
    if value is not None:
        assert scoring["value"].hex() == value.hex()


@pytest.mark.parametrize("batch", [False, True])
def test_detailed_update_keeps_extensions_and_exact_metadata(tmp_path, batch):
    path = tmp_path / "m.md"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    document = codec.pending_document("2026-01-05", "NVDA", "Sell", "Rating: Buy")
    document["extension"] = {"opaque": [42, -0.0]}
    document["provenance"] = {"extension": "keep"}
    path.write_bytes(codec.encode(document))
    detailed = outcome()
    update = codec.OutcomeUpdate(
        ticker="NVDA",
        trade_date="2026-01-05",
        raw_return=detailed.raw_return,
        alpha_return=detailed.alpha_return,
        holding_days=5,
        reflection="lesson",
        detailed_outcome=detailed,
    )
    if batch:
        log.batch_update_with_outcomes([update])
    else:
        log.update_with_outcome(**update)
    record = codec.decode(path.read_bytes())[0]
    facts = record.entry["outcome"]
    assert facts["asset"] == {
        "provider": "yahoo",
        "symbol": "NVDA",
        "entry_date": "2026-01-06",
        "exit_date": "2026-01-12",
        "entry_price": 100.0,
        "exit_price": 105.0,
    }
    assert facts["benchmark"]["symbol"] == "SPY"
    assert facts["benchmark_label"] == "SPY"
    assert facts["benchmark_return"].hex() == detailed.benchmark_return.hex()
    assert facts["alpha_return"].hex() == detailed.alpha_return.hex()
    assert facts["window"] == {"kind": "trading_sessions", "count": 5}
    assert record.entry["resolved"] == detailed.resolution_date
    assert record.entry["provenance"] == {
        "extension": "keep",
        "asset": facts["asset"],
        "benchmark": facts["benchmark"],
        "window": facts["window"],
    }
    assert record.document["extension"] == document["extension"]
    assert facts["scoring"]["direction"] == -1


def test_invalid_batch_is_atomic_even_after_valid_metadata(tmp_path):
    path = tmp_path / "m.md"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    for ticker in ("NVDA", "BAD"):
        log.store_decision(ticker, "2026-01-05", "Rating: Buy")
    before = path.read_bytes()
    detailed = outcome()
    valid = codec.OutcomeUpdate(
        ticker="NVDA",
        trade_date="2026-01-05",
        raw_return=detailed.raw_return,
        alpha_return=detailed.alpha_return,
        holding_days=5,
        reflection="lesson",
        detailed_outcome=detailed,
    )
    with pytest.raises(ValueError, match="Out of range float"):
        log.batch_update_with_outcomes(
            [valid, {**valid, "ticker": "BAD", "detailed_outcome": None, "raw_return": math.inf}]
        )
    assert path.read_bytes() == before
    assert not path.with_suffix(".tmp").exists()


def test_models_are_frozen_and_reject_invalid_contracts():
    detailed = outcome()
    with pytest.raises(FrozenInstanceError):
        detailed.asset.symbol = "other"
    for value in (math.nan, math.inf, -math.inf, 0.0, -1.0):
        with pytest.raises(SettlementContractError):
            replace(detailed.asset, entry_price=value)
    for changes in ({"alpha_return": math.nan}, {"window_count": 0}, {"alpha_return": 0.5}):
        with pytest.raises(SettlementContractError):
            replace(detailed, **changes)
