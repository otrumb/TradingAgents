from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from tradingagents.dataflows.crypto_settlement import SettlementRequest
from tradingagents.memory import TradingMemoryLog, settlement


@pytest.mark.parametrize("holding_days", [0, -1])
def test_settlement_request_rejects_nonpositive_holding_days(holding_days):
    with pytest.raises(ValidationError, match="holding_days"):
        SettlementRequest(asset="BTC", decision_date=date(2026, 1, 2), holding_days=holding_days)


@pytest.mark.parametrize(
    "ticker,config,expected_days",
    [
        ("BTCUSD", {}, 7),
        ("ETHUSDT", {"holding_period_days": 2}, 7),
        ("SOL-USD", {"crypto_holding_period_days": 3, "holding_period_days": 2}, 3),
        ("NVDA", {"crypto_holding_period_days": 3}, 5),
    ],
)
def test_pending_uses_asset_specific_holding_period(
    monkeypatch,
    tmp_path,
    ticker,
    config,
    expected_days,
):
    def crypto_returns(ticker, trade_date, holding_days):
        assert holding_days == expected_days
        return 0.1, 0.1, holding_days, "2026-01-10"

    monkeypatch.setattr(settlement, "fetch_crypto_returns", crypto_returns)
    days = pd.bdate_range("2026-01-02", periods=7)
    prices = pd.Series(range(100, 107), index=days)
    monkeypatch.setattr(settlement, "get_closes", lambda *args: prices)
    monkeypatch.setattr(settlement, "_now", lambda: pd.Timestamp("2026-02-01", tz="UTC"))
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md")})
    log.store_decision(ticker, "2026-01-02", "Rating: Buy")

    class Reflector:
        def reflect_on_final_decision(self, **kwargs):
            assert kwargs["holding_days"] == expected_days
            return "holding period lesson"

    result = settlement.settle_pending(ticker, log, Reflector(), config)

    assert result.settled == [(ticker, "2026-01-02")]
    assert result.failed == []
