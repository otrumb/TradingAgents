from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from tradingagents.agents import market_tools, tools
from tradingagents.agents.analysts import market_analyst
from tradingagents.dataflows.crypto_spot_models import CompletedDailyCandle, SpotPriceContext


def _spot_context() -> SpotPriceContext:
    server_time = datetime(2026, 6, 1, 0, 0, 5, 1_000, tzinfo=UTC)
    cutoff = server_time - timedelta(seconds=5)
    start = cutoff.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=80)
    candles = []
    for index in range(60):
        open_time = start + timedelta(days=index)
        close = Decimal(100 + index)
        candles.append(
            CompletedDailyCandle(
                asset="BTC",
                symbol="BTCUSDT",
                retrieved_at=server_time,
                server_time=server_time,
                cutoff=cutoff,
                open_time=open_time,
                close_time=open_time + timedelta(days=1, milliseconds=-1),
                open=close - 1,
                high=close + 2,
                low=close - 2,
                close=close,
                volume=1000 + index,
            )
        )
    return SpotPriceContext(asset="BTC", daily_candles=tuple(candles))


@pytest.mark.parametrize(
    "indicator",
    ("rsi", "macd", "close_10_ema", "boll", "boll_ub", "boll_lb", "atr"),
)
def test_crypto_tools_share_finalized_binance_rows(monkeypatch, indicator):
    context = _spot_context()
    payload = context.model_dump_json()

    def forbidden(*args, **kwargs):
        pytest.fail("crypto market tool routed to stock vendor")

    monkeypatch.setattr(market_tools, "route_to_vendor", forbidden)
    raw = tools.get_stock_data.func(
        "BTC-USD",
        "2000-01-01",
        datetime.now(UTC).date().isoformat(),
        trade_date=datetime.now(UTC).date().isoformat(),
        asset_type="crypto",
        crypto_market_data=payload,
    )
    indicator_result = tools.get_indicators.func(
        "BTC-USD",
        indicator,
        "2026-06-01",
        trade_date="2026-06-01",
        asset_type="crypto",
        crypto_market_data=payload,
    )
    snapshot = tools.get_verified_market_snapshot.func(
        "BTC-USD",
        datetime.now(UTC).date().isoformat(),
        trade_date=datetime.now(UTC).date().isoformat(),
        asset_type="crypto",
        crypto_market_data=payload,
    )

    latest = context.daily_candles[-1]
    assert latest.open_time.date().isoformat() in raw
    assert str(latest.close) in raw
    assert latest.open_time.date().isoformat() in indicator_result
    assert indicator in indicator_result
    assert f"{latest.open_time:%Y-%m-%d}: N/A" not in indicator_result
    assert f"Latest trading row used: {latest.open_time.date().isoformat()}" in snapshot
    for result in (raw, indicator_result, snapshot):
        assert "Binance Spot" in result
        assert "finality: closed" in result
        assert "intraday_not_daily_close" not in result


def test_historical_crypto_market_does_not_fetch_current_data(monkeypatch):
    def forbidden(request):
        pytest.fail("historical market analyst fetched current crypto data")

    monkeypatch.setattr("tradingagents.dataflows.crypto_spot.collect_spot_prices", forbidden)
    monkeypatch.setattr(
        market_analyst,
        "take_turn",
        lambda prompt, llm, offered_tools, messages: (AIMessage("unavailable"), "unavailable"),
    )
    node = market_analyst.create_market_analyst(FakeListChatModel(responses=["unavailable"]))
    result = node(
        {
            "messages": [],
            "company_of_interest": "BTC-USD",
            "asset_type": "crypto",
            "trade_date": "2000-01-01",
        }
    )

    assert result["market_report"] == "unavailable"
    assert result["crypto_market_data"] == ""


def test_stock_market_tools_keep_vendor_routing(monkeypatch):
    calls = []

    def route(*args):
        calls.append(args)
        return "stock result"

    monkeypatch.setattr(market_tools, "route_to_vendor", route)

    assert (
        tools.get_stock_data.func("AAPL", "2026-01-01", "2026-01-02", trade_date="2026-01-02")
        == "stock result"
    )
    assert (
        tools.get_indicators.func("AAPL", "rsi", "2026-01-02", trade_date="2026-01-02")
        == "stock result"
    )
    assert calls == [
        ("get_stock_data", "AAPL", "2026-01-01", "2026-01-02"),
        ("get_indicators", "AAPL", "rsi", "2026-01-02", 30),
    ]
