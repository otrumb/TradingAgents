"""Tests for deterministic instrument-identity resolution (#814) and the
context-anchored message placeholder (#888)."""

import unittest
from unittest.mock import patch

import pytest

from tradingagents.agents.context import (
    _identity,
    build_instrument_context,
    get_instrument_context_from_state,
    resolve_instrument_identity,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.parametrize(
    "ticker, name, pair",
    (
        ("BTC-USD", "Bitcoin", "BTCUSDT"),
        ("ETHUSDT", "Ethereum", "ETHUSDT"),
        ("SOL", "Solana", "SOLUSDT"),
    ),
)
def test_supported_crypto_identity_uses_binance_without_yfinance(ticker, name, pair):
    _identity.cache_clear()
    with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as vendor:
        identity = resolve_instrument_identity(ticker, "crypto")
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        context = graph.resolve_instrument_context(ticker, "crypto")
    vendor.assert_not_called()
    assert identity["company_name"] == name
    assert identity["exchange"] == "Binance Spot"
    assert identity["symbol"] == pair
    assert identity["quote_currency"] == "USDT"
    assert identity["source"] == "https://data-api.binance.vision"
    assert pair in context
    assert "Binance Spot" in context
    assert "https://data-api.binance.vision" in context


def test_stock_sol_keeps_yfinance_identity():
    _identity.cache_clear()
    with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as vendor:
        vendor.return_value.info = {"longName": "Emeren Group Ltd"}
        identity = resolve_instrument_identity("SOL", "stock")
    vendor.assert_called_once_with("SOL")
    assert identity["company_name"] == "Emeren Group Ltd"


@pytest.mark.unit
class ResolveInstrumentIdentityTests(unittest.TestCase):
    def setUp(self):
        _identity.cache_clear()

    def test_resolves_company_metadata_from_yfinance(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {
                "longName": "TOTO LTD.",
                "shortName": "TOTO",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
                "quoteType": "EQUITY",
            }
            identity = resolve_instrument_identity("totdy")
        mock.assert_called_once_with("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO LTD.")
        self.assertEqual(identity["sector"], "Industrials")
        self.assertEqual(identity["industry"], "Building Products & Equipment")
        self.assertEqual(identity["exchange"], "PNK")

    def test_falls_back_to_short_name(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"shortName": "TOTO", "sector": "Industrials"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO")

    def test_skips_placeholder_values(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "  ", "sector": "None", "industry": "n/a"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity, {})

    def test_fails_open_on_exception(self):
        with patch(
            "tradingagents.dataflows.vendors.yahoo.market.yf.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            self.assertEqual(resolve_instrument_identity("TOTDY"), {})

    def test_a_failed_lookup_is_asked_again(self):
        # A long process (a backtest) must not run every later cell without an
        # identity because one request failed.
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            type(mock.return_value).info = property(lambda self: (_ for _ in ()).throw(TimeoutError()))
            self.assertEqual(resolve_instrument_identity("TOTDY"), {})
            type(mock.return_value).info = {"longName": "TOTO LTD."}
            self.assertEqual(resolve_instrument_identity("TOTDY")["company_name"], "TOTO LTD.")

    def test_result_is_cached(self):
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "TOTO LTD."}
            first = resolve_instrument_identity("TOTDY")
            second = resolve_instrument_identity("TOTDY")
        mock.assert_called_once()  # second call served from cache
        self.assertEqual(first, second)


@pytest.mark.unit
class BuildInstrumentContextTests(unittest.TestCase):
    def test_mentions_exact_symbol_without_identity(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)
        self.assertNotIn("Resolved identity", context)

    def test_injects_resolved_identity(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {
                "company_name": "TOTO LTD.",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
            },
        )
        self.assertIn("Company: TOTO LTD.", context)
        self.assertIn("Industrials / Building Products & Equipment", context)
        self.assertIn("Exchange: PNK", context)
        self.assertIn("Do not substitute a different company", context)

    def test_crypto_uses_name_label_and_keeps_hint(self):
        context = build_instrument_context(
            "BTC-USD", "crypto", {"company_name": "Bitcoin USD"}
        )
        self.assertIn("Name: Bitcoin USD", context)
        self.assertIn("crypto asset rather than a company", context)


@pytest.mark.unit
class GetInstrumentContextFromStateTests(unittest.TestCase):
    def test_prefers_precomputed_context(self):
        state = {"company_of_interest": "TOTDY", "instrument_context": "PRECOMPUTED"}
        self.assertEqual(get_instrument_context_from_state(state), "PRECOMPUTED")

    def test_fallback_is_network_free_ticker_only(self):
        # No instrument_context and no yfinance call — must not hit the network.
        with patch("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker") as mock:
            context = get_instrument_context_from_state(
                {"company_of_interest": "NVDA", "asset_type": "stock"}
            )
        mock.assert_not_called()
        self.assertIn("NVDA", context)

    def test_fallback_respects_asset_type(self):
        context = get_instrument_context_from_state(
            {"company_of_interest": "BTC-USD", "asset_type": "crypto"}
        )
        self.assertIn("crypto asset", context)


if __name__ == "__main__":
    unittest.main()
