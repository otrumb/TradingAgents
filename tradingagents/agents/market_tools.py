from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.dataflows.crypto_market import (
    UnsupportedCryptoIndicatorError,
    crypto_indicators,
    crypto_stock_data,
    crypto_verified_snapshot,
)
from tradingagents.dataflows.date_window import as_of, as_of_window
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.router import no_data_available, vendor_unavailable
from tradingagents.dataflows.vendors.yahoo.snapshot import build_verified_market_snapshot


def route_to_vendor(*args):
    from tradingagents.agents import tools

    return tools.route_to_vendor(*args)


@tool
def get_stock_data(
    symbol: Annotated[str, InjectedState("company_of_interest")],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
    asset_type: Annotated[str, InjectedState("asset_type")] = "stock",
    crypto_market_data: Annotated[str, InjectedState("crypto_market_data")] = "",
) -> str:
    """Retrieve OHLCV for the run instrument over the requested date range."""
    try:
        start_date, end_date = as_of_window(start_date, end_date, trade_date)
    except ValueError as exc:
        return str(exc)
    if asset_type == "crypto":
        try:
            return crypto_stock_data(crypto_market_data, symbol, start_date, end_date)
        except NoMarketDataError as exc:
            return no_data_available(exc)
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)


@tool
def get_indicators(
    symbol: Annotated[str, InjectedState("company_of_interest")],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
    asset_type: Annotated[str, InjectedState("asset_type")] = "stock",
    crypto_market_data: Annotated[str, InjectedState("crypto_market_data")] = "",
) -> str:
    """Retrieve technical indicator values for the run instrument.

    Crypto supports rsi, macd, macds, macdh, close_10_ema, close_50_sma,
    close_200_sma, boll, boll_ub, boll_lb, atr and vwma; renders up to 30 rows.
    """
    curr_date = as_of(curr_date, trade_date)
    indicators = [item.strip().lower() for item in indicator.split(",") if item.strip()]
    results = []
    for selected in indicators:
        try:
            if asset_type == "crypto":
                results.append(
                    crypto_indicators(
                        crypto_market_data, symbol, selected, curr_date, look_back_days
                    )
                )
            else:
                results.append(
                    route_to_vendor("get_indicators", symbol, selected, curr_date, look_back_days)
                )
        except NoMarketDataError as exc:
            results.append(no_data_available(exc))
        except UnsupportedCryptoIndicatorError as exc:
            results.append(str(exc))
        except ValueError as exc:
            if asset_type == "crypto":
                raise
            results.append(str(exc))
    return "\n\n".join(results)


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, InjectedState("company_of_interest")],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int, "number of recent trading rows to include for sanity-checking"
    ] = 30,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
    asset_type: Annotated[str, InjectedState("asset_type")] = "stock",
    crypto_market_data: Annotated[str, InjectedState("crypto_market_data")] = "",
) -> str:
    """Return latest OHLCV and common indicators from one verified row set."""
    try:
        current_date = as_of(curr_date, trade_date)
        if asset_type == "crypto":
            return crypto_verified_snapshot(
                crypto_market_data, symbol, current_date, look_back_days
            )
        return build_verified_market_snapshot(symbol, current_date, look_back_days)
    except VendorUnavailableError as exc:
        return vendor_unavailable("get_verified_market_snapshot", exc)
    except NoMarketDataError as exc:
        return no_data_available(exc)
