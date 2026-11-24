"""Read-only collector adapter and crypto analyst, with current-date isolation."""

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.state import AgentState
from tradingagents.dataflows.crypto_models import SnapshotRequest
from tradingagents.dataflows.crypto_spot_models import CryptoAnalystContext
from tradingagents.dataflows.symbols import crypto_base


class CryptoReport(TypedDict):
    messages: list[AIMessage]
    crypto_report: str


def collect_crypto_report(ticker: str, trade_date: str) -> str:
    """Serve today's public context only; never fetch live data for other dates.

    Collector contract: crypto_context.collect_crypto_context(SnapshotRequest) returns
    crypto_models.CryptoContext, whose render() returns canonical JSON.
    """
    if date.fromisoformat(trade_date) != datetime.now(UTC).date():
        return (
            f"Crypto context unavailable for {trade_date}: collector supplies current "
            "snapshots only; historical point-in-time data and Binance Spot closed UTC "
            "candles/intraday prices are unavailable. No current endpoints were fetched."
        )
    asset = crypto_base(ticker) or ticker.strip().upper()
    if asset not in {"BTC", "ETH", "SOL"}:
        return f"Crypto context unavailable for {ticker}: collector supports BTC, ETH and SOL only."
    from tradingagents.dataflows.crypto_context import collect_crypto_context
    from tradingagents.dataflows.crypto_spot import collect_spot_prices

    request = SnapshotRequest.model_validate({"asset": asset})
    context = collect_crypto_context(request)
    return CryptoAnalystContext(
        asset=request.asset,
        snapshots=context.snapshots,
        unknowns=context.unknowns,
        spot_prices=collect_spot_prices(request),
    ).render()


def create_crypto_analyst(llm: BaseChatModel) -> Callable[[AgentState], CryptoReport]:
    """Pre-fetch public context; model receives no exchange or external tools."""

    def crypto_node(state: AgentState) -> CryptoReport:
        if state.get("asset_type", "stock") != "crypto":
            raise CryptoAssetRequiredError()
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        context = collect_crypto_report(ticker, trade_date)
        if date.fromisoformat(trade_date) != datetime.now(UTC).date():
            report = context
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        "You are a read-only crypto analyst. Analyze only supplied public "
                        "snapshots: positioning, derivatives, network activity, venue coverage, "
                        "timestamps, freshness and unknowns. Missing data is unavailable, not zero. "
                        "Snapshots describe current context, not historical evidence. Do not "
                        "mix spot_prices.daily_candles (eligible completed UTC days) with "
                        "spot_prices.intraday (current price, never a daily close). State latest "
                        "eligible candle date, close, Binance Spot source, USDT quote, UTC interval, "
                        "server/retrieval times, strict cutoff and 5000ms finality lag. Label "
                        "intraday price separately with retrieval time and absent source timestamp. "
                        "Preserve failures and insufficient history; Yahoo adjusted/local-date rows "
                        "are not exchange-final UTC candles. Do not "
                        "invent observations, call tools, or place orders."
                        + get_language_instruction()
                    ),
                    HumanMessage(f"Asset: {ticker}\nAnalysis date: {trade_date}\n{context}"),
                ]
            )
            report = (
                response.text + "\n\nPublic crypto evidence (provenance and finality):\n" + context
            )
        return {"messages": [AIMessage(content=report)], "crypto_report": report}

    return crypto_node


class CryptoAssetRequiredError(ValueError):
    """Crypto analyst was selected for an incompatible asset."""

    def __init__(self) -> None:
        super().__init__("crypto analyst is only available for asset_type crypto")
