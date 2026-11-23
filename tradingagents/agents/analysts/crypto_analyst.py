"""Read-only collector adapter and crypto analyst, with current-date isolation."""

from collections.abc import Callable
from datetime import date
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.state import AgentState
from tradingagents.dataflows.crypto_models import SnapshotRequest
from tradingagents.dataflows.symbols import crypto_base


class CryptoReport(TypedDict):
    messages: list[AIMessage]
    crypto_report: str


def collect_crypto_report(ticker: str, trade_date: str) -> str:
    """Serve today's public context only; never fetch live data for other dates.

    Collector contract: crypto_context.collect_crypto_context(SnapshotRequest) returns
    crypto_models.CryptoContext, whose render() returns canonical JSON.
    """
    if date.fromisoformat(trade_date) != date.today():
        return (
            f"Crypto context unavailable for {trade_date}: collector supplies current "
            "snapshots only; historical point-in-time data is unavailable."
        )
    asset = crypto_base(ticker) or ticker.strip().upper()
    if asset not in {"BTC", "ETH", "SOL"}:
        return f"Crypto context unavailable for {ticker}: collector supports BTC, ETH and SOL only."
    from tradingagents.dataflows.crypto_context import collect_crypto_context

    request = SnapshotRequest.model_validate({"asset": asset})
    return collect_crypto_context(request).render()


def create_crypto_analyst(llm: BaseChatModel) -> Callable[[AgentState], CryptoReport]:
    """Pre-fetch public context; model receives no exchange or external tools."""

    def crypto_node(state: AgentState) -> CryptoReport:
        if state.get("asset_type", "stock") != "crypto":
            raise CryptoAssetRequiredError()
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        context = collect_crypto_report(ticker, trade_date)
        if date.fromisoformat(trade_date) != date.today():
            report = context
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        "You are a read-only crypto analyst. Analyze only supplied public "
                        "snapshots: positioning, derivatives, network activity, venue coverage, "
                        "timestamps, freshness and unknowns. Missing data is unavailable, not zero. "
                        "Snapshots describe current context, not historical evidence. Do not "
                        "invent observations, call tools, or place orders."
                        + get_language_instruction()
                    ),
                    HumanMessage(f"Asset: {ticker}\nAnalysis date: {trade_date}\n{context}"),
                ]
            )
            report = response.text
        return {"messages": [AIMessage(content=report)], "crypto_report": report}

    return crypto_node


class CryptoAssetRequiredError(ValueError):
    """Crypto analyst was selected for an incompatible asset."""

    def __init__(self) -> None:
        super().__init__("crypto analyst is only available for asset_type crypto")
