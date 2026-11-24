"""Read-only collector adapter and crypto analyst, with current-date isolation."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.state import AgentState
from tradingagents.dataflows.crypto_models import CryptoContext, SnapshotRequest
from tradingagents.dataflows.crypto_spot_models import CryptoAnalystContext, SpotPriceContext
from tradingagents.dataflows.symbols import crypto_base


class CryptoReport(TypedDict):
    messages: list[AIMessage]
    crypto_report: str
    crypto_context_data: str
    crypto_derivatives_data: str


def collect_crypto_report(ticker: str, trade_date: str, spot_prices: str = "") -> str:
    """Serve today's public context only; never fetch live data for other dates.

    Collector contract: crypto_context.collect_crypto_context(SnapshotRequest) returns
    crypto_models.CryptoContext, whose render() returns canonical JSON.
    """
    if date.fromisoformat(trade_date) != datetime.now(UTC).date():
        unavailable = (
            f"Crypto context unavailable for {trade_date}: collector supplies current "
            "snapshots only; historical positioning, options and current intraday metrics "
            "are skipped. No current endpoints were fetched."
        )
        if spot_prices:
            unavailable += (
                "\n\nBinance Spot evidence:\n```json\n"
                + json.dumps(
                    SpotPriceContext.model_validate_json(spot_prices)
                    .summary()
                    .model_dump(mode="json"),
                    sort_keys=True,
                    indent=2,
                )
                + "\n```"
            )
        return unavailable
    asset = crypto_base(ticker) or ticker.strip().upper()
    if asset not in {"BTC", "ETH", "SOL"}:
        return f"Crypto context unavailable for {ticker}: collector supports BTC, ETH and SOL only."
    from tradingagents.dataflows.crypto_context import collect_crypto_context

    request = SnapshotRequest.model_validate({"asset": asset})
    context = collect_crypto_context(request)
    return CryptoAnalystContext(
        asset=request.asset,
        snapshots=context.snapshots,
        derivatives=context.derivatives,
        unknowns=context.unknowns,
        spot_prices=(
            SpotPriceContext.model_validate_json(spot_prices)
            if spot_prices
            else SpotPriceContext(asset=request.asset)
        ),
    ).render()


def create_crypto_analyst(llm: BaseChatModel) -> Callable[[AgentState], CryptoReport]:
    """Pre-fetch public context; model receives no exchange or external tools."""

    def crypto_node(state: AgentState) -> CryptoReport:
        if state.get("asset_type", "stock") != "crypto":
            raise CryptoAssetRequiredError()
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        context = collect_crypto_report(ticker, trade_date, state.get("crypto_market_data", ""))
        context_data = ""
        derivatives_data = ""
        if date.fromisoformat(trade_date) != datetime.now(UTC).date() or not context.startswith(
            "{"
        ):
            report = context
        else:
            evidence = CryptoAnalystContext.model_validate_json(context)
            if evidence.derivatives:
                derivatives_data = evidence.derivatives.model_dump_json()
            context_data = CryptoContext(
                asset=evidence.asset, snapshots=evidence.snapshots, unknowns=evidence.unknowns
            ).render()
            context = json.dumps(
                {
                    **evidence.summary().model_dump(mode="json"),
                    "spot_prices": evidence.spot_prices.summary().model_dump(mode="json"),
                },
                sort_keys=True,
                indent=2,
            )
            response = llm.invoke(
                [
                    SystemMessage(
                        "You are a read-only crypto analyst. Analyze only supplied public "
                        "snapshots: positioning, derivatives, network activity, venue coverage, "
                        "timestamps, freshness and unknowns. Missing data is unavailable, not zero. "
                        "Snapshots describe current context. Derivatives features summarize observed "
                        "Binance USD-M history ending at retrieval, not historical report-date reconstruction. "
                        "Funding mean uses arithmetic event rates; trend subtracts previous 7 UTC days "
                        "from last 7 UTC days. OI deltas use hourly contract observations and nearest "
                        "baseline within 30 minutes. Preserve horizons, counts, coverage and failures. Do not "
                        "mix spot_prices.latest_finalized (eligible completed UTC day) with "
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
                response.text
                + "\n\nPublic crypto evidence (provenance and finality):\n```json\n"
                + context
                + "\n```"
            )
        return {
            "messages": [AIMessage(content=report)],
            "crypto_report": report,
            "crypto_context_data": context_data,
            "crypto_derivatives_data": derivatives_data,
        }

    return crypto_node


class CryptoAssetRequiredError(ValueError):
    """Crypto analyst was selected for an incompatible asset."""

    def __init__(self) -> None:
        super().__init__("crypto analyst is only available for asset_type crypto")
