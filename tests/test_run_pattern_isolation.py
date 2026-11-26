from langchain_core.outputs import ChatResult
from pydantic import Field

from tests.crypto_watch_fixtures import CUTOFF, wire_rows
from tests.test_graph_end_to_end import ScriptedModel, _graph, offline
from tests.test_run_pattern_evidence import END_AT, spot_context, watch_snapshot
from tradingagents.dataflows.crypto_spot_models import utc_milliseconds
from tradingagents.dataflows.crypto_watch_models import (
    DURATIONS,
    WatchCandle,
    WatchSeries,
    WatchSnapshot,
)
from tradingagents.graph import run_decisions, trading_graph
from tradingagents.portfolio import PortfolioContext, Position

__all__ = ["offline"]


class RecordingModel(ScriptedModel):
    prompts: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(tuple((message.type, message.content) for message in messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_patterns_leave_all_prompts_decisions_and_journal_values_identical(
    tmp_path, monkeypatch, offline
):
    monkeypatch.setattr(
        trading_graph, "collect_spot_prices", lambda request: spot_context(indices=range(160))
    )
    watch = WatchSnapshot.model_validate(
        watch_snapshot().model_dump()
        | {
            "series": tuple(
                WatchSeries(
                    interval=interval,
                    candles=tuple(
                        WatchCandle(
                            interval=interval,
                            open_time=utc_milliseconds(row[0]) + (END_AT - CUTOFF),
                            close_time=utc_milliseconds(row[6]) + (END_AT - CUTOFF),
                            open=row[1],
                            high=row[2],
                            low=row[3],
                            close=row[4],
                            volume=row[5],
                        )
                        for row in wire_rows(interval)
                    ),
                )
                for interval in DURATIONS
            )
        }
    )
    monkeypatch.setattr(run_decisions, "collect_watch", lambda request: watch)
    portfolio = PortfolioContext(positions=(Position(ticker="BTC-USD", quantity=1),))
    enabled_model, disabled_model = RecordingModel(), RecordingModel()
    enabled = _graph(tmp_path / "enabled", monkeypatch, enabled_model)
    enabled_state, enabled_signal = enabled.propagate("BTC-USD", "2024-01-02", "crypto", portfolio)

    initialize = trading_graph.initialize_decisions

    def without_patterns(state, book, contexts, *, current):
        state["crypto_pattern_evidence"] = "disabled"
        result = initialize(state, book, contexts, current=current)
        result["crypto_pattern_evidence"] = result["crypto_pattern_report"] = ""
        return result

    monkeypatch.setattr(trading_graph, "initialize_decisions", without_patterns)
    disabled = _graph(tmp_path / "disabled", monkeypatch, disabled_model)
    disabled_state, disabled_signal = disabled.propagate(
        "BTC-USD", "2024-01-02", "crypto", portfolio
    )

    assert enabled_state["crypto_pattern_evidence"]
    assert enabled_signal == disabled_signal
    assert sorted(enabled_model.prompts, key=repr) == sorted(disabled_model.prompts, key=repr)
    for field in (
        "final_rating",
        "final_trade_decision",
        "spot_action_data",
        "spot_allocation_data",
        "crypto_futures_watch_data",
        "crypto_futures_watch_evidence",
        "portfolio_context",
        "instrument_context",
        "crypto_market_data",
        "investment_plan",
        "trader_investment_plan",
    ):
        assert enabled_state[field] == disabled_state[field], field
    enabled_entry = enabled.memory_log.load_entries()[0]
    disabled_entry = disabled.memory_log.load_entries()[0]
    assert '"action":"consider_long"' in enabled_state["crypto_futures_watch_data"]
    for field in ("ticker", "date", "rating", "decision"):
        assert enabled_entry[field] == disabled_entry[field], field
