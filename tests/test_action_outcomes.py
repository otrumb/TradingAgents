"""Action scoring preserves factual returns across reflection and backtest."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.settlement_fixtures import outcome
from tradingagents.agents.rating import action_adjusted_return, parse_rating
from tradingagents.backtest import summarize
from tradingagents.memory import TradingMemoryLog, settlement
from tradingagents.memory.reflection import Reflector


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating,expected",
    [
        ("Buy", 0.04),
        ("Overweight", 0.04),
        ("Underweight", -0.04),
        ("Sell", -0.04),
        ("Strong Buy", 0.04),
        ("Strong Sell", -0.04),
        (" strong SELL ", -0.04),
        ("Hold", None),
        ("REVIEW", None),
        ("unknown", None),
    ],
)
def test_action_adjusted_return_scores_only_directional_calls(rating, expected):
    assert action_adjusted_return(rating, 0.04) == expected


@pytest.mark.unit
@pytest.mark.parametrize("rating", ["Strong Buy", "Strong Sell"])
def test_outcome_aliases_do_not_expand_generated_rating_parser(rating):
    assert parse_rating(f"Rating: {rating}") == "REVIEW"


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        ("Sell", -0.08, -0.05, (0.08, 0.05)),
        ("Underweight", 0.08, 0.05, (-0.08, -0.05)),
        ("Buy", 0.08, 0.05, (0.08, 0.05)),
        ("Strong Sell", -0.08, -0.05, (0.08, 0.05)),
        ("Strong Buy", -0.08, -0.05, (-0.08, -0.05)),
        ("Hold", 0.08, 0.05, None),
        ("REVIEW", 0.08, 0.05, None),
        ("unknown", 0.08, 0.05, None),
    ],
)
def test_settlement_reflection_scores_stored_rating_and_keeps_factual_memory(
    tmp_path,
    monkeypatch,
    case,
):
    rating, raw, alpha, adjusted = case
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    decision = "Rating: Buy\nConflicting legacy prose must not override stored rating."
    log.store_decision("NVDA", "2026-01-05", decision, rating=rating)
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="lesson")
    monkeypatch.setattr(
        settlement,
        "fetch_outcome",
        lambda *a, **kw: outcome(raw, alpha, 5, "2026-02-01"),
    )

    result = settlement.settle_pending("NVDA", log, Reflector(llm), {})

    assert result.settled == [("NVDA", "2026-01-05")]
    entry = log.load_entries()[0]
    assert (entry["rating"], entry["raw"], entry["alpha"], entry["reflection"]) == (
        rating,
        f"{raw:+.1%}",
        f"{alpha:+.1%}",
        "lesson",
    )
    messages = dict(llm.invoke.call_args.args[0])
    context = messages["human"].split("Final Decision:\n", 1)[0]
    facts = dict(line.split(": ", 1) for line in context.splitlines() if ": " in line)
    assert facts["Rating"] == rating
    assert facts["Raw return over 5 trading days"] == f"{raw:+.1%}"
    assert facts["Alpha vs SPY"] == f"{alpha:+.1%}"
    if adjusted is None:
        assert facts["Directional outcome"].startswith("unscored")
        assert "Action-adjusted alpha vs SPY" not in facts
    else:
        assert facts["Action-adjusted raw return"] == f"{adjusted[0]:+.1%}"
        assert facts["Action-adjusted alpha vs SPY"] == f"{adjusted[1]:+.1%}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating,hit_rate",
    [
        ("Buy", 0.0),
        ("Overweight", 0.0),
        ("Sell", 1.0),
        ("Underweight", 1.0),
        ("Strong Buy", 0.0),
        ("Strong Sell", 1.0),
        ("Hold", None),
        ("unknown", None),
    ],
)
def test_backtest_uses_action_score_and_keeps_factual_mean_alpha(tmp_path, rating, hit_rate):
    path = tmp_path / "m.md"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    log.store_decision("NVDA", "2026-01-05", "decision", rating=rating)
    log.update_with_outcome("NVDA", "2026-01-05", -0.08, -0.05, 5, "lesson", "2026-02-01")

    score = summarize(path).by_rating[rating]

    assert score.hit_rate == hit_rate
    assert score.mean_alpha == -0.05
