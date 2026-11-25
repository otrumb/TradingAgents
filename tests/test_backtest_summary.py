"""Exact statistics at the journal-to-summary boundary."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from tradingagents.backtest import summarize
from tradingagents.backtest_summary import summarize_entries
from tradingagents.memory import journal_records as codec


def _document(
    alpha: float,
    rating: str = "Buy",
    provenance: tuple[str, str, str, str, int] = ("BTCUSDT", "cash", "USDT", "utc_days", 5),
) -> codec.JSONObject:
    asset, provider, benchmark, kind, count = provenance
    document = codec.pending_document("2025-12-31", asset, rating, "decision")
    document["reflection"] = "settled"
    resolved = (date(2026, 1, 1) + timedelta(days=count)).isoformat()
    document["outcome"] = {
        "raw_return": alpha,
        "alpha_return": alpha,
        "holding_days": count,
        "resolution_date": resolved,
        "asset": {
            "provider": "binance_spot",
            "symbol": asset,
            "entry_date": "2026-01-01",
            "exit_date": resolved,
            "entry_price": 1.0,
            "exit_price": 1.0 + alpha,
        },
        "benchmark": {
            "provider": provider,
            "symbol": benchmark,
            "entry_date": "2026-01-01",
            "exit_date": resolved,
            "entry_price": 1.0,
            "exit_price": 1.0,
        },
        "benchmark_label": benchmark,
        "benchmark_return": 0.0,
        "window": {"kind": kind, "count": count},
    }
    return document


def _journal(tmp_path: Path, documents: list[codec.JSONObject]) -> Path:
    path = tmp_path / "journal.md"
    path.write_bytes(b"".join(codec.encode(document) for document in documents))
    return path


def test_same_rating_separates_asset_benchmark_and_window_and_combines_same_key(tmp_path: Path):
    documents = [
        _document(0.02, provenance=("NVDA", "yahoo", "SPY", "trading_sessions", 5)),
        _document(0.04, provenance=("ETHUSDT", "binance_spot", "BTCUSDT", "utc_days", 5)),
        _document(0.06),
        _document(-0.02),
        _document(0.08, provenance=("BTCUSDT", "binance_spot", "BTCUSDT", "utc_days", 5)),
        _document(0.10, provenance=("BTCUSDT", "cash", "USDT", "utc_days", 21)),
        _document(0.12, provenance=("BTCUSDT", "cash", "USDT", "trading_sessions", 5)),
    ]
    summary = summarize(_journal(tmp_path, documents))

    actual = {
        (key.asset, key.benchmark, key.window, key.rating): (
            score.count,
            score.hit_rate,
            score.mean_alpha,
        )
        for key, score in summary.by_group.items()
    }
    expected = {
        ("BTCUSDT", "cash:USDT", "utc_days:5", "Buy"): (2, 0.5, 0.02),
        ("BTCUSDT", "cash:USDT", "utc_days:21", "Buy"): (1, 1.0, 0.10),
        ("BTCUSDT", "cash:USDT", "trading_sessions:5", "Buy"): (1, 1.0, 0.12),
        ("BTCUSDT", "binance_spot:BTCUSDT", "utc_days:5", "Buy"): (1, 1.0, 0.08),
        ("ETHUSDT", "binance_spot:BTCUSDT", "utc_days:5", "Buy"): (1, 1.0, 0.04),
        ("NVDA", "yahoo:SPY", "trading_sessions:5", "Buy"): (1, 1.0, 0.02),
    }
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key] == pytest.approx(values)
    assert summary.by_rating["Buy"].count == 7
    assert summary.by_rating["Buy"].mean_alpha == pytest.approx(0.4 / 7)
    assert summary.by_rating["Buy"].hit_rate == pytest.approx(6 / 7)


@pytest.mark.parametrize("alpha", [5e-324, 1.2345678901234567e-19, -1e-20])
def test_exact_alpha_survives_rounded_display(tmp_path: Path, alpha: float):
    summary = summarize(_journal(tmp_path, [_document(alpha)]))

    assert summary.by_rating["Buy"].mean_alpha.hex() == alpha.hex()
    assert summary.by_rating["Buy"].hit_rate == float(alpha > 0)
    score = next(iter(summary.by_group.values()))
    assert score.mean_alpha.hex() == alpha.hex()


def test_legacy_unknown_provenance_never_infers_ticker_or_calendar(tmp_path: Path):
    path = _journal(tmp_path, [_document(0.04)])
    with path.open("ab") as file:
        file.write(
            (
                "[2026-01-01 | BTC | Buy | +1.0% | +0.2% | 5d]\n"
                "DECISION:\nold\nREFLECTION:\nold" + codec.SEPARATOR
            ).encode()
        )

    summary = summarize(path)

    unknown = next(key for key in summary.by_group if key.asset == "unknown")
    assert (unknown.benchmark, unknown.window) == ("unknown", "unknown:5d")
    assert summary.by_group[unknown].mean_alpha == 0.002
    assert summary.by_rating["Buy"].count == 2


@pytest.mark.parametrize("rating", ["Hold", "mystery"])
def test_no_direction_retains_factual_alpha_with_null_hit(tmp_path: Path, rating: str):
    document = _document(-0.004, rating)
    outcome = document["outcome"]
    assert isinstance(outcome, dict)
    outcome["scoring"] = {
        "method": "rating-direction",
        "version": 1,
        "basis": "alpha",
        "value": 1.0,
        "hit": True,
    }

    entry = codec.decode(codec.encode(_document(-0.004, rating)))[0].entry
    assert entry is not None
    entry["outcome"] = outcome
    summary = summarize_entries([entry])

    score = next(iter(summary.by_group.values()))
    assert (score.count, score.mean_alpha, score.hit_rate) == (1, -0.004, None)
    assert summary.by_rating[rating] == score


@pytest.mark.parametrize(
    "rating,alpha,expected",
    [
        ("Buy", 0.04, 1.0),
        ("Buy", 5e-324, 1.0),
        ("Sell", 0.04, 0.0),
        ("Sell", -0.04, 1.0),
    ],
)
@pytest.mark.parametrize("stored", [(0.5, True), (0.5, None), (None, True)])
def test_hits_use_rating_and_exact_alpha_despite_stored_scoring(rating, alpha, expected, stored):
    document = _document(alpha, rating)
    outcome = document["outcome"]
    assert isinstance(outcome, dict)
    outcome["scoring"] = {
        "method": "rating-direction",
        "version": 1,
        "basis": "alpha",
        "direction": -1.0,
        "value": -stored[0] if rating == "Buy" and stored[0] is not None else stored[0],
        "hit": False if rating == "Buy" else stored[1],
    }

    entry = codec.decode(codec.encode(_document(alpha, rating)))[0].entry
    assert entry is not None
    entry["outcome"] = outcome
    summary = summarize_entries([entry])

    assert summary.by_rating[rating].hit_rate == expected
    assert next(iter(summary.by_group.values())).hit_rate == expected


def test_review_is_unscored_and_pending_never_enters_groups(tmp_path: Path):
    documents = [
        _document(0.03, "REVIEW"),
        codec.pending_document("2026-01-02", "BTC", "REVIEW", "pending review"),
        codec.pending_document("2026-01-03", "BTC", "Buy", "pending"),
        _document(0.01, "Hold"),
    ]

    summary = summarize(_journal(tmp_path, documents))

    assert (summary.resolved, summary.pending, summary.unscored) == (1, 2, 1)
    assert list(summary.by_rating) == ["Hold"]
    assert [key.rating for key in summary.by_group] == ["Hold"]


@pytest.mark.parametrize(
    "pending,rating,alpha,expected",
    [
        (True, "REVIEW", None, (1, 0, 0)),
        (True, "Buy", "+9.0%", (1, 0, 0)),
        (False, "REVIEW", "+9.0%", (0, 0, 1)),
        (False, "REVIEW", None, (0, 0, 1)),
        (False, "Buy", None, (0, 0, 1)),
        (False, "Buy", "invalid", (0, 0, 1)),
        (False, "Buy", "nan%", (0, 0, 1)),
        (False, "Buy", "inf%", (0, 0, 1)),
        (False, "Buy", "-inf%", (0, 0, 1)),
        (False, "Buy", "+9.0%", (0, 1, 0)),
        (False, "Hold", "+9.0%", (0, 1, 0)),
    ],
)
def test_summary_buckets_partition_entries(
    pending: bool, rating: str, alpha: str | None, expected: tuple[int, int, int]
) -> None:
    entry = codec.decode(b"[2026-01-01 | BTC | Buy | +1.0% | +0.2% | 5d]\nDECISION:\nold")[0].entry
    assert entry is not None
    entry.update(pending=pending, rating=rating, alpha=alpha, alpha_return=None)
    entries = [entry]

    summary = summarize_entries(entries)

    assert (summary.pending, summary.resolved, summary.unscored) == expected
    assert len(entries) == summary.pending + summary.resolved + summary.unscored
    assert sum(score.count for score in summary.by_rating.values()) == expected[1]
    assert sum(score.count for score in summary.by_group.values()) == expected[1]
    rendered = summary.render()
    assert f"pending: {expected[0]}" in rendered
    assert ("Pending cells are not scored above; re-run to settle them." in rendered) == pending


def test_mixed_entries_aggregate_only_scoreable_resolved_rows() -> None:
    documents = [
        _document(0.02),
        _document(-0.04),
        _document(0.06, "Hold"),
        _document(0.90, "REVIEW"),
        codec.pending_document("2026-01-02", "BTC", "REVIEW", "pending review"),
        codec.pending_document("2026-01-03", "BTC", "Buy", "pending"),
    ]
    entries = [
        record.entry
        for document in documents
        for record in codec.decode(codec.encode(document))
        if record.entry is not None
    ]
    entries.append({**entries[0], "alpha_return": None, "alpha": "invalid"})

    summary = summarize_entries(entries)

    assert (summary.pending, summary.resolved, summary.unscored) == (2, 3, 2)
    assert len(entries) == summary.pending + summary.resolved + summary.unscored
    assert set(summary.by_rating) == {"Buy", "Hold"}
    buy = summary.by_rating["Buy"]
    assert (buy.count, buy.hit_rate, buy.mean_alpha) == pytest.approx((2, 0.5, -0.01))
    hold = summary.by_rating["Hold"]
    assert (hold.count, hold.hit_rate, hold.mean_alpha) == (1, None, 0.06)
    assert {key.rating: score for key, score in summary.by_group.items()} == summary.by_rating


def test_group_order_is_independent_of_journal_order(tmp_path: Path):
    documents = [
        _document(0.02, "Sell"),
        _document(0.03, "Buy"),
        _document(0.01, provenance=("AAPL", "yahoo", "SPY", "trading_sessions", 5)),
    ]
    path = _journal(tmp_path, documents)
    first = summarize(path)
    path.write_bytes(b"".join(codec.encode(document) for document in reversed(documents)))

    second = summarize(path)

    assert list(first.by_group) == list(second.by_group) == sorted(first.by_group)
    assert first.render() == second.render()


@pytest.mark.parametrize("exact", [None, float("nan"), float("inf"), -float("inf")])
def test_alpha_uses_finite_exact_value_then_legacy_percent(exact: float | None):
    from tradingagents.backtest import _alpha

    entry = codec.decode(b"[2026-01-01 | BTC | Buy | +1.0% | +0.2% | 5d]\nDECISION:\nold")[0].entry
    assert entry is not None
    entry["alpha_return"] = exact

    assert _alpha(entry) == 0.002


def test_explicit_provenance_and_benchmark_label_supply_group_identity(tmp_path: Path):
    document = _document(0.01)
    outcome = document["outcome"]
    assert isinstance(outcome, dict)
    entry = codec.decode(codec.encode(document))[0].entry
    assert entry is not None
    entry["provenance"] = {
        "asset": outcome.pop("asset"),
        "benchmark": {"provider": "cash"},
        "benchmark_label": "USDT cash",
        "window": outcome.pop("window"),
    }

    entry["outcome"] = outcome
    summary = summarize_entries([entry])

    key = next(iter(summary.by_group))
    assert (key.asset, key.benchmark, key.window) == ("BTCUSDT", "cash:USDT cash", "utc_days:5")
