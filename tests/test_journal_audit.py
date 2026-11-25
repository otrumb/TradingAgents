import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.journal_audit import AuditInputError, AuditQuery, audit_journal
from tradingagents.memory import journal_records as codec


def frame(
    start: int,
    *,
    rating: str = "Underweight",
    alpha: float = -0.02,
    asset: str = "ETHUSDT",
    provider: str = "binance_spot",
) -> bytes:
    first = date(2026, 1, 1) + timedelta(days=start)
    last = first + timedelta(days=7)
    document = codec.pending_document(
        (first - timedelta(days=1)).isoformat(), asset, rating, "SECRET decision"
    )
    document.update(
        outcome={
            "raw_return": alpha,
            "alpha_return": alpha,
            "benchmark_return": 0.0,
            "holding_days": 7,
            "resolution_date": last.isoformat(),
            "asset": {
                "provider": provider,
                "symbol": asset,
                "entry_date": first.isoformat(),
                "exit_date": last.isoformat(),
                "entry_price": 100.0,
                "exit_price": 100 * (1 + alpha),
            },
            "benchmark": {
                "provider": "cash",
                "symbol": "USDT",
                "entry_date": first.isoformat(),
                "exit_date": last.isoformat(),
                "entry_price": 1.0,
                "exit_price": 1.0,
            },
            "benchmark_label": "cash",
            "window": {"kind": "utc_days", "count": 7},
        },
        reflection="SECRET reflection",
    )
    return codec.encode(document)


def test_inventory_counts_all_spans_without_inventing_provenance(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    legacy = b"[2026-01-01 | BTCUSDT | Underweight | +1% | -2% | 7d]" + codec.SEPARATOR.encode()
    pending = codec.encode(codec.pending_document("2026-01-01", "ETHUSDT", "Buy", "SECRET"))
    path.write_bytes(
        b"opaque\xff"
        + codec.SEPARATOR.encode()
        + legacy
        + pending
        + frame(0)
        + frame(8, rating="REVIEW")
    )

    audit = audit_journal(path)

    assert asdict(audit.inventory) == {
        "physical": 5,
        "opaque": 1,
        "recognized": 4,
        "pending": 1,
        "resolved": 3,
        "scoreable": 2,
        "unscored": 1,
        "complete_provenance": 1,
    }
    assert len(audit.cohorts) == 1


def test_closed_boundary_and_cross_rating_selection(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    path.write_bytes(frame(0, rating="Buy") + frame(0) + frame(7) + frame(8))

    audit = audit_journal(path, AuditQuery(min_sample=1))

    buy, underweight = audit.cohorts
    assert [s.physical_index for s in buy.selected] == [0]
    assert [s.physical_index for s in underweight.selected] == [3]
    assert underweight.excluded_overlap == 2
    assert underweight.favorable is True


@pytest.mark.parametrize("alpha", [5e-324, -0.0, 1.2345678901234567e-19, -0.02])
def test_exact_alpha_stays_internal_and_aggregate_hex_survives(
    tmp_path: Path, alpha: float
) -> None:
    path = tmp_path / "journal"
    path.write_bytes(frame(0, alpha=alpha))

    audit = audit_journal(path, AuditQuery(min_sample=1))

    cohort = json.loads(audit.render_json())["cohorts"][0]
    assert audit.cohorts[0].selected[0].alpha_hex == alpha.hex()
    assert audit.cohorts[0].selected[0].alpha.hex() == alpha.hex()
    assert cohort["mean_alpha"] == alpha
    assert cohort["mean_alpha_hex"] == cohort["mean_alpha"].hex()


def test_insufficient_is_null_even_with_positive_evidence(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    path.write_bytes(frame(0))

    audit = audit_journal(path)

    cohort = audit.cohorts[0]
    assert (cohort.status, cohort.count, cohort.favorable, cohort.mean_alpha, cohort.hit_rate) == (
        "insufficient",
        1,
        None,
        None,
        None,
    )
    assert all(row["favorable"] is None for row in json.loads(audit.render_json())["primary"])
    assert cohort.negative_alpha_count == 1
    assert cohort.negative_alpha_fraction is None


def test_threshold_is_advisory_and_sufficient_negative_is_false(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    path.write_bytes(b"".join(frame(n * 8, alpha=0.02) for n in range(5)))

    cohort = audit_journal(path).cohorts[0]

    assert (cohort.status, cohort.count, cohort.favorable, cohort.hit_rate) == (
        "sufficient",
        5,
        False,
        0.0,
    )


def test_cohorts_keep_assets_and_providers_separate(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    path.write_bytes(frame(0) + frame(0, asset="BTCUSDT") + frame(0, provider="yahoo"))

    audit = audit_journal(path)

    assert [(c.key.asset, c.key.asset_provider, c.count) for c in audit.cohorts] == [
        ("BTCUSDT", "binance_spot", 1),
        ("ETHUSDT", "binance_spot", 1),
        ("ETHUSDT", "yahoo", 1),
    ]


def test_read_decode_once_no_mutation_or_sensitive_output(tmp_path: Path) -> None:
    path = tmp_path / "private-journal"
    data = frame(0) + b"opaque SECRET\xff"
    path.write_bytes(data)

    with (
        patch.object(Path, "read_bytes", autospec=True, return_value=data) as read,
        patch.object(codec, "decode", wraps=codec.decode) as decode,
    ):
        audit = audit_journal(path)

    assert read.call_count == decode.call_count == 1
    assert path.read_bytes() == data
    for rendered in (audit.render(), audit.render_json()):
        assert "SECRET" not in rendered and str(path) not in rendered and "\xff" not in rendered
        for forbidden in (
            '"selected"',
            '"alpha"',
            '"alpha_hex"',
            "physical_index",
            "analysis_date",
            "entry_date",
            "exit_date",
            "2026-01-01",
            "2026-01-08",
        ):
            assert forbidden not in rendered


def test_earliest_finish_wins_even_when_physical_order_differs(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    path.write_bytes(frame(8) + frame(1) + frame(0, rating="Hold") + frame(16))

    audit = audit_journal(path, AuditQuery(min_sample=1))

    hold, underweight = audit.cohorts
    assert [s.physical_index for s in hold.selected] == [2]
    assert [s.physical_index for s in underweight.selected] == [0, 3]
    assert hold.hit_rate is None and hold.favorable is None


def test_v2_provenance_extensions_do_not_supply_missing_outcome_facts(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    document = codec.pending_document("2026-01-01", "ETHUSDT", "Underweight", "SECRET")
    document.update(
        outcome={
            "raw_return": 0.01,
            "alpha_return": -0.02,
            "holding_days": 7,
            "resolution_date": "2026-01-09",
        },
        reflection="SECRET",
        provenance={
            "asset": {"symbol": "ETHUSDT"},
            "benchmark": {"provider": "cash", "symbol": "USDT"},
            "window": {"kind": "utc_days", "count": 7},
        },
    )
    path.write_bytes(codec.encode(document))

    audit = audit_journal(path)

    assert audit.inventory.scoreable == 1
    assert audit.inventory.complete_provenance == 0
    assert audit.cohorts == ()


@pytest.mark.parametrize(
    "query",
    [
        {"min_sample": 0},
        {"min_sample": True},
        {"window": "utc_days:0"},
        {"window": "other:7"},
        {"assets": ()},
        {"rating": "Hold"},
    ],
)
def test_invalid_query_rejected(query) -> None:
    with pytest.raises(AuditInputError):
        AuditQuery(**query)


def test_empty_journal_reports_missing_primary_as_insufficient(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")

    audit = audit_journal(path)

    assert audit.inventory.physical == 0
    assert [
        (r["asset"], r["status"], r["favorable"])
        for r in json.loads(audit.render_json())["primary"]
    ] == [("BTCUSDT", "insufficient", None), ("ETHUSDT", "insufficient", None)]
    for row in json.loads(audit.render_json())["primary"]:
        assert row["negative_alpha_count"] == 0
        assert row["negative_alpha_fraction"] is None
        assert row["mean_alpha_hex"] is None


@pytest.mark.parametrize(
    "alphas,negative_count,favorable",
    [
        ((-0.01, -0.01, -0.01, 0.10, 0.10), 3, True),
        ((0.01, 0.01, 0.01, -0.10, -0.10), 2, False),
        ((-0.01, -0.01, 0.0, -0.0, 0.10, 0.10), 2, False),
        ((-0.01, -0.01, -0.01, 0.10, 0.10, 0.10), 3, False),
    ],
)
def test_underweight_answer_uses_negative_majority_not_mean(
    tmp_path: Path,
    alphas: tuple[float, ...],
    negative_count: int,
    favorable: bool,
) -> None:
    path = tmp_path / "journal"
    path.write_bytes(b"".join(frame(n * 8, alpha=alpha) for n, alpha in enumerate(alphas)))

    audit = audit_journal(path)

    cohort = json.loads(audit.render_json())["cohorts"][0]
    assert cohort["negative_alpha_count"] == negative_count
    assert cohort["negative_alpha_fraction"] == negative_count / len(alphas)
    assert cohort["favorable"] is favorable
    assert cohort["mean_alpha"] == pytest.approx(sum(alphas) / len(alphas))
    assert cohort["hit_rate"] == negative_count / len(alphas)
