from tradingagents.dataflows.settlement_models import SettlementLeg, SettlementOutcome


def outcome(raw=0.05, alpha=0.02, days=5, resolved="2026-01-12"):
    return SettlementOutcome(
        asset=SettlementLeg(
            provider="yahoo",
            symbol="NVDA",
            entry_date="2026-01-06",
            exit_date=resolved,
            entry_price=100.0,
            exit_price=100.0 * (1 + raw),
        ),
        benchmark=SettlementLeg(
            provider="yahoo",
            symbol="SPY",
            entry_date="2026-01-06",
            exit_date=resolved,
            entry_price=100.0,
            exit_price=100.0 * (1 + raw - alpha),
        ),
        benchmark_label="SPY",
        window_kind="trading_sessions",
        window_count=days,
        raw_return=raw,
        benchmark_return=raw - alpha,
        alpha_return=raw - (raw - alpha),
    )
