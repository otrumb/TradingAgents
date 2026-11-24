# /// script
# requires-python = ">=3.11"
# dependencies = ["requests>=2.32.4", "pydantic>=2"]
# ///
# Run from repo: .venv\Scripts\python.exe -m tests.live_crypto_derivatives
"""Live credential-free BTC/ETH/SOL driver; prints summaries or exact API failures."""

from tradingagents.dataflows.crypto_derivatives import collect_derivatives
from tradingagents.dataflows.crypto_models import SnapshotRequest


def main() -> None:
    for asset in ("BTC", "ETH", "SOL"):
        history = collect_derivatives(SnapshotRequest(asset=asset))
        print(f"{history.symbol}: funding={len(history.funding)}, OI={len(history.interest)}")
        print(
            f"funding_failure={history.funding_failure}; interest_failure={history.interest_failure}"
        )
        for feature in history.features:
            print(f"  {feature.name}: {feature.value} [{feature.status}] {feature.reason or ''}")


if __name__ == "__main__":
    main()
