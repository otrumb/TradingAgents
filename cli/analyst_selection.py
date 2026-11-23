from typing import Final

from cli.models import AnalystType, AssetType

DEFAULT_ANALYSTS: Final = (
    AnalystType.MARKET,
    AnalystType.SOCIAL,
    AnalystType.NEWS,
    AnalystType.FUNDAMENTALS,
)

ANALYST_CHOICES = [
    ("Market Analyst", AnalystType.MARKET),
    ("Sentiment Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
    ("Crypto Analyst", AnalystType.CRYPTO),
]


def filter_analysts_for_asset_type(
    analysts: list[AnalystType], asset_type: AssetType
) -> list[AnalystType]:
    excluded = {
        AssetType.STOCK: AnalystType.CRYPTO,
        AssetType.CRYPTO: AnalystType.FUNDAMENTALS,
    }[asset_type]
    return [analyst for analyst in analysts if analyst != excluded]


def parse_analysts(value: str, asset_type: AssetType) -> list[AnalystType]:
    names = [
        {"sentiment": "social"}.get(n, n)
        for n in (n.strip().lower() for n in value.split(","))
        if n
    ]
    if not names:
        raise ValueError("name at least one analyst")
    known = {a.value: a for a in AnalystType}
    unknown = [n for n in names if n not in known]
    if unknown:
        choices = ", ".join("sentiment" if name == "social" else name for name in known)
        raise ValueError(f"unknown analyst {', '.join(unknown)}; choose from {choices}")
    available = filter_analysts_for_asset_type(list(known.values()), asset_type)
    unavailable = [n for n in names if known[n] not in available]
    if unavailable:
        raise ValueError(f"{', '.join(unavailable)} is not available for {asset_type.value}")
    return [a for a in available if a.value in names]
