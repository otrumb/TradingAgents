"""Decision journal storage with legacy Markdown context formatting."""

import contextlib
from pathlib import Path

from tradingagents.agents.rating import parse_rating
from tradingagents.dataflows.files import locked
from tradingagents.dataflows.settlement_models import SettlementOutcome
from tradingagents.memory import journal_records as codec
from tradingagents.memory.journal_outcomes import OutcomeUpdate, outcome_document


class TradingMemoryLog:
    """Persist trading decisions and reflections in a versioned journal."""

    _SEPARATOR = codec.SEPARATOR

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._log_path = None
        path = cfg.get("memory_log_path")
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional cap on resolved entries. None disables rotation.
        self._max_entries = cfg.get("memory_log_max_entries")

    # --- Write: a run records its decision ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
        rating: str | None = None,
    ) -> None:
        """Append pending entry at end of propagate(). No LLM call.

        ``rating`` is the decision's own rating when the caller has it; without
        one it is read from the decision text.
        """
        if not self._log_path:
            return
        with locked(self._log_path):
            if any(e["date"] == trade_date and e["ticker"] == ticker for e in self.load_entries()):
                return
            rating = rating or parse_rating(final_trade_decision)
            entry = codec.encode(
                codec.pending_document(trade_date, ticker, rating, final_trade_decision)
            )
            with self._log_path.open("ab") as file:
                file.write(entry)

    def settling(self, wait: bool = True):
        """Hold the log for one settlement pass at a time, across threads and processes.

        A pass reads the pending entries, asks the model for a reflection on
        each and writes it; two passes at once would pay for the same
        reflections. A separate lock from the one each write takes. Yields
        whether it is held; with ``wait=False`` another pass is not waited for.
        """
        if not self._log_path:
            return contextlib.nullcontext(True)
        return locked(f"{self._log_path}.settle", wait=wait)

    # --- Read ---

    def load_entries(self) -> list[codec.Entry]:
        """Parse all entries from log. Returns list of dicts."""
        if not self._log_path or not self._log_path.exists():
            return []
        return [
            record.entry
            for record in codec.decode(self._log_path.read_bytes())
            if record.entry is not None
        ]

    def get_pending_entries(self) -> list[codec.Entry]:
        """Return entries with outcome:pending, for settlement."""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(
        self, ticker: str, n_same: int = 5, n_cross: int = 3, as_of: str | None = None
    ) -> str:
        """Return formatted past context string for agent prompt injection.

        When ``as_of`` (yyyy-mm-dd) is given, only lessons whose outcome was
        already known by that date are included — an entry is kept only if it
        stores a resolution date (``resolved:...``) that is on or before
        ``as_of``. This keeps a historical/backtest run from learning from
        outcomes that had not happened yet (#1251). ``as_of=None`` disables the
        filter, so live runs and pre-migration entries are unaffected.
        """
        entries = [e for e in self.load_entries() if not e.get("pending")]
        if as_of is not None:
            entries = [e for e in entries if e.get("resolved") and e["resolved"] <= as_of]
        if not entries:
            return ""

        same, cross = [], []
        # Most recent by decision date, then by the later write: a backfill writes
        # older decisions last.
        newest = sorted(range(len(entries)), key=lambda i: (entries[i]["date"], i), reverse=True)
        for e in (entries[i] for i in newest):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    # --- Settle: record a decision's outcome and reflection ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
        resolution_date: str | None = None,
        detailed_outcome: SettlementOutcome | None = None,
    ) -> bool:
        """Resolve the first matching pending record using atomic replacement.

        Returns whether an entry was updated: False when the decision was no
        longer pending (another pass settled it first).

        Stores exact returns and the date the outcome became known. Uses a
        temp-file + os.replace() so a crash mid-write never corrupts the log.
        """
        return self._update_outcomes(
            [
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "raw_return": raw_return,
                    "alpha_return": alpha_return,
                    "holding_days": holding_days,
                    "reflection": reflection,
                    "resolution_date": resolution_date,
                    "detailed_outcome": detailed_outcome,
                }
            ]
        )

    def batch_update_with_outcomes(self, updates: list[OutcomeUpdate]) -> None:
        """Apply multiple outcome updates in a single read + atomic write.

        Each element of updates must have keys: ticker, trade_date,
        raw_return, alpha_return, holding_days, reflection.
        """
        self._update_outcomes(updates)

    def _update_outcomes(self, updates: list[OutcomeUpdate]) -> bool:
        if not self._log_path or not self._log_path.exists() or not updates:
            return False
        with locked(self._log_path):
            records = codec.decode(self._log_path.read_bytes())
            update_map = {(u["trade_date"], u["ticker"]): u for u in updates}
            updated = False
            for index, record in enumerate(records):
                entry = record.entry
                if entry is None or not entry["pending"]:
                    continue
                update = update_map.pop((entry["date"], entry["ticker"]), None)
                if update is None:
                    continue
                outcome = outcome_document(update, entry["rating"])
                if update.get("detailed_outcome") is not None:
                    provenance = record.document.get("provenance") if record.document else None
                    document = record.document or codec.pending_document(
                        entry["date"], entry["ticker"], entry["rating"], entry["decision"]
                    )
                    record = codec.Record(
                        record.raw,
                        entry,
                        {
                            **document,
                            "provenance": {
                                **(provenance if isinstance(provenance, dict) else {}),
                                "asset": outcome["asset"],
                                "benchmark": outcome["benchmark"],
                                "window": outcome["window"],
                            },
                        },
                    )
                records[index] = codec.decode(codec.resolve(record, outcome, update["reflection"]))[
                    0
                ]
                updated = True
            if not updated:
                return False
            content = b"".join(record.raw for record in codec.rotate(records, self._max_entries))
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_bytes(content)
            tmp_path.replace(self._log_path)
            return True

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
