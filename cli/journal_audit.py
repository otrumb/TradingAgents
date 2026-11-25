from pathlib import Path
from typing import Annotated

import typer

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.journal_audit import AuditQuery, audit_journal


def journal_audit(
    path: Annotated[
        Path | None,
        typer.Option("--path", help="Journal path; defaults to configured memory_log_path."),
    ] = None,
    min_sample: Annotated[
        str, typer.Option("--min-sample", help="Advisory minimum independent samples.")
    ] = "5",
    json_output: Annotated[bool, typer.Option("--json", help="Emit metadata JSON only.")] = False,
) -> None:
    """Inventory journal and audit ETHUSDT/BTCUSDT utc_days:7 Underweight calls."""
    try:
        result = audit_journal(
            path if path is not None else Path(DEFAULT_CONFIG["memory_log_path"]),
            AuditQuery(min_sample=int(min_sample)),
        )
    except (OSError, ValueError):
        typer.echo("Error: journal unreadable or invalid audit input.", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.render_json() if json_output else result.render())
