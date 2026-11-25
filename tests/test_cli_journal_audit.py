import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.main import app
from tests.test_journal_audit import frame
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.memory import TradingMemoryLog


@pytest.mark.parametrize("json_output", [False, True])
def test_command_is_read_only_with_explicit_path(tmp_path: Path, json_output: bool) -> None:
    path = tmp_path / "journal"
    data = frame(0)
    path.write_bytes(data)

    with patch.object(
        TradingMemoryLog, "__init__", side_effect=AssertionError("constructor forbidden")
    ):
        result = CliRunner().invoke(
            app, ["journal-audit", "--path", str(path), *(["--json"] if json_output else [])]
        )

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == data
    assert "SECRET" not in result.output and str(path) not in result.output
    if json_output:
        document = json.loads(result.output)
        assert (document["schema"], document["version"]) == ("tradingagents.journal-audit", 1)
        assert document["cohorts"][0]["favorable"] is None
    else:
        assert "insufficient" in result.output and "\x1b" not in result.output


def test_default_path_and_empty_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    monkeypatch.setitem(DEFAULT_CONFIG, "memory_log_path", str(path))

    result = CliRunner().invoke(app, ["journal-audit", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["inventory"]["physical"] == 0


@pytest.mark.parametrize("case", ["missing", "directory", "threshold", "bad-threshold"])
def test_invalid_input_exits_one_without_traceback(tmp_path: Path, case: str) -> None:
    path = tmp_path / "missing" if case == "missing" else tmp_path
    args = ["journal-audit", "--path", str(path)]
    if case in {"threshold", "bad-threshold"}:
        args += ["--min-sample", "0" if case == "threshold" else "abc"]

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
