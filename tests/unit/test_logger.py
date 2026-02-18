"""Unit tests for the structured logging module."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import pytest

from ncaa_eval.utils.logger import (
    DEBUG,
    NORMAL,
    QUIET,
    VERBOSE,
    configure_logging,
    get_logger,
)

# Public-facing root logger name — part of the observable API documented in logger.py.
_NCAA_LOGGER = "ncaa_eval"


@pytest.fixture(autouse=True)
def _reset_ncaa_eval_logger() -> Iterator[None]:
    """Reset the ncaa_eval root logger between every test in this module."""
    yield
    root = logging.getLogger(_NCAA_LOGGER)
    root.handlers.clear()
    root.setLevel(logging.WARNING)


@pytest.mark.smoke
class TestConfigureLogging:
    """Tests for `configure_logging`."""

    def test_quiet_sets_warning_level(self) -> None:
        configure_logging("QUIET")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == logging.WARNING

    def test_normal_sets_info_level(self) -> None:
        configure_logging("NORMAL")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == logging.INFO

    def test_verbose_sets_custom_level(self) -> None:
        configure_logging("VERBOSE")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == VERBOSE
        assert root.level == 15

    def test_debug_sets_debug_level(self) -> None:
        configure_logging("DEBUG")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == logging.DEBUG

    def test_level_is_case_insensitive(self) -> None:
        configure_logging("verbose")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == VERBOSE

    def test_invalid_level_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown log level"):
            configure_logging("TRACE")

    def test_handler_uses_stderr(self) -> None:
        import sys

        configure_logging("NORMAL")
        root = logging.getLogger(_NCAA_LOGGER)
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr

    def test_reconfigure_does_not_duplicate_handlers(self) -> None:
        configure_logging("NORMAL")
        configure_logging("DEBUG")
        configure_logging("QUIET")
        root = logging.getLogger(_NCAA_LOGGER)
        assert len(root.handlers) == 1

    def test_propagate_is_false(self) -> None:
        configure_logging("NORMAL")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.propagate is False

    def test_log_format_applied(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging("DEBUG")
        log = get_logger("fmtcheck")
        log.info("format-test")
        captured = capsys.readouterr()
        # Verify pipe-delimited format: timestamp | name | level | message
        assert " | ncaa_eval.fmtcheck | " in captured.err
        assert "INFO" in captured.err
        assert "format-test" in captured.err


@pytest.mark.smoke
class TestEnvVarOverride:
    """Tests for `NCAA_EVAL_LOG_LEVEL` environment variable."""

    def test_env_var_sets_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NCAA_EVAL_LOG_LEVEL", "VERBOSE")
        configure_logging()
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == VERBOSE

    def test_explicit_level_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NCAA_EVAL_LOG_LEVEL", "DEBUG")
        configure_logging("QUIET")
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == QUIET

    def test_default_is_normal_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NCAA_EVAL_LOG_LEVEL", raising=False)
        configure_logging()
        root = logging.getLogger(_NCAA_LOGGER)
        assert root.level == NORMAL


@pytest.mark.smoke
class TestGetLogger:
    """Tests for `get_logger`."""

    def test_returns_child_of_ncaa_eval(self) -> None:
        log = get_logger("ingest")
        assert log.name == "ncaa_eval.ingest"

    def test_nested_name(self) -> None:
        log = get_logger("transform.features")
        assert log.name == "ncaa_eval.transform.features"

    def test_returns_logger_instance(self) -> None:
        log = get_logger("test")
        assert isinstance(log, logging.Logger)


@pytest.mark.smoke
class TestLogOutput:
    """Tests that verify log output contains expected components."""

    def test_output_contains_timestamp_module_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging("DEBUG")
        log = get_logger("mymodule")
        log.info("hello test")
        captured = capsys.readouterr()
        # Output goes to stderr
        assert "mymodule" in captured.err
        assert "INFO" in captured.err
        assert "hello test" in captured.err
        # Timestamp present as ISO-like datetime prefix (e.g., "2026-02-18 12:34:56")
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", captured.err)


@pytest.mark.smoke
class TestLevelConstants:
    """Tests that exported level constants have correct values."""

    def test_quiet_value(self) -> None:
        assert QUIET == 30

    def test_normal_value(self) -> None:
        assert NORMAL == 20

    def test_verbose_value(self) -> None:
        assert VERBOSE == 15

    def test_debug_value(self) -> None:
        assert DEBUG == 10
