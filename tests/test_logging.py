from __future__ import annotations

import logging
from unittest.mock import patch

import structlog

from rag_intelligence.logging import setup_logging


def _get_renderer() -> object:
    """Extract the final renderer processor from the root logger."""
    root = logging.getLogger()
    fmt = root.handlers[0].formatter
    assert isinstance(fmt, structlog.stdlib.ProcessorFormatter)
    return fmt.processors[-1]


class TestSetupLogging:
    def test_json_renderer_when_json_logs_true(self):
        setup_logging(json_logs=True)
        assert isinstance(_get_renderer(), structlog.processors.JSONRenderer)

    def test_console_renderer_when_json_logs_false(self):
        setup_logging(json_logs=False)
        assert isinstance(_get_renderer(), structlog.dev.ConsoleRenderer)

    def test_auto_detects_json_when_not_tty(self):
        with patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            setup_logging(json_logs=None)

        assert isinstance(_get_renderer(), structlog.processors.JSONRenderer)

    def test_auto_detects_console_when_tty(self):
        with patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = True
            setup_logging(json_logs=None)

        assert isinstance(_get_renderer(), structlog.dev.ConsoleRenderer)

    def test_sets_log_level(self):
        setup_logging(log_level="DEBUG", json_logs=False)
        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_is_uppercased(self):
        setup_logging(log_level="warning", json_logs=False)
        assert logging.getLogger().level == logging.WARNING

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        setup_logging(json_logs=False)
        assert len(root.handlers) == 1
