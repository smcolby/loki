"""Shared pytest fixtures for the loki test suite."""

import pytest

from loki.config import KiwixFile, LokiConfig, PortsConfig


@pytest.fixture
def sample_config() -> LokiConfig:
    """Return a synthetic LokiConfig instance for use across tests."""
    return LokiConfig(
        url="loki.local",
        ports=PortsConfig(caddy=80, kiwix=8080, ollama=11434),
        kiwix_files=[
            KiwixFile(
                name="wikipedia_en_test",
                url="https://download.kiwix.org/zim/wikipedia/wikipedia_en_test_2024-01.zim",
            )
        ],
        ollama_models=["llama3:8b"],
    )


@pytest.fixture(autouse=True)
def _stub_shutil_which(mocker):
    """Prevent _require_tool from failing when external tools are not on PATH.

    All CLI command tests mock subprocess.run, so external tools never execute.
    This stub ensures shutil.which returns a truthy value so the pre-flight
    check passes in every test by default. Tests that specifically exercise
    _require_tool can override this by re-patching shutil.which to None.
    """
    mocker.patch("loki.cli.shutil.which", return_value="/usr/bin/stub")
