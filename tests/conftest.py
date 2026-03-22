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
