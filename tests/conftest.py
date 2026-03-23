"""Shared pytest fixtures for the loki test suite."""

from pathlib import Path

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


@pytest.fixture(autouse=True)
def _stub_system(mocker):
    """Stub loki.system functions so setup tests run without real system changes.

    Functions are patched at their usage site (loki.cli.*) since cli.py imports
    them with ``from loki.system import ...``. Individual tests can override
    these stubs to exercise specific prompt branches.
    """
    mocker.patch("loki.cli.is_installed", return_value=True)
    mocker.patch("loki.cli.detect_package_manager", return_value="apt-get")
    mocker.patch("loki.cli.is_ollama_binding_configured", return_value=True)
    mocker.patch("loki.cli.detect_shell_profile", return_value=Path.home() / ".bashrc")
    mocker.patch("loki.cli.loki_root_already_exported", return_value=True)
    mocker.patch("loki.cli.get_local_ip", return_value="192.168.1.100")
    mocker.patch("loki.cli.start_avahi_publish")
    mocker.patch("loki.cli.stop_avahi_publish")
