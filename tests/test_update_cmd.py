"""Tests for the update subcommand."""

import subprocess
from unittest.mock import MagicMock

from click.testing import CliRunner

from loki.cli import cli


def _completed(returncode: int = 0, stdout: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = returncode
    m.stdout = stdout
    return m


# --- system package upgrade ---


def test_update_upgrades_installed_packages(mocker):
    """Update calls upgrade_packages for system packages that are installed."""
    mock_upgrade = mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    mock_upgrade.assert_called_once()
    assert "System packages upgraded." in result.output


def test_update_warns_on_package_upgrade_failure(mocker):
    """Update prints a warning when upgrade_packages returns False."""
    mocker.patch("loki.cli.upgrade_packages", return_value=False)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    assert "system package upgrade failed" in result.output


def test_update_warns_when_no_package_manager(mocker):
    """Update prints a warning and continues when no package manager is found."""
    mocker.patch("loki.cli.detect_package_manager", return_value=None)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    assert "no supported package manager found" in result.output


def test_update_skips_upgrade_when_no_packages_installed(mocker):
    """Update skips upgrade_packages when no loki packages are installed."""
    mocker.patch("loki.cli.is_installed", return_value=False)
    mock_upgrade = mocker.patch("loki.cli.upgrade_packages")
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    mock_upgrade.assert_not_called()
    assert "No loki system packages found" in result.output


# --- Ollama upgrade ---


def test_update_upgrades_ollama_when_installed(mocker):
    """Update re-runs the Ollama install script when Ollama is on PATH."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_install = mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    mock_install.assert_called_once()
    assert "Ollama upgraded." in result.output


def test_update_warns_on_ollama_upgrade_failure(mocker):
    """Update prints a warning when the Ollama install script fails."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=False)
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    assert "Ollama upgrade failed" in result.output


def test_update_skips_ollama_when_not_installed(mocker):
    """Update skips the Ollama upgrade and prints a message when Ollama is absent."""
    mocker.patch("loki.cli.is_installed", side_effect=lambda cmd: cmd != "ollama")
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_install = mocker.patch("loki.cli.install_ollama")
    mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    result = CliRunner().invoke(cli, ["update"])

    mock_install.assert_not_called()
    assert "Ollama is not installed" in result.output


# --- Docker image pull ---


def test_update_pulls_docker_images(mocker):
    """Update runs docker compose pull."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[_completed(), _completed(stdout="")],
    )

    CliRunner().invoke(cli, ["update"])

    pull_calls = [c for c in mock_run.call_args_list if "pull" in c[0][0]]
    assert pull_calls


def test_update_warns_and_returns_early_on_pull_failure(mocker):
    """Update prints a warning and stops when docker compose pull fails."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        return_value=_completed(returncode=1),
    )

    result = CliRunner().invoke(cli, ["update"])

    assert "docker compose pull failed" in result.output
    ps_calls = [c for c in mock_run.call_args_list if "ps" in c[0][0]]
    assert not ps_calls


def test_update_restarts_stack_when_running(mocker):
    """Update runs docker compose up -d when the stack is already running."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[
            _completed(),  # pull
            _completed(stdout="abc\n"),  # ps -q → stack running
            _completed(),  # up -d
        ],
    )

    result = CliRunner().invoke(cli, ["update"])

    up_calls = [c for c in mock_run.call_args_list if "up" in c[0][0]]
    assert up_calls
    assert "Restarting Docker Compose stack" in result.output


def test_update_skips_restart_when_stack_not_running(mocker):
    """Update does not run docker compose up when the stack is stopped."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        side_effect=[
            _completed(),  # pull
            _completed(stdout=""),  # ps -q → stack not running
        ],
    )

    result = CliRunner().invoke(cli, ["update"])

    up_calls = [c for c in mock_run.call_args_list if "up" in c[0][0]]
    assert not up_calls
    assert "loki start" in result.output


def test_update_exits_when_docker_not_found(mocker):
    """Update exits with an error message when docker is not on PATH."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch(
        "loki.cli.shutil.which",
        side_effect=lambda cmd: None if cmd == "docker" else "/usr/bin/stub",
    )

    result = CliRunner().invoke(cli, ["update"])

    assert result.exit_code != 0
    assert "docker" in result.output
