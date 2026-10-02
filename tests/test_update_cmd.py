"""Tests for the update subcommand."""

import subprocess
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from loki.cli import cli


def _completed(returncode: int = 0, stdout: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = returncode
    m.stdout = stdout
    return m


@pytest.fixture(autouse=True)
def _config(mocker, sample_config):
    """Serve the sample config to every update test."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)


# --- system package upgrade ---


def test_update_upgrades_installed_packages(mocker):
    """Update calls upgrade_packages for system packages that are installed."""
    mock_upgrade = mocker.patch("loki.cli.upgrade_packages", return_value=True)

    result = CliRunner().invoke(cli, ["update"])

    mock_upgrade.assert_called_once()
    assert "System packages upgraded." in result.output


def test_update_warns_on_package_upgrade_failure(mocker):
    """Update prints a warning when upgrade_packages returns False."""
    mocker.patch("loki.cli.upgrade_packages", return_value=False)

    result = CliRunner().invoke(cli, ["update"])

    assert "system package upgrade failed" in result.output


def test_update_warns_when_no_package_manager(mocker):
    """Update prints a warning and continues when no package manager is found."""
    mocker.patch("loki.cli.detect_package_manager", return_value=None)

    result = CliRunner().invoke(cli, ["update"])

    assert "no supported package manager found" in result.output


def test_update_skips_upgrade_when_no_packages_installed(mocker):
    """Update skips upgrade_packages when no loki packages are installed."""
    mocker.patch("loki.cli.is_installed", return_value=False)
    mock_upgrade = mocker.patch("loki.cli.upgrade_packages")

    result = CliRunner().invoke(cli, ["update"])

    mock_upgrade.assert_not_called()
    assert "No loki system packages found" in result.output


# --- Docker images ---


def test_update_pulls_only_published_images(mocker):
    """Update pulls registry images and skips the locally built llama image."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=_completed())

    CliRunner().invoke(cli, ["update"])

    first = mock_run.call_args_list[0].args[0]
    assert first[-2:] == ["pull", "--ignore-buildable"]


def test_update_builds_only_missing_images(mocker, missing_services):
    """Update builds the services whose configured image tag is not present locally."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    missing_services.return_value = ["llama"]
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=_completed())

    result = CliRunner().invoke(cli, ["update"])

    commands = [call.args[0] for call in mock_run.call_args_list]
    assert [cmd[-2:] for cmd in commands if "build" in cmd] == [["build", "llama"]]
    assert "Building loki-llama:gfx1100-rocm7.2.4-abc123" in result.output


def test_update_skips_build_when_images_current(mocker):
    """Update builds nothing when every configured image is already present."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=_completed())

    result = CliRunner().invoke(cli, ["update"])

    assert not any("build" in call.args[0] for call in mock_run.call_args_list)
    assert "Local images are current." in result.output


def test_update_writes_generated_files_before_compose(mocker, tmp_path, missing_services):
    """Update refreshes .env and compose.strata.yaml before its first Compose call."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    missing_services.return_value = ["llama"]
    seen = []

    def _record(cmd, *_, **__):
        if cmd[:2] == ["docker", "compose"]:
            seen.append(
                ((tmp_path / ".env").is_file(), (tmp_path / "compose.strata.yaml").is_file())
            )
        return _completed()

    mocker.patch("loki.cli.subprocess.run", side_effect=_record)

    CliRunner().invoke(cli, ["update"])

    assert seen[0] == (True, True)


def test_update_warns_and_returns_early_on_pull_failure(mocker):
    """Update prints a warning and stops when docker compose pull fails."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run", autospec=True, return_value=_completed(returncode=1)
    )

    result = CliRunner().invoke(cli, ["update"])

    assert "docker compose pull failed" in result.output
    assert len(mock_run.call_args_list) == 1


def test_update_stops_on_build_failure(mocker, missing_services):
    """Update warns and leaves the running stack alone when the image build fails."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    missing_services.return_value = ["llama"]
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=[_completed(), _completed(returncode=1)],
    )

    result = CliRunner().invoke(cli, ["update"])

    assert "image build failed" in result.output
    assert len(mock_run.call_args_list) == 2


def test_update_stops_on_invalid_preset(mocker, sample_config, write_preset):
    """Update does not pull, build, or restart when a model preset is invalid."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    write_preset(sample_config.llama.models_dir, "broken", "[broken]\nmodel = gone.gguf\n")
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=_completed())

    result = CliRunner().invoke(cli, ["update"])

    assert "invalid model preset" in result.output
    mock_run.assert_not_called()


def test_update_restarts_stack_when_running(mocker):
    """Update runs docker compose up -d when the stack is already running."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=[
            _completed(),  # pull
            _completed(stdout="abc\n"),  # ps -q → stack running
            _completed(),  # up -d
        ],
    )

    result = CliRunner().invoke(cli, ["update"])

    assert mock_run.call_args_list[-1].args[0][-3:] == ["up", "-d", "--remove-orphans"]
    assert "Restarting Docker Compose stack" in result.output


def test_update_skips_restart_when_stack_not_running(mocker):
    """Update does not run docker compose up when the stack is stopped."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=[
            _completed(),  # pull
            _completed(stdout=""),  # ps -q → stack not running
        ],
    )

    result = CliRunner().invoke(cli, ["update"])

    assert not any("up" in call.args[0] for call in mock_run.call_args_list)
    assert "loki start" in result.output


def test_update_exits_when_docker_not_found(mocker):
    """Update exits with an error message when docker is not on PATH."""
    mocker.patch("loki.cli.upgrade_packages", return_value=True)
    mocker.patch(
        "loki.cli.shutil.which",
        side_effect=lambda cmd: None if cmd == "docker" else "/usr/bin/stub",
    )

    result = CliRunner().invoke(cli, ["update"])

    assert result.exit_code != 0
    assert "docker" in result.output
