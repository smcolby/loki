"""Tests for the start subcommand — Ollama health check, model pull, compose up, and mDNS."""

import subprocess

import requests
from click.testing import CliRunner

from loki.cli import cli
from loki.config import LokiConfig, PortsConfig


def test_start_pulls_models_and_runs_compose(mocker, sample_config):
    """The start command pulls each configured model then starts the compose stack."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    calls = mock_run.call_args_list
    assert any(call.args[0] == ["ollama", "pull", "llama3:8b"] for call in calls)
    assert any("compose" in call.args[0] and "up" in call.args[0] for call in calls)


def test_start_compose_called_after_model_pull(mocker, sample_config):
    """The start command calls docker compose up -d after all ollama pull commands."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    commands = [call.args[0] for call in mock_run.call_args_list]
    compose_index = next(i for i, cmd in enumerate(commands) if "compose" in cmd)
    pull_indices = [i for i, cmd in enumerate(commands) if "ollama" in cmd and "pull" in cmd]
    assert all(i < compose_index for i in pull_indices), (
        "All ollama pull calls should precede docker compose up -d."
    )


def test_start_uses_configured_ollama_port(mocker, sample_config):
    """The start command pings the Ollama port specified in config."""
    config = sample_config.model_copy(update={"ports": PortsConfig(ollama=12000)})
    mocker.patch("loki.cli.load_config", return_value=config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_get = mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    pinged_url = mock_get.call_args_list[0].args[0]
    assert ":12000" in pinged_url


def test_start_warns_when_ollama_offline(mocker, sample_config):
    """The start command prints a warning when Ollama is not reachable."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch(
        "loki.cli.requests.get",
        autospec=True,
        side_effect=requests.exceptions.ConnectionError,
    )
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["start"])

    assert "Warning" in result.output
    assert "OLLAMA_HOST" in result.output


def test_start_still_runs_compose_when_ollama_offline(mocker, sample_config):
    """The start command still runs docker compose up -d even when Ollama is unreachable."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch(
        "loki.cli.requests.get",
        autospec=True,
        side_effect=requests.exceptions.ConnectionError,
    )
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    commands = [call.args[0] for call in mock_run.call_args_list]
    assert any("compose" in cmd and "up" in cmd and "-d" in cmd for cmd in commands)


def test_start_no_models_skips_pull(mocker):
    """The start command does not call ollama pull when ollama_models is empty."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    commands = [call.args[0] for call in mock_run.call_args_list]
    assert not any("ollama" in cmd and "pull" in cmd for cmd in commands)


def test_start_warns_on_failed_model_pull(mocker, sample_config):
    """The start command prints a warning when ollama pull exits with a non-zero code."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    failed = mocker.MagicMock(spec=subprocess.CompletedProcess)
    failed.returncode = 1
    success = mocker.MagicMock(spec=subprocess.CompletedProcess)
    success.returncode = 0
    # First call is ollama pull (fails), second is docker compose up (succeeds).
    mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=[failed, success],
    )

    result = CliRunner().invoke(cli, ["start"])

    assert "Warning" in result.output
    assert "llama3:8b" in result.output


# ---------------------------------------------------------------------------
# mDNS / avahi-publish-address
# ---------------------------------------------------------------------------

def test_start_broadcasts_mdns_for_local_url(mocker, sample_config):
    """The start command spawns avahi-publish-address when the URL ends in .local."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.subprocess.run", autospec=True)
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_called_once()


def test_start_avahi_called_with_correct_hostname(mocker, sample_config):
    """The start command calls start_avahi_publish with the configured URL."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.subprocess.run", autospec=True)
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    CliRunner().invoke(cli, ["start"])

    call_args = mock_avahi.call_args
    assert call_args[0][0] == "loki.local"


def test_start_skips_mdns_for_non_local_url(mocker):
    """The start command skips avahi-publish-address when the URL does not end in .local."""
    config = LokiConfig(url="loki.home")
    mocker.patch("loki.cli.load_config", return_value=config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.subprocess.run", autospec=True)
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    result = CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_not_called()
    assert "skipping mdns broadcast" in result.output.lower()


def test_start_skips_mdns_and_warns_when_ip_unavailable(mocker, sample_config):
    """The start command warns and skips mDNS when no local IP can be determined."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.subprocess.run", autospec=True)
    mocker.patch("loki.cli.get_local_ip", return_value="")
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    result = CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_not_called()
    assert "Warning" in result.output


def test_start_warns_when_avahi_publish_not_installed(mocker, sample_config):
    """The start command warns and skips mDNS when avahi-publish-address is absent."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.subprocess.run", autospec=True)
    # avahi-publish-address missing; all other tools present
    mocker.patch("loki.cli.is_installed",
                 side_effect=lambda cmd: cmd != "avahi-publish-address")
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    result = CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_not_called()
    assert "Warning" in result.output
    assert "avahi-publish-address" in result.output
