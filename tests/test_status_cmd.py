"""Tests for the status subcommand — service health checks."""

import requests
from click.testing import CliRunner

from loki.cli import cli
from loki.config import LokiConfig, PortsConfig


def test_status_prints_online_when_services_respond(mocker, sample_config):
    """The status command prints ONLINE for both services when both return HTTP 200."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)

    result = CliRunner().invoke(cli, ["status"])

    assert "Ollama: ONLINE" in result.output
    assert "Kiwix: ONLINE" in result.output


def test_status_prints_offline_on_connection_error(mocker, sample_config):
    """The status command prints OFFLINE for a service that raises a connection error."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch(
        "loki.cli.requests.get",
        autospec=True,
        side_effect=requests.exceptions.ConnectionError("refused"),
    )

    result = CliRunner().invoke(cli, ["status"])

    assert "Ollama: OFFLINE" in result.output
    assert "Kiwix: OFFLINE" in result.output


def test_status_prints_offline_on_non_200(mocker, sample_config):
    """The status command prints OFFLINE when a service returns a non-200 HTTP status."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 503
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)

    result = CliRunner().invoke(cli, ["status"])

    assert "OFFLINE" in result.output
    assert "503" in result.output


def test_status_checks_both_endpoints(mocker, sample_config):
    """The status command makes exactly two HTTP GET requests — one per service."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_get = mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)

    CliRunner().invoke(cli, ["status"])

    assert mock_get.call_count == 2


def test_status_uses_configured_ports(mocker):
    """The status command uses the kiwix and ollama ports from config."""
    config = LokiConfig(ports=PortsConfig(kiwix=9090, ollama=12000))
    mocker.patch("loki.cli.load_config", return_value=config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_get = mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)

    CliRunner().invoke(cli, ["status"])

    called_urls = {call.args[0] for call in mock_get.call_args_list}
    assert any(":9090" in url for url in called_urls)
    assert any(":12000" in url for url in called_urls)
