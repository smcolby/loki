"""Tests for the status subcommand — service health checks."""

import requests
from click.testing import CliRunner

from loki.cli import cli
from loki.config import LokiConfig, PortsConfig


def _mock_subprocess_not_found(mocker):
    """Return a subprocess mock that simulates containers not found (docker inspect fails)."""
    proc = mocker.MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    return mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)


def test_status_prints_online_when_services_respond(mocker, sample_config):
    """The status command prints ONLINE for services when they return HTTP 200."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

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
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "Ollama: OFFLINE" in result.output
    assert "Kiwix: OFFLINE" in result.output


def test_status_prints_offline_on_non_200(mocker, sample_config):
    """The status command prints OFFLINE when a service returns a non-200 HTTP status."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 503
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "OFFLINE" in result.output
    assert "503" in result.output


def test_status_checks_three_http_endpoints(mocker, sample_config):
    """The status command makes three HTTP GET requests: Ollama (localhost), Ollama (network),
    and Kiwix.
    """
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_get = mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

    CliRunner().invoke(cli, ["status"])

    assert mock_get.call_count == 3


def test_status_uses_configured_ports(mocker):
    """The status command uses the kiwix and ollama ports from config."""
    config = LokiConfig(ports=PortsConfig(kiwix=9090, ollama=12000))
    mocker.patch("loki.cli.load_config", return_value=config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_get = mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

    CliRunner().invoke(cli, ["status"])

    called_urls = {call.args[0] for call in mock_get.call_args_list}
    assert any(":9090" in url for url in called_urls)
    assert any(":12000" in url for url in called_urls)


# ---------------------------------------------------------------------------
# Ollama network binding check
# ---------------------------------------------------------------------------


def test_status_prints_ollama_network_online(mocker, sample_config):
    """The status command reports the Ollama network binding as ONLINE when the LAN IP responds."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    # conftest stubs get_local_ip to return "192.168.1.100"
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "Ollama (network): ONLINE" in result.output


def test_status_warns_ollama_bound_to_loopback(mocker, sample_config):
    """The status command warns when Ollama is online on localhost but not on the LAN IP."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    ok = mocker.MagicMock(spec=requests.Response)
    ok.status_code = 200
    # Localhost Ollama succeeds; LAN IP Ollama fails; Kiwix succeeds.
    mocker.patch(
        "loki.cli.requests.get",
        autospec=True,
        side_effect=[ok, requests.exceptions.ConnectionError("refused"), ok],
    )
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "Ollama (network): OFFLINE" in result.output
    assert "127.0.0.1" in result.output
    assert "loki setup" in result.output


def test_status_skips_network_check_when_no_local_ip(mocker, sample_config):
    """The status command skips the Ollama network check when no local IP can be determined."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.get_local_ip", return_value="")
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "Ollama (network): SKIP" in result.output


# ---------------------------------------------------------------------------
# Docker container status
# ---------------------------------------------------------------------------


def test_status_prints_container_running(mocker, sample_config):
    """The status command reports a container as RUNNING when docker inspect returns 'running'."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    proc = mocker.MagicMock()
    proc.returncode = 0
    proc.stdout = "running\n"
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)

    result = CliRunner().invoke(cli, ["status"])

    assert "loki-open-webui: RUNNING" in result.output
    assert "loki-caddy: RUNNING" in result.output
    assert "loki-kiwix: RUNNING" in result.output


def test_status_prints_container_not_found(mocker, sample_config):
    """The status command reports a container as NOT FOUND when docker inspect fails."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "NOT FOUND" in result.output


def test_status_prints_container_non_running_state(mocker, sample_config):
    """The status command reports the actual state when a container exists but is not running."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    proc = mocker.MagicMock()
    proc.returncode = 0
    proc.stdout = "exited\n"
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)

    result = CliRunner().invoke(cli, ["status"])

    assert "NOT RUNNING (exited)" in result.output


def test_status_skips_docker_when_not_installed(mocker, sample_config):
    """The status command skips container checks when docker is not on PATH."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=mock_response)
    mocker.patch("loki.cli.shutil.which", return_value=None)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["status"])

    mock_run.assert_not_called()
    assert "not installed" in result.output
