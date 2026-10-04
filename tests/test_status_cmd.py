"""Tests for the status subcommand: service health checks."""

import pytest
import requests
from click.testing import CliRunner

from loki.cli import _model_states, cli
from loki.config import LokiConfig, PortsConfig


def _mock_subprocess_not_found(mocker):
    """Return a subprocess mock that simulates containers not found (docker inspect fails)."""
    proc = mocker.MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    return mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)


def _response(mocker, status_code=200, payload=None):
    """Return a requests.Response stand-in with a status code and JSON body."""
    response = mocker.MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    return response


def _get_by_url(mocker, responses: dict[str, object]):
    """Patch ``requests.get`` to answer each URL suffix from ``responses``."""

    def _get(url, *_, **__):
        for suffix, response in responses.items():
            if url.endswith(suffix):
                return response
        return _response(mocker)

    return mocker.patch("loki.cli.requests.get", autospec=True, side_effect=_get)


def test_status_prints_online_when_services_respond(mocker, sample_config):
    """The status command prints ONLINE for services when they return HTTP 200."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "Model API: ONLINE" in result.output
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

    assert "Model API: OFFLINE" in result.output
    assert "Kiwix: OFFLINE" in result.output


def test_status_prints_offline_on_non_200(mocker, sample_config):
    """The status command prints OFFLINE when a service returns a non-200 HTTP status."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker, 503))
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "Model API: OFFLINE — HTTP 503" in result.output


def test_status_uses_configured_ports(mocker):
    """The status command uses the kiwix and api ports from config."""
    config = LokiConfig(ports=PortsConfig(kiwix=9090, api=9000))
    mocker.patch("loki.cli.load_config", return_value=config)
    mock_get = mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    _mock_subprocess_not_found(mocker)

    CliRunner().invoke(cli, ["status"])

    called_urls = {call.args[0] for call in mock_get.call_args_list}
    assert "http://localhost:9090" in called_urls
    assert "http://localhost:9000/health" in called_urls
    assert "http://localhost:9000/v1/models" in called_urls


def test_status_lists_engines_and_models(mocker, sample_config):
    """The status command lists each engine, the one holding the GPU, and each model's state."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    health = {"status": "ok", "active": "strata", "engines": {"llama": "idle", "strata": "loaded"}}
    models = {
        "data": [
            {"id": "alpha", "owned_by": "llama", "status": {"value": "unloaded"}},
            {"id": "flash", "owned_by": "strata", "status": "loaded"},
        ]
    }
    _get_by_url(
        mocker,
        {"/health": _response(mocker, 200, health), "/v1/models": _response(mocker, 200, models)},
    )
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "engine llama: idle\n" in result.output
    assert "engine strata: loaded (holds the GPU)" in result.output
    assert "alpha [llama]: unloaded" in result.output
    assert "flash [strata]: loaded" in result.output


def test_status_skips_model_list_when_gateway_offline(mocker, sample_config):
    """The status command does not query the model list when the gateway is down."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_get = mocker.patch(
        "loki.cli.requests.get",
        autospec=True,
        side_effect=requests.exceptions.ConnectionError("refused"),
    )
    _mock_subprocess_not_found(mocker)

    CliRunner().invoke(cli, ["status"])

    assert not any("/v1/models" in call.args[0] for call in mock_get.call_args_list)


# ---------------------------------------------------------------------------
# _model_states
# ---------------------------------------------------------------------------


def test_model_states_reads_gateway_listing(mocker):
    """Each model id maps to its engine and the value of its status object."""
    payload = {"data": [{"id": "alpha", "owned_by": "llama", "status": {"value": "loading"}}]}
    mocker.patch(
        "loki.cli.requests.get", autospec=True, return_value=_response(mocker, 200, payload)
    )

    assert _model_states(8090) == {"alpha": ("llama", "loading")}


def test_model_states_accepts_plain_status_string(mocker):
    """A status given as a bare string is reported as-is."""
    payload = {"data": [{"id": "alpha", "owned_by": "strata", "status": "loaded"}]}
    mocker.patch(
        "loki.cli.requests.get", autospec=True, return_value=_response(mocker, 200, payload)
    )

    assert _model_states(8090) == {"alpha": ("strata", "loaded")}


def test_model_states_skips_entries_without_id(mocker):
    """Entries that are not objects or lack an id are ignored; missing fields read as unknown."""
    payload = {"data": ["alpha", {"status": {"value": "loaded"}}, {"id": "beta"}]}
    mocker.patch(
        "loki.cli.requests.get", autospec=True, return_value=_response(mocker, 200, payload)
    )

    assert _model_states(8090) == {"beta": ("unknown", "unknown")}


def test_model_states_empty_on_unexpected_shape(mocker):
    """A listing whose data field is not a list yields no models."""
    mocker.patch(
        "loki.cli.requests.get", autospec=True, return_value=_response(mocker, 200, {"data": {}})
    )

    assert _model_states(8090) == {}


def test_model_states_empty_on_invalid_json(mocker):
    """A response body that is not JSON yields no models."""
    response = _response(mocker)
    response.json.side_effect = ValueError("not json")
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=response)

    assert _model_states(8090) == {}


def test_model_states_empty_on_http_error(mocker):
    """A non-200 listing yields no models."""
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker, 500))

    assert _model_states(8090) == {}


# ---------------------------------------------------------------------------
# Docker container status
# ---------------------------------------------------------------------------


def test_status_prints_container_running(mocker, sample_config):
    """The status command reports a container as RUNNING when docker inspect returns 'running'."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    proc = mocker.MagicMock()
    proc.returncode = 0
    proc.stdout = "running\n"
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)

    result = CliRunner().invoke(cli, ["status"])

    assert "loki-gateway: RUNNING" in result.output
    assert "loki-llama: RUNNING" in result.output
    assert "loki-strata" not in result.output
    assert "loki-open-webui: RUNNING" in result.output
    assert "loki-caddy: RUNNING" in result.output
    assert "loki-kiwix: RUNNING" in result.output


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_status_lists_kokoro_only_when_enabled(mocker, sample_config, enabled):
    """The Kokoro container is checked only when text-to-speech is enabled."""
    sample_config.tts.enabled = enabled
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    proc = mocker.MagicMock()
    proc.returncode = 0
    proc.stdout = "running\n"
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)

    result = CliRunner().invoke(cli, ["status"])

    assert ("loki-kokoro: RUNNING" in result.output) is enabled


def test_status_prints_container_not_found(mocker, sample_config):
    """The status command reports a container as NOT FOUND when docker inspect fails."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "NOT FOUND" in result.output


def test_status_prints_container_non_running_state(mocker, sample_config):
    """The status command reports the actual state when a container exists but is not running."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    proc = mocker.MagicMock()
    proc.returncode = 0
    proc.stdout = "exited\n"
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=proc)

    result = CliRunner().invoke(cli, ["status"])

    assert "NOT RUNNING (exited)" in result.output


def test_status_skips_docker_when_not_installed(mocker, sample_config):
    """The status command skips container checks when docker is not on PATH."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    mocker.patch("loki.cli.shutil.which", return_value=None)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["status"])

    mock_run.assert_not_called()
    assert "not installed" in result.output


def test_status_checks_each_strata_container(mocker, sample_config, add_strata_engines):
    """The status command checks one container per configured Strata engine."""
    add_strata_engines(sample_config, "qwen", "swift")
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.requests.get", autospec=True, return_value=_response(mocker))
    _mock_subprocess_not_found(mocker)

    result = CliRunner().invoke(cli, ["status"])

    assert "loki-strata-qwen: NOT FOUND" in result.output
    assert "loki-strata-swift: NOT FOUND" in result.output
