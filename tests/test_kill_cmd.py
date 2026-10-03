"""Tests for the kill subcommand: restarting Open WebUI, the gateway, and every engine."""

from click.testing import CliRunner

from loki.cli import cli
from loki.config import ImageEngineConfig


def test_kill_restarts_open_webui_gateway_and_every_engine(
    mocker, sample_config, add_strata_engines
):
    """The kill command restarts every service holding requests and leaves Caddy and Kiwix."""
    add_strata_engines(sample_config, "qwen", "swift")
    sample_config.image.engines = {
        "base": ImageEngineConfig(model="qwen-image", args=["--diffusion-model", "/m.gguf"])
    }
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)
    mock_run.return_value.returncode = 0

    result = CliRunner().invoke(cli, ["kill"])

    assert result.exit_code == 0
    cmd = mock_run.call_args.args[0]
    assert cmd[:2] == ["docker", "compose"]
    assert cmd[-7:] == [
        "restart",
        "open-webui",
        "gateway",
        "llama",
        "strata-qwen",
        "strata-swift",
        "image-base",
    ]


def test_kill_fails_when_restart_fails(mocker, sample_config):
    """A failed docker compose restart exits non-zero with an error naming the cause."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)
    mock_run.return_value.returncode = 1

    result = CliRunner().invoke(cli, ["kill"])

    assert result.exit_code != 0
    assert "docker compose restart failed" in result.output
