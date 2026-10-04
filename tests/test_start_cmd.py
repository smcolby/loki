"""Tests for the start subcommand: generated files, compose up, and mDNS."""

import yaml
from click.testing import CliRunner

from loki.cli import cli
from loki.config import LokiConfig


def test_start_runs_compose_up(mocker, sample_config):
    """The start command brings the compose stack up detached, removing orphaned containers."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    commands = [call.args[0] for call in mock_run.call_args_list]
    assert any(
        cmd[:2] == ["docker", "compose"] and cmd[-3:] == ["up", "-d", "--remove-orphans"]
        for cmd in commands
    )


def test_start_writes_models_preset(mocker, sample_config, tmp_path, write_preset):
    """The start command regenerates models.ini from the presets under models_dir."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    write_preset(sample_config.llama.models_dir, "alpha")

    result = CliRunner().invoke(cli, ["start"])

    assert "[alpha]" in (tmp_path / "models.ini").read_text()
    assert "alpha" in result.output


def test_start_writes_env_file(mocker, sample_config, tmp_path):
    """The start command rewrites .env so config edits reach Compose."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)

    CliRunner().invoke(cli, ["start"])

    assert "LLAMA_CPP_REF=abc123" in (tmp_path / ".env").read_text()


def test_start_aborts_on_invalid_preset(mocker, sample_config, tmp_path, write_preset):
    """The start command exits without starting the stack when a preset is invalid."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    write_preset(sample_config.llama.models_dir, "broken", "[broken]\nmodel = missing.gguf\n")
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["start"])

    assert result.exit_code != 0
    assert "invalid model preset" in result.output
    mock_run.assert_not_called()
    assert not (tmp_path / "models.ini").exists()


def test_start_warns_when_no_presets(mocker, sample_config):
    """The start command warns when the models directory has no presets."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)

    result = CliRunner().invoke(cli, ["start"])

    assert "no */preset.ini" in result.output


def test_start_prints_api_url_with_engines(mocker, sample_config):
    """The start command prints the gateway API address and the engines behind it."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)

    result = CliRunner().invoke(cli, ["start"])

    assert "Model API (llama): http://loki.local:8090/v1" in result.output


def test_start_reports_the_preload_model(mocker, sample_config):
    """With a preload model set, start says the model is loading in the background."""
    sample_config.preload = "swift-flash"
    mocker.patch("loki.cli.load_config", return_value=sample_config)

    result = CliRunner().invoke(cli, ["start"])

    assert "Loading swift-flash in the background" in result.output


def test_start_writes_strata_services_and_routes(
    mocker, sample_config, tmp_path, add_strata_engines
):
    """With Strata engines configured, start writes their services and routes each one."""
    add_strata_engines(sample_config, "qwen", "swift")
    mocker.patch("loki.cli.load_config", return_value=sample_config)

    result = CliRunner().invoke(cli, ["start"])

    env = (tmp_path / ".env").read_text()
    assert "strata-qwen=http://strata-qwen:8080,strata-swift=http://strata-swift:8080\n" in env
    services = yaml.safe_load((tmp_path / "compose.engines.yaml").read_text())["services"]
    assert list(services) == ["strata-qwen", "strata-swift"]
    assert "Model API (llama, strata-qwen, strata-swift)" in result.output


def test_start_aborts_when_strata_config_missing(
    mocker, sample_config, tmp_path, add_strata_engines
):
    """A configured engine without its engine config stops start before writing or starting."""
    data_dir = add_strata_engines(sample_config, "qwen", "swift")
    (data_dir / "swift.json").unlink()
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["start"])

    assert result.exit_code != 0
    assert f"Strata engine config not found: {data_dir / 'swift.json'}" in result.output
    mock_run.assert_not_called()
    assert not (tmp_path / ".env").exists()


def test_start_uses_both_compose_files(mocker, sample_config, tmp_path):
    """Every Compose call reads compose.yaml and the generated engines file."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["start"])

    cmd = mock_run.call_args_list[0].args[0]
    files = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-f"]
    assert files == [str(tmp_path / "compose.yaml"), str(tmp_path / "compose.engines.yaml")]


def test_start_exits_when_docker_not_found(mocker, sample_config):
    """The start command exits with an error message when docker is not on PATH."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.shutil.which", return_value=None)

    result = CliRunner().invoke(cli, ["start"])

    assert result.exit_code != 0
    assert "docker" in result.output


# ---------------------------------------------------------------------------
# mDNS / avahi-publish-address
# ---------------------------------------------------------------------------


def test_start_broadcasts_mdns_for_local_url(mocker, sample_config):
    """The start command spawns avahi-publish-address when the URL ends in .local."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_called_once()


def test_start_avahi_called_with_correct_hostname(mocker, sample_config):
    """The start command calls start_avahi_publish with the configured URL."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    CliRunner().invoke(cli, ["start"])

    call_args = mock_avahi.call_args
    assert call_args[0][0] == "loki.local"


def test_start_skips_mdns_for_non_local_url(mocker):
    """The start command skips avahi-publish-address when the URL does not end in .local."""
    config = LokiConfig(url="loki.home")
    mocker.patch("loki.cli.load_config", return_value=config)
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    result = CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_not_called()
    assert "skipping mdns broadcast" in result.output.lower()


def test_start_skips_mdns_and_warns_when_ip_unavailable(mocker, sample_config):
    """The start command warns and skips mDNS when no local IP can be determined."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.get_local_ip", return_value="")
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    result = CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_not_called()
    assert "Warning" in result.output


def test_start_warns_when_avahi_publish_not_installed(mocker, sample_config):
    """The start command warns and skips mDNS when avahi-publish-address is absent."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    # avahi-publish-address missing; all other tools present
    mocker.patch("loki.cli.is_installed", side_effect=lambda cmd: cmd != "avahi-publish-address")
    mock_avahi = mocker.patch("loki.cli.start_avahi_publish")

    result = CliRunner().invoke(cli, ["start"])

    mock_avahi.assert_not_called()
    assert "Warning" in result.output
    assert "avahi-publish-address" in result.output
