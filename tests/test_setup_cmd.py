"""Tests for the setup subcommand — Caddyfile generation and ZIM file download via aria2c."""

import os
from pathlib import Path

from click.testing import CliRunner

from loki.cli import cli, _aria2c_threads
from loki.config import LokiConfig, PortsConfig


def test_setup_writes_caddyfile(mocker, sample_config, tmp_path):
    """The setup command writes a Caddyfile using the url from config."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["setup"])

    content = (tmp_path / "Caddyfile").read_text()
    assert "http://loki.local" in content
    assert "reverse_proxy open-webui:8080" in content


def test_setup_writes_env_file(mocker, sample_config, tmp_path):
    """The setup command writes a .env file with the configured port values."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["setup"])

    content = (tmp_path / ".env").read_text()
    assert "CADDY_PORT=80" in content
    assert "KIWIX_PORT=8080" in content
    assert "OLLAMA_PORT=11434" in content


def test_setup_env_file_uses_custom_ports(mocker, tmp_path):
    """The setup command writes custom port values to the .env file."""
    config = LokiConfig(ports=PortsConfig(caddy=8000, kiwix=9090, ollama=12000))
    mocker.patch("loki.cli.load_config", return_value=config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["setup"])

    content = (tmp_path / ".env").read_text()
    assert "CADDY_PORT=8000" in content
    assert "KIWIX_PORT=9090" in content
    assert "OLLAMA_PORT=12000" in content


def test_setup_uses_default_caddy_url_when_missing(mocker, tmp_path):
    """The setup command uses the default url when it is absent from config."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    caddy_out = tmp_path / "Caddyfile"
    mocker.patch("loki.cli.caddyfile_path", return_value=caddy_out)
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["setup"])

    assert LokiConfig().url in caddy_out.read_text()


def test_setup_prints_caddyfile_confirmation(mocker, sample_config, tmp_path):
    """The setup command prints a confirmation message after writing the Caddyfile."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["setup"])

    assert "Caddyfile written" in result.output
    assert "loki.local" in result.output


def test_setup_prints_port_confirmation(mocker, sample_config, tmp_path):
    """The setup command prints a confirmation message after writing the .env file."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["setup"])

    assert "caddy=80" in result.output
    assert "kiwix=8080" in result.output
    assert "ollama=11434" in result.output


def test_aria2c_threads_is_half_cpu_count(mocker):
    """_aria2c_threads returns half the logical CPU count."""
    mocker.patch("loki.cli.os.cpu_count", return_value=8)
    assert _aria2c_threads() == 4


def test_aria2c_threads_minimum_is_one(mocker):
    """_aria2c_threads returns at least 1 even on a single-core machine."""
    mocker.patch("loki.cli.os.cpu_count", return_value=1)
    assert _aria2c_threads() == 1


def test_aria2c_threads_handles_none_cpu_count(mocker):
    """_aria2c_threads falls back gracefully when os.cpu_count() returns None."""
    mocker.patch("loki.cli.os.cpu_count", return_value=None)
    assert _aria2c_threads() >= 1


def test_setup_calls_aria2c_for_missing_file(mocker, sample_config, tmp_path):
    """The setup command invokes aria2c with the correct arguments for a missing ZIM file."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.os.cpu_count", return_value=8)
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["setup"])

    expected_url = sample_config.kiwix_files[0].url
    mock_run.assert_called_once_with(
        ["aria2c", "-x", "4", "-s", "4", "-d", str(tmp_path), expected_url],
        check=False,
    )


def test_setup_skips_existing_file(mocker, sample_config, tmp_path):
    """The setup command skips a ZIM file that already exists in the data directory."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    filename = Path(sample_config.kiwix_files[0].url).name
    (tmp_path / filename).touch()

    CliRunner().invoke(cli, ["setup"])

    mock_run.assert_not_called()


def test_setup_prints_skip_message(mocker, sample_config, tmp_path):
    """The setup command prints a skip message when the ZIM file already exists."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    filename = Path(sample_config.kiwix_files[0].url).name
    (tmp_path / filename).touch()

    result = CliRunner().invoke(cli, ["setup"])

    assert "Skipping" in result.output
    assert filename in result.output


def test_setup_empty_kiwix_files(mocker, tmp_path):
    """The setup command prints a message and makes no subprocess calls when kiwix_files is empty."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["setup"])

    mock_run.assert_not_called()
    assert "No kiwix_files" in result.output


def test_setup_prints_failure_on_nonzero_exit(mocker, sample_config, tmp_path):
    """The setup command prints an error message when aria2c exits with a non-zero code."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_result = mocker.MagicMock()
    mock_result.returncode = 1
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=mock_result)

    result = CliRunner().invoke(cli, ["setup"])

    assert "failed" in result.output.lower()
