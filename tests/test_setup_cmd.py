"""Tests for the setup subcommand — system setup prompts, Caddyfile generation, and ZIM
downloads.
"""

import os
import subprocess
from pathlib import Path

from click.testing import CliRunner

from loki.cli import _aria2c_threads, cli
from loki.config import LokiConfig, PortsConfig


def test_setup_writes_caddyfile(mocker, sample_config, tmp_path):
    """The setup command writes a Caddyfile using the url from config."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    CliRunner().invoke(cli, ["setup"], input="y\n")

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

    CliRunner().invoke(cli, ["setup"], input="y\n")

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

    CliRunner().invoke(cli, ["setup"], input="y\n")

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

    CliRunner().invoke(cli, ["setup"], input="y\n")

    assert LokiConfig().url in caddy_out.read_text()


def test_setup_prints_caddyfile_confirmation(mocker, sample_config, tmp_path):
    """The setup command prints a confirmation message after writing the Caddyfile."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "Caddyfile written" in result.output
    assert "loki.local" in result.output


def test_setup_prints_port_confirmation(mocker, sample_config, tmp_path):
    """The setup command prints a confirmation message after writing the .env file."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

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

    CliRunner().invoke(cli, ["setup"], input="y\n")

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

    CliRunner().invoke(cli, ["setup"], input="y\n")

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

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "Skipping" in result.output
    assert filename in result.output


def test_setup_empty_kiwix_files(mocker, tmp_path):
    """The setup command prints a message and makes no subprocess calls when kiwix_files is
    empty.
    """
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_run = mocker.patch("loki.cli.subprocess.run", autospec=True)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_run.assert_not_called()
    assert "No kiwix_files" in result.output


def test_setup_prints_failure_on_nonzero_exit(mocker, sample_config, tmp_path):
    """The setup command prints an error message when aria2c exits with a non-zero code."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_result = mocker.MagicMock(spec=subprocess.CompletedProcess)
    mock_result.returncode = 1
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=mock_result)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "failed" in result.output.lower()


def test_setup_copies_default_config_when_missing(mocker, tmp_path):
    """The setup command copies the bundled default config when config.yaml is absent."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "kiwix")
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_copy = mocker.patch("loki.cli.shutil.copy")

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_copy.assert_called_once()
    assert "Created" in result.output


def test_setup_does_not_overwrite_existing_config(mocker, tmp_path):
    """The setup command does not overwrite config.yaml when it already exists."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "kiwix")
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    (tmp_path / "config.yaml").write_text("url: custom.local\n")
    mock_copy = mocker.patch("loki.cli.shutil.copy")

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_copy.assert_not_called()
    assert (tmp_path / "config.yaml").read_text() == "url: custom.local\n"


# ---------------------------------------------------------------------------
# Config review (Step 0)
# ---------------------------------------------------------------------------


def test_setup_config_review_decline_exits(mocker, tmp_path):
    """Declining the config review prompt exits setup without writing any files."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "kiwix")
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("click.confirm", return_value=False)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert not (tmp_path / "Caddyfile").exists()
    assert result.exit_code != 0


def test_setup_config_review_decline_shows_path(mocker, tmp_path):
    """Declining the config review prompt prints the config file path."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "kiwix")
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("click.confirm", return_value=False)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert str(tmp_path / "config.yaml") in result.output


def test_setup_config_review_displays_config_contents(mocker, tmp_path):
    """The setup command displays the config.yaml contents for review."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "kiwix")
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("url: loki.local\n")

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "url: loki.local" in result.output


# ---------------------------------------------------------------------------
# System packages (Step 1)
# ---------------------------------------------------------------------------


def test_setup_prompts_to_install_missing_packages(mocker, tmp_path):
    """The setup command prompts to install packages when any are missing."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_installed", return_value=False)
    mocker.patch("loki.cli.install_docker", return_value=True)
    mocker.patch("loki.cli.install_ollama", return_value=True)
    mocker.patch("click.confirm", return_value=True)
    mock_install = mocker.patch("loki.cli.install_packages", return_value=True)

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_install.assert_called_once()


def test_setup_skips_package_prompt_when_all_installed(mocker, tmp_path):
    """The setup command skips the package install prompt when all tools are present."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    # is_installed is True via conftest autouse — no prompt expected.
    mock_install = mocker.patch("loki.cli.install_packages")

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_install.assert_not_called()


def test_setup_package_install_decline_prints_readme_notice(mocker, tmp_path):
    """Declining the package install prompt prints a 'see README' message."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_installed", return_value=False)
    mocker.patch("click.confirm", side_effect=[True, False])  # confirm config; decline pkgs

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "README" in result.output


def test_setup_warns_when_no_package_manager_found(mocker, tmp_path):
    """The setup command warns when no supported package manager is found."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_installed", return_value=False)
    mocker.patch("loki.cli.detect_package_manager", return_value=None)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "no supported package manager" in result.output.lower()


# ---------------------------------------------------------------------------
# Docker (Step 2)
# ---------------------------------------------------------------------------


def test_setup_prompts_to_install_docker_when_missing(mocker, tmp_path):
    """The setup command prompts to install Docker when it is not on PATH."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_installed", side_effect=lambda cmd: cmd != "docker")
    mocker.patch("click.confirm", return_value=True)
    mock_install = mocker.patch("loki.cli.install_docker", return_value=True)

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_install.assert_called_once()


def test_setup_prints_docker_group_notice_after_install(mocker, tmp_path):
    """After installing Docker, setup prints a notice about the docker group."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_installed", side_effect=lambda cmd: cmd != "docker")
    mocker.patch("click.confirm", return_value=True)
    mocker.patch("loki.cli.install_docker", return_value=True)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "log out" in result.output.lower() or "newgrp" in result.output


# ---------------------------------------------------------------------------
# Ollama (Step 3)
# ---------------------------------------------------------------------------


def test_setup_prompts_to_install_ollama_when_missing(mocker, tmp_path):
    """The setup command prompts to install Ollama when it is not on PATH."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_installed", side_effect=lambda cmd: cmd != "ollama")
    mocker.patch("click.confirm", return_value=True)
    mock_install = mocker.patch("loki.cli.install_ollama", return_value=True)

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_install.assert_called_once()


def test_setup_skips_ollama_prompt_when_installed(mocker, tmp_path):
    """The setup command does not prompt to install Ollama when it is already on PATH."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mock_install = mocker.patch("loki.cli.install_ollama")

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_install.assert_not_called()


# ---------------------------------------------------------------------------
# Ollama binding (Step 4)
# ---------------------------------------------------------------------------


def test_setup_prompts_to_configure_ollama_binding_when_not_set(mocker, tmp_path):
    """The setup command prompts to configure Ollama binding when not already set."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.is_ollama_binding_configured", return_value=False)
    mocker.patch("click.confirm", return_value=True)
    mock_configure = mocker.patch("loki.cli.configure_ollama_binding", return_value=True)

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_configure.assert_called_once()


def test_setup_skips_ollama_binding_when_already_configured(mocker, tmp_path):
    """The setup command skips the Ollama binding step when override is already in place."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    # is_ollama_binding_configured is True via conftest autouse.
    mock_configure = mocker.patch("loki.cli.configure_ollama_binding")

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_configure.assert_not_called()


# ---------------------------------------------------------------------------
# LOKI_ROOT shell profile (Step 5)
# ---------------------------------------------------------------------------


def test_setup_adds_loki_root_to_profile_when_not_exported(mocker, tmp_path):
    """The setup command adds LOKI_ROOT to the shell profile when not already present."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.loki_root_already_exported", return_value=False)
    mocker.patch("click.confirm", return_value=True)
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("LOKI_ROOT", None)
    mock_add = mocker.patch("loki.cli.add_loki_root_to_profile", return_value=True)

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_add.assert_called_once()


def test_setup_skips_loki_root_prompt_when_already_exported(mocker, tmp_path):
    """The setup command skips the LOKI_ROOT prompt when it is already in the profile."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    # loki_root_already_exported is True via conftest autouse.
    mock_add = mocker.patch("loki.cli.add_loki_root_to_profile")

    CliRunner().invoke(cli, ["setup"], input="y\n")

    mock_add.assert_not_called()


def test_setup_loki_root_prints_source_instruction(mocker, tmp_path):
    """After adding LOKI_ROOT to the profile, setup prints a 'source profile' message."""
    mocker.patch("loki.cli.load_config", return_value=LokiConfig())
    mocker.patch("loki.cli.loki_root", return_value=tmp_path)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.loki_root_already_exported", return_value=False)
    mocker.patch("click.confirm", return_value=True)
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("LOKI_ROOT", None)
    mocker.patch("loki.cli.add_loki_root_to_profile", return_value=True)

    result = CliRunner().invoke(cli, ["setup"], input="y\n")

    assert "source" in result.output
