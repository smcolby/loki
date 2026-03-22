"""Tests for config.py — YAML loading, path resolution, and Caddyfile generation."""

import textwrap

import pytest
from pydantic import ValidationError

from loki.config import (
    build_caddyfile, build_env_file, caddyfile_path, env_file_path,
    kiwix_dir, load_config,
    KiwixFile, LokiConfig, PortsConfig,
    REPO_ROOT,
)


def test_load_config_parses_kiwix_files(tmp_path):
    """load_config returns a LokiConfig with a populated kiwix_files list."""
    config_text = textwrap.dedent("""\
        kiwix_files:
          - name: wikipedia_en_test
            url: https://download.kiwix.org/zim/wikipedia/wikipedia_en_test.zim
        ollama_models:
          - llama3:8b
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)

    config = load_config(config_file)

    assert len(config.kiwix_files) == 1
    assert config.kiwix_files[0].name == "wikipedia_en_test"
    assert config.kiwix_files[0].url.endswith(".zim")


def test_load_config_parses_ollama_models(tmp_path):
    """load_config returns a LokiConfig with the correct ollama_models list."""
    config_text = textwrap.dedent("""\
        kiwix_files: []
        ollama_models:
          - llama3:8b
          - qwen3:30b
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)

    config = load_config(config_file)

    assert config.ollama_models == ["llama3:8b", "qwen3:30b"]


def test_load_config_empty_lists(tmp_path):
    """load_config handles configs with empty kiwix_files and ollama_models."""
    config_text = "kiwix_files: []\nollama_models: []\n"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)

    config = load_config(config_file)

    assert config.kiwix_files == []
    assert config.ollama_models == []


def test_load_config_applies_defaults_for_missing_keys(tmp_path):
    """load_config fills in model defaults when optional keys are absent."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n")

    config = load_config(config_file)

    assert config.url == "loki.local"
    assert config.ports.caddy == 80
    assert config.ports.kiwix == 8080
    assert config.ports.ollama == 11434


def test_load_config_raises_on_invalid_port_type(tmp_path):
    """load_config raises a ValidationError when a port value is not an integer."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ports:\n  caddy: not_a_number\n")

    with pytest.raises(ValidationError):
        load_config(config_file)


def test_load_config_raises_on_missing_kiwix_url(tmp_path):
    """load_config raises a ValidationError when a kiwix_files entry is missing the url field."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("kiwix_files:\n  - name: test\n")

    with pytest.raises(ValidationError):
        load_config(config_file)


def test_kiwix_dir_is_under_repo_root():
    """kiwix_dir returns a path inside the repository root."""
    assert kiwix_dir() == REPO_ROOT / "data" / "kiwix"


def test_repo_root_contains_pyproject():
    """REPO_ROOT points to the actual repository root (contains pyproject.toml)."""
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_build_caddyfile_contains_url():
    """build_caddyfile returns content that routes the given URL to Open WebUI."""
    result = build_caddyfile("loki.local")
    assert "http://loki.local" in result
    assert "reverse_proxy open-webui:8080" in result


def test_build_caddyfile_uses_custom_url():
    """build_caddyfile uses the provided URL, not the default."""
    result = build_caddyfile("myserver.local")
    assert "http://myserver.local" in result
    assert "loki.local" not in result


def test_loki_config_default_url_is_local_tld():
    """LokiConfig defaults to a .local TLD for mDNS compatibility."""
    assert LokiConfig().url.endswith(".local")


def test_caddyfile_path_is_under_repo_root():
    """caddyfile_path returns the Caddyfile path at the repository root."""
    assert caddyfile_path() == REPO_ROOT / "Caddyfile"


def test_ports_config_defaults():
    """PortsConfig provides default values when no ports are specified."""
    ports = PortsConfig()
    assert ports.caddy == 80
    assert ports.kiwix == 8080
    assert ports.ollama == 11434


def test_ports_config_partial_override():
    """PortsConfig uses the provided value for a given key and defaults for the rest."""
    ports = PortsConfig(kiwix=9090)
    assert ports.kiwix == 9090
    assert ports.caddy == 80
    assert ports.ollama == 11434


def test_ports_config_full_override():
    """PortsConfig accepts fully custom port values."""
    ports = PortsConfig(caddy=8000, kiwix=9090, ollama=12000)
    assert ports.caddy == 8000
    assert ports.kiwix == 9090
    assert ports.ollama == 12000


def test_build_env_file_contains_all_ports():
    """build_env_file returns content with CADDY_PORT, KIWIX_PORT, and OLLAMA_PORT."""
    content = build_env_file(PortsConfig())
    assert "CADDY_PORT=80" in content
    assert "KIWIX_PORT=8080" in content
    assert "OLLAMA_PORT=11434" in content


def test_build_env_file_uses_custom_ports():
    """build_env_file reflects custom port values."""
    content = build_env_file(PortsConfig(caddy=8000, kiwix=9090, ollama=12000))
    assert "CADDY_PORT=8000" in content
    assert "KIWIX_PORT=9090" in content
    assert "OLLAMA_PORT=12000" in content


def test_env_file_path_is_under_repo_root():
    """env_file_path returns the .env path at the repository root."""
    assert env_file_path() == REPO_ROOT / ".env"
