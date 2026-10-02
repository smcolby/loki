"""Tests for config.py — YAML loading, path resolution, and Caddyfile generation."""

import textwrap

import pytest
from pydantic import ValidationError

from loki.config import (
    DEFAULT_PRESET_SETTINGS,
    REPO_ROOT,
    LlamaConfig,
    LokiConfig,
    PortsConfig,
    build_caddyfile,
    build_env_file,
    caddyfile_path,
    env_file_path,
    kiwix_dir,
    load_config,
    models_preset_path,
)


def test_load_config_parses_kiwix_files(tmp_path):
    """load_config returns a LokiConfig with a populated kiwix_files list."""
    config_text = textwrap.dedent("""\
        kiwix_files:
          - name: wikipedia_en_test
            url: https://download.kiwix.org/zim/wikipedia/wikipedia_en_test.zim
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)

    config = load_config(config_file)

    assert len(config.kiwix_files) == 1
    assert config.kiwix_files[0].name == "wikipedia_en_test"
    assert config.kiwix_files[0].url.endswith(".zim")


def test_load_config_parses_llama_section(tmp_path):
    """load_config reads the llama image settings and replaces the default preset settings."""
    config_text = textwrap.dedent(f"""\
        llama:
          models_dir: {tmp_path / "models"}
          ref: deadbeef
          gpu_targets: gfx1201
          rocm_version: 7.2.4
          max_loaded: 2
          defaults:
            n-gpu-layers: 40
            jinja: true
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)

    llama = load_config(config_file).llama

    assert llama.models_dir == tmp_path / "models"
    assert (llama.ref, llama.gpu_targets, llama.max_loaded) == ("deadbeef", "gfx1201", 2)
    assert llama.defaults == {"n-gpu-layers": 40, "jinja": True}


def test_load_config_empty_lists(tmp_path):
    """load_config handles configs with an empty kiwix_files list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("kiwix_files: []\n")

    config = load_config(config_file)

    assert config.kiwix_files == []


@pytest.mark.parametrize(
    "text",
    ["ollama_models:\n  - llama3:8b\n", "ports:\n  ollama: 11434\n", "llama:\n  model_dir: /x\n"],
    ids=["ollama-models", "ollama-port", "misspelled-llama-key"],
)
def test_load_config_rejects_unknown_keys(tmp_path, text):
    """load_config fails loudly on keys it does not understand, including retired Ollama keys."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(text)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(config_file)


def test_load_config_applies_defaults_for_missing_keys(tmp_path):
    """load_config fills in model defaults when optional keys are absent."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n")

    config = load_config(config_file)

    assert config.url == "loki.local"
    assert config.ports.caddy == 80
    assert config.ports.kiwix == 8080
    assert config.ports.llama == 8090
    assert config.llama.defaults == DEFAULT_PRESET_SETTINGS


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


def test_load_config_raises_on_missing_file(tmp_path):
    """load_config exits with a clear message when config.yaml does not exist."""
    with pytest.raises(SystemExit, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_raises_on_malformed_yaml(tmp_path):
    """load_config exits with a clear message when the YAML is syntactically invalid."""
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text(":\n  bad: [unclosed\n")
    with pytest.raises(SystemExit, match="invalid YAML"):
        load_config(bad_config)


def test_kiwix_dir_is_under_loki_root(monkeypatch, tmp_path):
    """kiwix_dir returns a path inside the directory set by LOKI_ROOT."""
    monkeypatch.setenv("LOKI_ROOT", str(tmp_path))
    assert kiwix_dir() == tmp_path / "data" / "kiwix"


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


def test_models_dir_expands_home(monkeypatch, tmp_path):
    """A models_dir starting with ~ resolves against the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert LlamaConfig(models_dir="~/weights").models_dir == tmp_path / "weights"


def test_default_models_dir_expands_home(monkeypatch, tmp_path):
    """The default models_dir is expanded too, so the container mount path is absolute."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert LlamaConfig().models_dir == tmp_path / ".llms"


def test_relative_models_dir_becomes_absolute(monkeypatch, tmp_path):
    """A relative models_dir is anchored to the current directory."""
    monkeypatch.chdir(tmp_path)
    assert LlamaConfig(models_dir="models").models_dir == tmp_path / "models"


def test_default_preset_settings_are_not_shared_between_configs():
    """Mutating one config's defaults leaves new configs untouched."""
    first = LlamaConfig()
    first.defaults["n-gpu-layers"] = 1
    assert LlamaConfig().defaults["n-gpu-layers"] == DEFAULT_PRESET_SETTINGS["n-gpu-layers"]


def test_models_preset_path_is_under_loki_root(monkeypatch, tmp_path):
    """models_preset_path returns models.ini inside the LOKI_ROOT directory."""
    monkeypatch.setenv("LOKI_ROOT", str(tmp_path))
    assert models_preset_path() == tmp_path / "models.ini"


def test_caddyfile_path_is_under_loki_root(monkeypatch, tmp_path):
    """caddyfile_path returns the Caddyfile path inside the LOKI_ROOT directory."""
    monkeypatch.setenv("LOKI_ROOT", str(tmp_path))
    assert caddyfile_path() == tmp_path / "Caddyfile"


def test_ports_config_defaults():
    """PortsConfig provides default values when no ports are specified."""
    ports = PortsConfig()
    assert ports.caddy == 80
    assert ports.kiwix == 8080
    assert ports.llama == 8090


def test_ports_config_partial_override():
    """PortsConfig uses the provided value for a given key and defaults for the rest."""
    ports = PortsConfig(kiwix=9090)
    assert ports.kiwix == 9090
    assert ports.caddy == 80
    assert ports.llama == 8090


def test_ports_config_full_override():
    """PortsConfig accepts fully custom port values."""
    ports = PortsConfig(caddy=8000, kiwix=9090, llama=9000)
    assert ports.caddy == 8000
    assert ports.kiwix == 9090
    assert ports.llama == 9000


def test_build_env_file_contains_all_ports():
    """build_env_file returns content with CADDY_PORT, KIWIX_PORT, and LLAMA_PORT."""
    content = build_env_file(LokiConfig())
    assert "CADDY_PORT=80" in content
    assert "KIWIX_PORT=8080" in content
    assert "LLAMA_PORT=8090" in content


def test_build_env_file_uses_custom_ports():
    """build_env_file reflects custom port values."""
    content = build_env_file(LokiConfig(ports=PortsConfig(caddy=8000, kiwix=9090, llama=9000)))
    assert "CADDY_PORT=8000" in content
    assert "KIWIX_PORT=9090" in content
    assert "LLAMA_PORT=9000" in content


def test_build_env_file_contains_llama_build_settings(tmp_path):
    """build_env_file passes the models mount and image build arguments to Compose."""
    llama = LlamaConfig(
        models_dir=tmp_path,
        ref="deadbeef",
        gpu_targets="gfx1201",
        rocm_version="7.2.4",
        max_loaded=2,
    )

    lines = build_env_file(LokiConfig(llama=llama)).splitlines()

    assert f"LLAMA_MODELS_DIR={tmp_path}" in lines
    assert "LLAMA_CPP_REF=deadbeef" in lines
    assert "LLAMA_GPU_TARGETS=gfx1201" in lines
    assert "LLAMA_MAX_LOADED=2" in lines
    assert "ROCM_VERSION=7.2.4" in lines


def test_env_file_path_is_under_loki_root(monkeypatch, tmp_path):
    """env_file_path returns the .env path inside the LOKI_ROOT directory."""
    monkeypatch.setenv("LOKI_ROOT", str(tmp_path))
    assert env_file_path() == tmp_path / ".env"
