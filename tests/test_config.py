"""Tests for config.py: YAML loading, path resolution, image names, and generated files."""

import textwrap

import pytest
import yaml
from pydantic import ValidationError

from loki.config import (
    DEFAULT_PRESET_SETTINGS,
    REPO_ROOT,
    ImageConfig,
    ImageEngineConfig,
    LlamaConfig,
    LokiConfig,
    PortsConfig,
    StrataConfig,
    TtsConfig,
    build_caddyfile,
    build_engines_compose,
    build_env_file,
    caddyfile_path,
    engines_compose_path,
    env_file_path,
    gateway_tag,
    image_names,
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
    [
        "ollama_models:\n  - llama3:8b\n",
        "ports:\n  ollama: 11434\n",
        "ports:\n  llama: 8090\n",
        "llama:\n  model_dir: /x\n",
    ],
    ids=["ollama-models", "ollama-port", "retired-llama-port", "misspelled-llama-key"],
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
    assert config.ports.api == 8090
    assert config.llama.defaults == DEFAULT_PRESET_SETTINGS
    assert config.strata.engines == {}
    assert config.engine_services() == ["llama"]


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


def test_build_caddyfile_sends_the_strata_dashboard_to_the_gateway():
    """Paths under /strata go to the gateway, ahead of the Open WebUI catch-all."""
    result = build_caddyfile("loki.local")

    assert "\thandle /strata* {\n\t\treverse_proxy gateway:8080" in result
    assert result.index("gateway:8080") < result.index("open-webui:8080")


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
    assert ports.api == 8090


def test_ports_config_partial_override():
    """PortsConfig uses the provided value for a given key and defaults for the rest."""
    ports = PortsConfig(kiwix=9090)
    assert ports.kiwix == 9090
    assert ports.caddy == 80
    assert ports.api == 8090


def test_ports_config_full_override():
    """PortsConfig accepts fully custom port values."""
    ports = PortsConfig(caddy=8000, kiwix=9090, api=9000)
    assert ports.caddy == 8000
    assert ports.kiwix == 9090
    assert ports.api == 9000


IMAGES = {
    "llama": "loki-llama:x",
    "strata": "loki-strata:y",
    "image": "loki-image:w",
    "gateway": "loki-gateway:z",
}
QWEN_IMAGE = ImageEngineConfig(
    model="qwen-image-2.1-q8_0", args=["--diffusion-model", "/m/q.gguf", "--steps", "20"]
)


def test_build_env_file_contains_all_ports():
    """build_env_file returns content with CADDY_PORT, KIWIX_PORT, and API_PORT."""
    lines = build_env_file(LokiConfig(), IMAGES).splitlines()
    assert "CADDY_PORT=80" in lines
    assert "KIWIX_PORT=8080" in lines
    assert "API_PORT=8090" in lines


def test_build_env_file_uses_custom_ports():
    """build_env_file reflects custom port values."""
    config = LokiConfig(ports=PortsConfig(caddy=8000, kiwix=9090, api=9000))
    lines = build_env_file(config, IMAGES).splitlines()
    assert "CADDY_PORT=8000" in lines
    assert "KIWIX_PORT=9090" in lines
    assert "API_PORT=9000" in lines


def test_build_env_file_contains_llama_build_settings(tmp_path):
    """build_env_file passes the models mount, image, and build arguments to Compose."""
    llama = LlamaConfig(
        models_dir=tmp_path,
        ref="deadbeef",
        gpu_targets="gfx1201",
        rocm_version="7.2.4",
        max_loaded=2,
    )

    lines = build_env_file(LokiConfig(llama=llama), IMAGES).splitlines()

    assert "LLAMA_IMAGE=loki-llama:x" in lines
    assert f"LLAMA_MODELS_DIR={tmp_path}" in lines
    assert "LLAMA_CPP_REF=deadbeef" in lines
    assert "LLAMA_GPU_TARGETS=gfx1201" in lines
    assert "LLAMA_MAX_LOADED=2" in lines
    assert "LLAMA_ROCM_VERSION=7.2.4" in lines
    assert "GATEWAY_IMAGE=loki-gateway:z" in lines


def test_build_env_file_without_engines_routes_only_llama():
    """Without Strata or image engines, the gateway sees only llama."""
    lines = build_env_file(LokiConfig(), IMAGES).splitlines()

    assert "LOKI_ENGINES=llama=http://llama:8080" in lines
    assert "LOKI_IMAGE_MODELS=" in lines


def test_build_env_file_routes_every_strata_engine(tmp_path):
    """Each Strata engine is routed under its service name, in config order."""
    strata = StrataConfig(engines={"qwen": "qwen.json", "swift": "swift.json"}, data_dir=tmp_path)

    lines = build_env_file(LokiConfig(strata=strata), IMAGES).splitlines()

    assert (
        "LOKI_ENGINES=llama=http://llama:8080,"
        "strata-qwen=http://strata-qwen:8080,strata-swift=http://strata-swift:8080"
    ) in lines


def test_build_env_file_routes_image_engines_with_their_model_ids():
    """Image engines follow Strata in the routes, and each publishes its configured id."""
    image = ImageConfig(engines={"qwen": QWEN_IMAGE})

    lines = build_env_file(LokiConfig(image=image), IMAGES).splitlines()

    assert "LOKI_ENGINES=llama=http://llama:8080,image-qwen=http://image-qwen:8080" in lines
    assert "LOKI_IMAGE_MODELS=image-qwen=qwen-image-2.1-q8_0" in lines


def test_build_engines_compose_runs_one_container_per_strata_engine(tmp_path):
    """Each engine gets a service on the shared image that mounts its own engine config."""
    strata = StrataConfig(
        engines={"qwen": "qwen.json", "swift": "configs/swift.json"},
        data_dir=tmp_path,
        ref="cafe",
        gpu_targets="gfx1100",
        rocm_version="7.10",
    )

    text = build_engines_compose(LokiConfig(strata=strata), IMAGES, tmp_path / "root")
    services = yaml.safe_load(text)["services"]

    assert list(services) == ["strata-qwen", "strata-swift"]
    swift = services["strata-swift"]
    assert swift["image"] == "loki-strata:y"
    assert swift["container_name"] == "loki-strata-swift"
    assert swift["build"] == {
        "context": str(tmp_path / "root" / "strata"),
        "args": {"STRATA_REF": "cafe", "GPU_TARGETS": "gfx1100", "ROCM_VERSION": "7.10"},
    }
    assert swift["volumes"][0] == f"{tmp_path}:{tmp_path}:ro"
    assert swift["volumes"][1]["source"] == str(tmp_path / "configs" / "swift.json")
    assert swift["volumes"][1]["target"] == "/etc/strata/strata.json"
    assert swift["environment"] == ["STRATA_ALLOWED_HOSTS=strata-swift"]
    assert swift["networks"] == ["loki-net"]
    assert "&" not in text


def test_build_engines_compose_runs_image_engines_with_their_args(tmp_path):
    """An image engine runs sd-server with its arguments and the models directory mounted."""
    image = ImageConfig(
        engines={"qwen": QWEN_IMAGE},
        models_dir=tmp_path,
        ref="beef",
        gpu_targets="gfx1201",
        rocm_version="7.2.4",
    )

    text = build_engines_compose(LokiConfig(image=image), IMAGES, tmp_path / "root")
    services = yaml.safe_load(text)["services"]

    assert list(services) == ["image-qwen"]
    qwen = services["image-qwen"]
    assert qwen["image"] == "loki-image:w"
    assert qwen["container_name"] == "loki-image-qwen"
    assert qwen["build"] == {
        "context": str(tmp_path / "root" / "image"),
        "args": {"SD_CPP_REF": "beef", "GPU_TARGETS": "gfx1201", "ROCM_VERSION": "7.2.4"},
    }
    assert qwen["command"] == ["--diffusion-model", "/m/q.gguf", "--steps", "20"]
    assert qwen["volumes"] == [f"{tmp_path}:{tmp_path}:ro"]
    assert qwen["devices"] == ["/dev/kfd", "/dev/dri"]


def test_build_engines_compose_runs_kokoro_on_the_cpu(tmp_path):
    """Enabled Kokoro runs its published image with the default voice and no GPU devices."""
    tts = TtsConfig(enabled=True, image="kokoro:cpu", voice="bf_emma")

    text = build_engines_compose(LokiConfig(tts=tts), IMAGES, tmp_path)

    assert yaml.safe_load(text)["services"] == {
        "kokoro": {
            "image": "kokoro:cpu",
            "container_name": "loki-kokoro",
            "restart": "unless-stopped",
            "environment": ["API_LOG_LEVEL=WARNING", "DEFAULT_VOICE=bf_emma"],
            "networks": ["loki-net"],
        }
    }


def test_build_engines_compose_without_engines_has_no_services(tmp_path):
    """Without Strata or image engines the file still parses, with no services."""
    text = build_engines_compose(LokiConfig(), IMAGES, tmp_path)

    assert yaml.safe_load(text) == {"services": {}}


@pytest.mark.parametrize(
    ("engines", "message"),
    [
        ({"Qwen": "q.json"}, "lowercase"),
        ({"-qwen": "q.json"}, "lowercase"),
        ({"qwen": "/abs/q.json"}, "relative to data_dir"),
        ({"qwen": "../q.json"}, "relative to data_dir"),
    ],
    ids=["uppercase", "leading-hyphen", "absolute", "parent"],
)
def test_strata_rejects_bad_engines(engines, message):
    """Engine names must fit a Compose service name and configs must stay in data_dir."""
    with pytest.raises(ValidationError, match=message):
        StrataConfig(engines=engines)


@pytest.mark.parametrize(
    ("engines", "message"),
    [
        ({"Qwen": {"model": "q"}}, "lowercase"),
        ({"qwen": {"model": "a,b"}}, "image model id"),
        ({"qwen": {"model": "a=b"}}, "image model id"),
        ({"qwen": {}}, "model"),
    ],
    ids=["uppercase-name", "comma-id", "equals-id", "no-id"],
)
def test_image_rejects_bad_engines(engines, message):
    """Engine names must fit a service name and model ids must fit the gateway's list."""
    with pytest.raises(ValidationError, match=message):
        ImageConfig.model_validate({"engines": engines})


@pytest.mark.parametrize(
    ("preload", "line"),
    [("swift-flash", "LOKI_PRELOAD=swift-flash"), (None, "LOKI_PRELOAD=")],
    ids=["set", "unset"],
)
def test_build_env_file_passes_the_preload_model(preload, line):
    """The gateway receives the preload model id, or an empty value to load nothing."""
    lines = build_env_file(LokiConfig(preload=preload), IMAGES).splitlines()

    assert line in lines


def test_preload_rejects_an_id_that_breaks_the_env_file():
    """A preload id with whitespace would split the .env line, so config loading rejects it."""
    with pytest.raises(ValidationError, match="preload model id"):
        LokiConfig(preload="qwen flash")


def test_image_names_carry_every_build_input():
    """Image tags name the GPU target, ROCm version, and commit, so a change names a new image."""
    config = LokiConfig(
        llama=LlamaConfig(ref="aaa", gpu_targets="gfx1201", rocm_version="7.2.4"),
        strata=StrataConfig(ref="bbb", gpu_targets="gfx1100", rocm_version="7.10.0a1"),
        image=ImageConfig(ref="ccc", gpu_targets="gfx1030", rocm_version="7.2.3"),
    )

    assert image_names(config, "0123abcd") == {
        "llama": "loki-llama:gfx1201-rocm7.2.4-aaa",
        "strata": "loki-strata:gfx1100-rocm7.10.0a1-bbb",
        "image": "loki-image:gfx1030-rocm7.2.3-ccc",
        "gateway": "loki-gateway:0123abcd",
    }


def test_gateway_tag_tracks_source_contents(gateway_sources):
    """The gateway tag is stable for unchanged sources and changes when any source changes."""
    first = gateway_tag(gateway_sources)
    assert gateway_tag(gateway_sources) == first
    assert len(first) == 12

    (gateway_sources / "gateway.py").write_text("changed\n")

    assert gateway_tag(gateway_sources) != first


def test_gateway_tag_fails_on_missing_source(tmp_path):
    """A gateway directory without its sources raises instead of naming an arbitrary image."""
    with pytest.raises(FileNotFoundError):
        gateway_tag(tmp_path)


def test_strata_data_dir_expands_home(monkeypatch, tmp_path):
    """The default Strata data_dir is absolute, and engine configs resolve inside it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    strata = StrataConfig(engines={"qwen": "qwen.json"})
    assert strata.data_dir == tmp_path / ".llms" / "strata"
    assert strata.services() == {"strata-qwen": tmp_path / ".llms" / "strata" / "qwen.json"}


def test_env_file_path_is_under_loki_root(monkeypatch, tmp_path):
    """env_file_path returns the .env path inside the LOKI_ROOT directory."""
    monkeypatch.setenv("LOKI_ROOT", str(tmp_path))
    assert env_file_path() == tmp_path / ".env"


def test_engines_compose_path_is_under_loki_root(monkeypatch, tmp_path):
    """engines_compose_path returns the generated Compose file inside LOKI_ROOT."""
    monkeypatch.setenv("LOKI_ROOT", str(tmp_path))
    assert engines_compose_path() == tmp_path / "compose.engines.yaml"
