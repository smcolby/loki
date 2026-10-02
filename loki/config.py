"""Configuration loading and path resolution for loki."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Only accurate for editable installs; kept for backward-compatibility with tests
REPO_ROOT = Path(__file__).parent.parent


def _default_root() -> Path:
    """Return the working root directory for config and data path resolution.

    Checks the ``LOKI_ROOT`` environment variable first, then falls back to
    the current working directory. This makes loki work correctly whether
    installed as an editable package or a standard wheel.

    Returns
    -------
    Path
        The resolved root directory as an absolute ``Path``.
    """
    env = os.environ.get("LOKI_ROOT")
    return Path(env) if env else Path.cwd()


class KiwixFile(BaseModel):
    """A ZIM file entry from the configuration.

    Attributes
    ----------
    name : str
        Human-readable label for the ZIM file.
    url : str
        Direct download URL for the .zim file.
    """

    name: str
    url: str


class PortsConfig(BaseModel):
    """Host port assignments for Docker Compose services.

    Attributes
    ----------
    caddy : int
        Host port for the Caddy reverse proxy. Default is 80.
    kiwix : int
        Host port for the Kiwix server. Default is 8080.
    llama : int
        Host port for the llama-server API. Default is 8090.
    """

    model_config = ConfigDict(extra="forbid")

    caddy: int = 80
    kiwix: int = 8080
    llama: int = 8090


# Settings applied to every model unless its preset.ini overrides them
DEFAULT_PRESET_SETTINGS: dict[str, str | int | float | bool] = {
    "n-gpu-layers": 99,
    "fit": "off",
    "flash-attn": "on",
    "cache-type-k": "q4_0",
    "cache-type-v": "q4_0",
    "parallel": 1,
    "cache-ram": 8192,
    "ctx-checkpoints": 32,
}


class LlamaConfig(BaseModel):
    """Settings for the llama-server container and the models it serves.

    Attributes
    ----------
    models_dir : Path
        Directory whose subdirectories each hold GGUF files and a ``preset.ini``.
        Mounted read-only into the container at the same absolute path.
    ref : str
        llama.cpp commit the image is built from.
    gpu_targets : str
        AMD GPU architecture the image is compiled for (e.g. ``"gfx1100"``).
    rocm_version : str
        ROCm release of the build toolchain and bundled runtime libraries.
    max_loaded : int
        Number of models held in VRAM at once; the least recently used is
        unloaded when another is requested.
    defaults : dict of str to scalar
        llama-server options applied to every model, written as the preset's
        global section. Keys are long option names without leading dashes.
    """

    model_config = ConfigDict(extra="forbid")

    models_dir: Path = Field(default=Path("~/.llms"), validate_default=True)
    ref: str = "5d806aa2575e01e126651fd69ab1ab6cefff861d"
    gpu_targets: str = "gfx1100"
    rocm_version: str = "7.2.4"
    max_loaded: int = 1
    defaults: dict[str, str | int | float | bool] = Field(
        default_factory=lambda: dict(DEFAULT_PRESET_SETTINGS)
    )

    @field_validator("models_dir")
    @classmethod
    def _absolute_models_dir(cls, value: Path) -> Path:
        """Expand ``~`` and anchor relative paths so the container mount matches the host."""
        return value.expanduser().absolute()


class LokiConfig(BaseModel):
    """Top-level configuration for the loki stack.

    Attributes
    ----------
    url : str
        Local hostname to expose via Caddy. Default is ``"loki.local"``.
    ports : PortsConfig
        Host port assignments for each service.
    llama : LlamaConfig
        llama-server image and model settings.
    kiwix_files : list of KiwixFile
        ZIM files to download during setup.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = "loki.local"
    ports: PortsConfig = PortsConfig()
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    kiwix_files: list[KiwixFile] = []


def load_config(path: Path | None = None) -> LokiConfig:
    """Load and validate config.yaml, returning a LokiConfig instance.

    Parameters
    ----------
    path : Path or None, optional
        Explicit path to the config file. If ``None``, defaults to
        ``$LOKI_ROOT/config.yaml`` when ``LOKI_ROOT`` is set, otherwise
        ``./config.yaml`` relative to the current working directory.

    Returns
    -------
    LokiConfig
        Validated configuration parsed from the YAML file.

    Raises
    ------
    SystemExit
        If the config file does not exist or contains invalid YAML.
    """
    config_path = path or _default_root() / "config.yaml"
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise SystemExit(
            f"Error: config file not found: {config_path}\n"
            "Run 'loki setup' to create it from the bundled defaults."
        ) from None
    except yaml.YAMLError as exc:
        raise SystemExit(f"Error: invalid YAML in {config_path}: {exc}") from exc
    return LokiConfig.model_validate(data)


def kiwix_dir() -> Path:
    """Return the path to the kiwix data directory under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/data/kiwix``
    """
    return _default_root() / "data" / "kiwix"


def caddyfile_path() -> Path:
    """Return the path to the Caddyfile under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/Caddyfile``
    """
    return _default_root() / "Caddyfile"


def env_file_path() -> Path:
    """Return the path to the generated ``.env`` file under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/.env``
    """
    return _default_root() / ".env"


def loki_root() -> Path:
    """Return the working root directory (``LOKI_ROOT`` env var or ``cwd``).

    Returns
    -------
    Path
        The resolved root directory; equivalent to ``_default_root()``.
    """
    return _default_root()


def avahi_pid_file() -> Path:
    """Return the path to the avahi-publish PID file under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/.avahi.pid``
    """
    return _default_root() / ".avahi.pid"


def models_preset_path() -> Path:
    """Return the path to the generated llama-server models preset under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/models.ini``
    """
    return _default_root() / "models.ini"


def build_caddyfile(caddy_url: str) -> str:
    """Return the Caddyfile content that routes a URL to Open WebUI.

    Parameters
    ----------
    caddy_url : str
        Local hostname to expose (e.g. ``"loki.local"``).

    Returns
    -------
    str
        Caddyfile configuration as a string.
    """
    return (
        f"# Route traffic for the configured local URL to Open WebUI.\n"
        f"http://{caddy_url} {{\n"
        f"    reverse_proxy open-webui:8080\n"
        f"}}\n"
    )


def build_env_file(config: LokiConfig) -> str:
    """Return .env file content with ports and llama-server build settings for Docker Compose.

    Parameters
    ----------
    config : LokiConfig
        Configuration whose ports and llama settings are written to the file.

    Returns
    -------
    str
        Contents of the .env file as a string.
    """
    ports = config.ports
    llama = config.llama
    return (
        "# Generated by loki from config.yaml; do not edit by hand.\n"
        f"CADDY_PORT={ports.caddy}\n"
        f"KIWIX_PORT={ports.kiwix}\n"
        f"LLAMA_PORT={ports.llama}\n"
        f"LLAMA_MODELS_DIR={llama.models_dir}\n"
        f"LLAMA_CPP_REF={llama.ref}\n"
        f"LLAMA_GPU_TARGETS={llama.gpu_targets}\n"
        f"LLAMA_MAX_LOADED={llama.max_loaded}\n"
        f"ROCM_VERSION={llama.rocm_version}\n"
    )
