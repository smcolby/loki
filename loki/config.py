"""Configuration loading and path resolution for loki."""

import hashlib
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Only accurate for editable installs; kept for backward-compatibility with tests
REPO_ROOT = Path(__file__).parent.parent

STRATA_ENGINE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


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
    api : int
        Host port for the gateway's OpenAI- and Anthropic-compatible API, which
        serves every engine's models. Default is 8090.
    """

    model_config = ConfigDict(extra="forbid")

    caddy: int = 80
    kiwix: int = 8080
    api: int = 8090


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


class StrataConfig(BaseModel):
    """Settings for the optional Strata containers and the models they serve.

    Attributes
    ----------
    engines : dict of str to str
        Engine name to its engine config file, relative to ``data_dir``. Each
        entry runs one container (Compose service ``strata-<name>``) serving
        one model; an empty mapping leaves Strata out of the stack.
    data_dir : Path
        Directory holding the engine configs and the model files they name.
        Mounted read-only into every Strata container at the same absolute path.
    ref : str
        Strata commit the image is built from.
    gpu_targets : str
        AMD GPU architecture the engine is compiled for (e.g. ``"gfx1100"``).
    rocm_version : str
        Version of AMD's TheRock ROCm wheels used to build and run the engine.
    """

    model_config = ConfigDict(extra="forbid")

    engines: dict[str, str] = {}
    data_dir: Path = Field(default=Path("~/.llms/strata"), validate_default=True)
    ref: str = "1678de333d0e0711bc414ad992b640e1a37dd814"
    gpu_targets: str = "gfx1100"
    rocm_version: str = "7.10.0a20251120"

    @field_validator("engines")
    @classmethod
    def _valid_engines(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject names unusable in a service name and configs outside ``data_dir``."""
        for name, config_file in value.items():
            if not STRATA_ENGINE_NAME.fullmatch(name):
                raise ValueError(
                    f"Strata engine name {name!r} must be lowercase letters, digits, "
                    "'_' or '-', starting with a letter or digit"
                )
            path = Path(config_file)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"Strata engine {name!r} config {config_file!r} must be relative to data_dir"
                )
        return value

    @field_validator("data_dir")
    @classmethod
    def _absolute_data_dir(cls, value: Path) -> Path:
        """Expand ``~`` and anchor relative paths so the container mount matches the host."""
        return value.expanduser().absolute()

    def services(self) -> dict[str, Path]:
        """Return each engine's Compose service name and the engine config it mounts."""
        return {f"strata-{name}": self.data_dir / file for name, file in self.engines.items()}


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
    strata : StrataConfig
        Strata image and model settings.
    kiwix_files : list of KiwixFile
        ZIM files to download during setup.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = "loki.local"
    ports: PortsConfig = PortsConfig()
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    strata: StrataConfig = Field(default_factory=StrataConfig)
    kiwix_files: list[KiwixFile] = []

    def engine_services(self) -> list[str]:
        """Return the Compose services that serve models, in gateway listing order."""
        return ["llama", *self.strata.services()]


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


def strata_compose_path() -> Path:
    """Return the path to the generated Compose file for Strata engines under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/compose.strata.yaml``
    """
    return _default_root() / "compose.strata.yaml"


GATEWAY_SOURCES = ("Dockerfile", "requirements.txt", "gateway.py")


def gateway_dir() -> Path:
    """Return the gateway image's build context under ``LOKI_ROOT``.

    Returns
    -------
    Path
        ``<LOKI_ROOT>/gateway``
    """
    return _default_root() / "gateway"


def gateway_tag(source_dir: Path) -> str:
    """Return an image tag that changes whenever the gateway's build inputs change.

    Parameters
    ----------
    source_dir : Path
        Directory holding the gateway's Dockerfile, requirements, and source.

    Returns
    -------
    str
        First 12 hex digits of a SHA-256 over the build inputs.

    Raises
    ------
    OSError
        If a build input is missing or unreadable.
    """
    digest = hashlib.sha256()
    for name in GATEWAY_SOURCES:
        digest.update(name.encode() + b"\0" + (source_dir / name).read_bytes() + b"\0")
    return digest.hexdigest()[:12]


def image_names(config: LokiConfig, gateway: str) -> dict[str, str]:
    """Return the ``repo:tag`` each locally built Compose service runs.

    Tags carry every build input, so changing the GPU target, ROCm version,
    or commit names a new image and an existing image is never rebuilt in place.

    Parameters
    ----------
    config : LokiConfig
        Configuration holding each engine's build settings.
    gateway : str
        Tag from :func:`gateway_tag`.

    Returns
    -------
    dict of str to str
        Service name (``llama``, ``strata``, ``gateway``) to image name.
    """
    llama, strata = config.llama, config.strata
    return {
        "llama": f"loki-llama:{llama.gpu_targets}-rocm{llama.rocm_version}-{llama.ref}",
        "strata": f"loki-strata:{strata.gpu_targets}-rocm{strata.rocm_version}-{strata.ref}",
        "gateway": f"loki-gateway:{gateway}",
    }


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


def build_env_file(config: LokiConfig, images: dict[str, str]) -> str:
    """Return .env file content with ports, images, and engine settings for Docker Compose.

    Parameters
    ----------
    config : LokiConfig
        Configuration whose ports and engine settings are written to the file.
    images : dict of str to str
        Image names from :func:`image_names`.

    Returns
    -------
    str
        Contents of the .env file as a string.
    """
    ports, llama = config.ports, config.llama
    engines = ",".join(f"{name}=http://{name}:8080" for name in config.engine_services())
    return (
        "# Generated by loki from config.yaml; do not edit by hand.\n"
        f"CADDY_PORT={ports.caddy}\n"
        f"KIWIX_PORT={ports.kiwix}\n"
        f"API_PORT={ports.api}\n"
        f"LOKI_ENGINES={engines}\n"
        f"GATEWAY_IMAGE={images['gateway']}\n"
        f"LLAMA_IMAGE={images['llama']}\n"
        f"LLAMA_MODELS_DIR={llama.models_dir}\n"
        f"LLAMA_CPP_REF={llama.ref}\n"
        f"LLAMA_GPU_TARGETS={llama.gpu_targets}\n"
        f"LLAMA_ROCM_VERSION={llama.rocm_version}\n"
        f"LLAMA_MAX_LOADED={llama.max_loaded}\n"
    )


def build_strata_compose(config: LokiConfig, image: str, context: Path) -> str:
    """Return a Compose file with one service per configured Strata engine.

    Every service runs the same image and mounts ``data_dir`` read-only at its
    host path, so paths inside an engine config resolve unchanged; only the
    engine config mounted at ``/etc/strata/strata.json`` differs.

    Parameters
    ----------
    config : LokiConfig
        Configuration whose ``strata`` section lists the engines.
    image : str
        Strata image name from :func:`image_names`.
    context : Path
        Build context holding Strata's Dockerfile.

    Returns
    -------
    str
        Compose YAML; its ``services`` mapping is empty when no engine is configured.
    """
    strata = config.strata
    data_dir = str(strata.data_dir)
    args = {
        "STRATA_REF": strata.ref,
        "GPU_TARGETS": strata.gpu_targets,
        "ROCM_VERSION": strata.rocm_version,
    }

    # Give each service its own dicts so the YAML carries no anchors
    services = {
        service: {
            "build": {"context": str(context), "args": dict(args)},
            "image": image,
            "container_name": f"loki-{service}",
            "restart": "unless-stopped",
            "devices": ["/dev/kfd", "/dev/dri"],
            "security_opt": ["seccomp=unconfined"],
            "ulimits": {"memlock": -1},
            "volumes": [
                f"{data_dir}:{data_dir}:ro",
                {
                    "type": "bind",
                    "source": str(config_file),
                    "target": "/etc/strata/strata.json",
                    "read_only": True,
                    "bind": {"create_host_path": False},
                },
            ],
            "networks": ["loki-net"],
        }
        for service, config_file in strata.services().items()
    }
    header = "# Generated by loki from config.yaml; do not edit by hand.\n"
    return header + yaml.safe_dump({"services": services}, sort_keys=False)
