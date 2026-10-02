"""CLI entry point for the loki management tool."""

import os
import shutil
import subprocess
from pathlib import Path

import click
import requests

from loki.config import (
    LokiConfig,
    avahi_pid_file,
    build_caddyfile,
    build_env_file,
    caddyfile_path,
    env_file_path,
    gateway_dir,
    gateway_tag,
    image_names,
    kiwix_dir,
    load_config,
    loki_root,
    models_preset_path,
)
from loki.presets import PresetError, build_models_preset
from loki.system import (
    PACKAGE_MAP,
    add_loki_root_to_profile,
    amd_gpu_present,
    detect_package_manager,
    detect_shell_profile,
    get_local_ip,
    install_docker,
    install_packages,
    is_installed,
    loki_root_already_exported,
    start_avahi_publish,
    stop_avahi_publish,
    upgrade_packages,
)

BUILT_SERVICES = ("gateway", "llama", "strata")


def _require_tool(name: str) -> None:
    """Exit with a clear message if ``name`` is not found on PATH.

    Provides better UX than letting subprocess raise a raw ``FileNotFoundError``.

    Parameters
    ----------
    name : str
        The command name to look up on PATH (e.g. ``"docker"``).

    Raises
    ------
    SystemExit
        If ``name`` is not found on PATH.
    """
    if shutil.which(name) is None:
        raise SystemExit(f"Error: '{name}' is not installed or not on PATH.")


def _compose(*args: str) -> list[str]:
    """Return a ``docker compose`` command rooted at ``LOKI_ROOT``."""
    return ["docker", "compose", "--project-directory", str(loki_root()), *args]


def _images(config: LokiConfig) -> dict[str, str]:
    """Return the image each locally built service runs, keyed by service name."""
    return image_names(config, gateway_tag(gateway_dir()))


def _needed_services(config: LokiConfig) -> list[str]:
    """Return the locally built services the configured stack runs."""
    return ["gateway", *config.engine_services()]


def _missing_services(config: LokiConfig) -> list[str]:
    """Return the needed services whose image is not present locally."""
    images = _images(config)
    return [
        service
        for service in _needed_services(config)
        if subprocess.run(
            ["docker", "image", "inspect", images[service]], capture_output=True, check=False
        ).returncode
        != 0
    ]


def _build_services(services: list[str]) -> bool:
    """Build images for ``services`` through Compose.

    Returns
    -------
    bool
        ``True`` if ``docker compose build`` exited with code 0.
    """
    return subprocess.run(_compose("build", *services), check=False).returncode == 0


def _api_url(config: LokiConfig) -> str:
    """Return the gateway's OpenAI-compatible base URL for LAN clients."""
    return f"http://{config.url}:{config.ports.api}/v1"


def _write_generated_files(config: LokiConfig) -> bool:
    """Write the Caddyfile, ``.env``, and ``models.ini`` derived from ``config``.

    Parameters
    ----------
    config : LokiConfig
        Configuration to render.

    Returns
    -------
    bool
        ``True`` when every file was written, ``False`` when Strata is enabled
        without its engine config or a model preset is invalid (the error is
        printed and ``models.ini`` is left untouched).
    """
    # Require Strata's engine config, which Compose would otherwise fail to mount
    strata_config = config.strata.config_file
    if config.strata.enabled and not strata_config.is_file():
        click.echo(
            f"Error: strata.enabled is true but {strata_config} does not exist "
            "(see README for its format).",
            err=True,
        )
        return False

    caddyfile_path().write_text(build_caddyfile(config.url))
    env_file_path().write_text(build_env_file(config, _images(config)))

    # Collect every model preset under the models directory
    models_dir = config.llama.models_dir
    try:
        preset, model_ids = build_models_preset(models_dir, config.llama.defaults)
    except PresetError as exc:
        click.echo(f"Error: invalid model preset: {exc}", err=True)
        return False
    models_preset_path().write_text(preset)

    if model_ids:
        click.echo(f"Models from {models_dir}: {', '.join(model_ids)}")
    else:
        click.echo(
            f"Warning: no */preset.ini under {models_dir}; llama-server will serve no models.",
            err=True,
        )
    return True


def _aria2c_threads() -> int:
    """Return half the number of logical CPU cores, with a minimum of 1."""
    return max(1, (os.cpu_count() or 2) // 2)


def _model_states(port: int) -> dict[str, tuple[str, str]]:
    """Return each served model id mapped to its engine and load status.

    Parameters
    ----------
    port : int
        Host port of the gateway API.

    Returns
    -------
    dict of str to tuple of (str, str)
        Model id to ``(engine, status)``, where status is ``"loaded"``,
        ``"unloaded"``, ``"loading"``, or ``"unknown"``; empty when the listing
        cannot be read.
    """
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        entries = r.json()["data"] if r.status_code == 200 else []
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return {}
    if not isinstance(entries, list):
        return {}

    states: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        status = entry.get("status")
        if isinstance(status, dict):
            status = status.get("value")
        engine = str(entry.get("owned_by", "unknown"))
        states[str(entry["id"])] = (engine, "unknown" if status is None else str(status))
    return states


@click.group()
def cli() -> None:
    """Manage the loki local LLM and offline knowledge server."""


@cli.command()
def setup() -> None:
    """Run the interactive loki setup wizard.

    Walks through config review, system-package installation (aria2,
    avahi-daemon, avahi-utils), Docker installation, an AMD GPU check, and the
    ``LOKI_ROOT`` shell-profile export. Then writes the Caddyfile, ``.env``, and
    ``models.ini``, offers to build any missing local images (gateway,
    llama-server, and Strata when enabled), and downloads any ZIM files listed
    in ``config.yaml``.
    """
    config_path = loki_root() / "config.yaml"
    if not config_path.exists():
        default = Path(__file__).parent / "config.default.yaml"
        shutil.copy(default, config_path)
        click.echo(f"Created {config_path} from defaults.")

    config = load_config()

    # Review configuration before proceeding
    try:
        click.echo(f"Configuration ({config_path}):\n")
        click.echo(config_path.read_text())
    except OSError:
        pass
    if not click.confirm("Proceed with this configuration?", default=True):
        raise SystemExit(f"Edit {config_path} and run `loki setup` again.")

    # Install system packages (aria2, avahi-daemon, avahi-utils)
    manager = detect_package_manager()
    if manager:
        pkg_map = PACKAGE_MAP[manager]
        missing_cmds = [cmd for cmd in pkg_map if not is_installed(cmd)]
        if missing_cmds:
            missing_pkgs = [pkg_map[cmd] for cmd in missing_cmds]
            click.echo(f"\nMissing packages: {', '.join(missing_pkgs)}")
            if click.confirm(f"Install with sudo {manager}?", default=True):
                if not install_packages(missing_pkgs, manager):
                    click.echo(
                        "Warning: package installation failed. See README for manual instructions.",
                        err=True,
                    )
                else:
                    click.echo("Packages installed.")
            else:
                click.echo("Skipping — see README for manual instructions.")
        else:
            click.echo("System packages already installed (aria2, avahi-daemon, avahi-utils).")
    else:
        click.echo(
            "Warning: no supported package manager found (apt-get, dnf). "
            "Install aria2, avahi-daemon, and avahi-utils manually.",
            err=True,
        )

    # Install Docker
    if not is_installed("docker"):
        click.echo("\nDocker is not installed.")
        if click.confirm("Install Docker via the official script with sudo?", default=True):
            if install_docker():
                click.echo("Docker installed.")
                click.echo(
                    "Note: log out and back in (or run `newgrp docker`) "
                    "before using docker without sudo."
                )
            else:
                click.echo(
                    "Warning: Docker installation failed. See README for manual instructions.",
                    err=True,
                )
        else:
            click.echo("Skipping — see README for manual instructions.")
    else:
        click.echo("Docker already installed.")

    # Check for the AMD GPU device nodes llama-server needs
    if amd_gpu_present():
        click.echo("AMD GPU devices found (/dev/kfd, /dev/dri).")
    else:
        click.echo(
            "Warning: /dev/kfd or /dev/dri is missing; install the amdgpu kernel driver "
            "before starting llama-server.",
            err=True,
        )

    # Add LOKI_ROOT to shell profile
    current_root = loki_root()
    if os.environ.get("LOKI_ROOT") != str(current_root):
        profile = detect_shell_profile()
        if not loki_root_already_exported(profile, current_root):
            click.echo(f"\nLOKI_ROOT is not set to {current_root}.")
            if click.confirm(f"Add `export LOKI_ROOT={current_root}` to {profile}?", default=True):
                if add_loki_root_to_profile(profile, current_root):
                    click.echo(
                        f"Added. Run `source {profile}` or open a new terminal "
                        "for it to take effect."
                    )
                else:
                    click.echo(
                        f"Warning: could not write to {profile}. "
                        f"Add manually: export LOKI_ROOT={current_root}",
                        err=True,
                    )
            else:
                click.echo("Skipping — see README for manual instructions.")
        else:
            click.echo(f"LOKI_ROOT already in {profile}; source it or open a new terminal.")
    else:
        click.echo(f"LOKI_ROOT already set to {current_root}.")

    click.echo("")

    # Write the Caddyfile, .env, and models.ini
    _write_generated_files(config)
    ports = config.ports
    click.echo(f"Caddyfile written for http://{config.url}")
    click.echo(
        f"Port configuration written: caddy={ports.caddy}, kiwix={ports.kiwix}, api={ports.api}"
    )

    # Build images missing for the configured engines, GPU, and commits
    if is_installed("docker"):
        missing = _missing_services(config)
        images = _images(config)
        if not missing:
            click.echo("\nLocal images already built.")
        elif click.confirm(
            "\nBuild missing images now (llama-server and Strata take 10-30 minutes each)?\n  "
            + "\n  ".join(images[service] for service in missing)
            + "\n",
            default=True,
        ):
            if _build_services(missing):
                click.echo("Images built.")
            else:
                click.echo("Warning: image build failed.", err=True)
        else:
            click.echo("Skipping; `loki start` builds missing images.")

    dest = kiwix_dir()
    dest.mkdir(parents=True, exist_ok=True)

    if not config.kiwix_files:
        click.echo("No kiwix_files entries found in config.yaml.")
        return

    for entry in config.kiwix_files:
        filename = Path(entry.url).name
        dest_file = dest / filename

        if dest_file.exists():
            click.echo(f"Skipping {filename} — already exists.")
            continue

        _require_tool("aria2c")
        click.echo(f"Downloading {entry.name} to {dest_file} ...")
        threads = str(_aria2c_threads())
        result = subprocess.run(
            ["aria2c", "-x", threads, "-s", threads, "-d", str(dest), entry.url],
            check=False,
        )
        if result.returncode != 0:
            click.echo(
                f"Download failed for {entry.name} (aria2c exit code {result.returncode}).",
                err=True,
            )
        else:
            click.echo(f"Finished downloading {entry.name}.")


@cli.command()
def update() -> None:
    """Update system packages and Docker Compose images.

    Upgrades aria2, avahi-daemon, and avahi-utils via the system package manager,
    pulls the latest published images, builds local images whose tag is missing
    (a changed commit, GPU target, ROCm version, or gateway source), and
    restarts the Compose stack if it is running. Existing local images are
    reused, since their tags name every build input.
    """
    # System packages
    manager = detect_package_manager()
    if manager:
        pkg_map = PACKAGE_MAP[manager]
        installed_pkgs = [pkg_map[cmd] for cmd in pkg_map if is_installed(cmd)]
        if installed_pkgs:
            click.echo(f"Upgrading system packages: {', '.join(installed_pkgs)}")
            if not upgrade_packages(installed_pkgs, manager):
                click.echo("Warning: system package upgrade failed.", err=True)
            else:
                click.echo("System packages upgraded.")
        else:
            click.echo("No loki system packages found to upgrade.")
    else:
        click.echo(
            "Warning: no supported package manager found; skipping system package upgrade.",
            err=True,
        )

    # Published Docker images
    _require_tool("docker")
    config = load_config()
    click.echo("\nPulling latest Docker images ...")
    pull = subprocess.run(_compose("pull", "--ignore-buildable"), check=False)
    if pull.returncode != 0:
        click.echo("Warning: docker compose pull failed.", err=True)
        return

    # Locally built images, only where config.yaml names one not yet built
    if not _write_generated_files(config):
        return
    missing = _missing_services(config)
    if missing:
        images = _images(config)
        click.echo(f"\nBuilding {', '.join(images[service] for service in missing)} ...")
        if not _build_services(missing):
            click.echo("Warning: image build failed.", err=True)
            return
    else:
        click.echo("\nLocal images are current.")

    # Restart the stack only if it is already running
    result = subprocess.run(_compose("ps", "-q"), capture_output=True, text=True, check=False)
    if result.stdout.strip():
        click.echo("Restarting Docker Compose stack with updated images ...")
        subprocess.run(_compose("up", "-d"), check=False)
    else:
        click.echo("Docker images updated. Start the stack with `loki start` when ready.")


@cli.command()
def start() -> None:
    """Write generated files, start the Docker Compose stack, and broadcast mDNS."""
    _require_tool("docker")
    config = load_config()

    # Refresh models.ini so preset edits take effect on this start
    if not _write_generated_files(config):
        raise SystemExit(1)

    # Compose builds any missing local image before starting its service
    click.echo("Starting Docker Compose stack ...")
    subprocess.run(_compose("up", "-d"), check=False)
    click.echo(f"Model API ({', '.join(config.engine_services())}): {_api_url(config)}")

    hostname = config.url
    if hostname.endswith(".local"):
        if not is_installed("avahi-publish-address"):
            click.echo(
                "Warning: avahi-publish-address not found; skipping mDNS broadcast. "
                "Run `loki setup` to install avahi-utils.",
                err=True,
            )
        else:
            ip = get_local_ip()
            if ip:
                start_avahi_publish(hostname, ip, avahi_pid_file())
                click.echo(f"Broadcasting {hostname} via mDNS (avahi-publish-address).")
            else:
                click.echo(
                    "Warning: could not determine local IP; skipping mDNS broadcast.",
                    err=True,
                )
    else:
        click.echo(f"Note: {hostname} does not use .local TLD; skipping mDNS broadcast.")


@cli.command()
def stop() -> None:
    """Stop the Docker Compose stack."""
    _require_tool("docker")
    stop_avahi_publish(avahi_pid_file())
    click.echo("Stopping Docker Compose stack ...")
    subprocess.run(_compose("down"), check=False)


@cli.command()
def status() -> None:
    """Check the health of running services."""
    config = load_config()
    click.echo("Checking service status ...")

    # Gateway health and each engine behind it
    api_url = f"http://localhost:{config.ports.api}/health"
    try:
        r = requests.get(api_url, timeout=5)
        health = r.json() if r.status_code == 200 else None
        if isinstance(health, dict):
            click.echo(f"  Model API: ONLINE ({api_url})")
        else:
            click.echo(f"  Model API: OFFLINE — HTTP {r.status_code} ({api_url})")
    except (requests.exceptions.RequestException, ValueError) as exc:
        health = None
        click.echo(f"  Model API: OFFLINE — {exc} ({api_url})")

    # Engine states, then served models and which are in VRAM
    if isinstance(health, dict):
        engines = health.get("engines")
        if isinstance(engines, dict):
            active = health.get("active")
            for name, state in engines.items():
                marker = " (holds the GPU)" if name == active else ""
                click.echo(f"    engine {name}: {state}{marker}")
        for model_id, (engine, state) in _model_states(config.ports.api).items():
            click.echo(f"    {model_id} [{engine}]: {state}")

    # Kiwix
    kiwix_url = f"http://localhost:{config.ports.kiwix}"
    try:
        r = requests.get(kiwix_url, timeout=5)
        if r.status_code == 200:
            click.echo(f"  Kiwix: ONLINE ({kiwix_url})")
        else:
            click.echo(f"  Kiwix: OFFLINE — HTTP {r.status_code} ({kiwix_url})")
    except requests.exceptions.RequestException as exc:
        click.echo(f"  Kiwix: OFFLINE — {exc} ({kiwix_url})")

    # Docker containers
    if shutil.which("docker"):
        engines = [f"loki-{service}" for service in config.engine_services()]
        for name in ("loki-gateway", *engines, "loki-open-webui", "loki-caddy", "loki-kiwix"):
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                state = result.stdout.strip()
                label = "RUNNING" if state == "running" else f"NOT RUNNING ({state})"
                click.echo(f"  {name}: {label}")
            else:
                click.echo(f"  {name}: NOT FOUND")
    else:
        click.echo("  Docker: not installed")


@cli.command()
def cleanup() -> None:
    """Remove ZIM files and locally built images no longer matching the config."""
    _require_tool("docker")
    config = load_config()

    # ZIM files
    kiwix = kiwix_dir()
    if kiwix.exists():
        expected_zims = {Path(entry.url).name for entry in config.kiwix_files}
        orphaned_zims = sorted(
            f for f in kiwix.iterdir() if f.suffix == ".zim" and f.name not in expected_zims
        )
    else:
        orphaned_zims = []

    if orphaned_zims:
        click.echo("Orphaned ZIM files not in config.yaml:")
        for f in orphaned_zims:
            click.echo(f"  {f.name}")
        if click.confirm(f"Delete {len(orphaned_zims)} ZIM file(s)?", default=False):
            for f in orphaned_zims:
                f.unlink()
                click.echo(f"Deleted {f.name}.")
        else:
            click.echo("Skipping ZIM file removal.")
    else:
        click.echo("No orphaned ZIM files found.")

    # Local images built for another commit, GPU target, ROCm version, or gateway source
    current = set(_images(config).values())
    stale: list[str] = []
    for service in BUILT_SERVICES:
        repo = f"loki-{service}"
        result = subprocess.run(
            ["docker", "images", repo, "--format", "{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        tags = {tag for tag in result.stdout.split() if tag != "<none>"}
        stale += sorted(f"{repo}:{tag}" for tag in tags if f"{repo}:{tag}" not in current)

    if stale:
        click.echo("Old loki images not matching config.yaml:")
        for image in stale:
            click.echo(f"  {image}")
        if click.confirm(f"Remove {len(stale)} image(s)?", default=False):
            for image in stale:
                subprocess.run(["docker", "image", "rm", image], check=False)
                click.echo(f"Removed {image}.")
        else:
            click.echo("Skipping image removal.")
    else:
        click.echo("No old loki images found.")


def main() -> None:
    """Entry point for the loki CLI."""
    cli()
