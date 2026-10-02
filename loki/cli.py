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

LLAMA_IMAGE = "loki-llama"


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


def _llama_image_tag(config: LokiConfig) -> str:
    """Return the image tag Compose builds for the configured GPU target and commit."""
    return f"{config.llama.gpu_targets}-{config.llama.ref}"


def _write_generated_files(config: LokiConfig) -> bool:
    """Write the Caddyfile, ``.env``, and ``models.ini`` derived from ``config``.

    Parameters
    ----------
    config : LokiConfig
        Configuration to render.

    Returns
    -------
    bool
        ``True`` when every file was written, ``False`` when a model preset is
        invalid (the error is printed and ``models.ini`` is left untouched).
    """
    caddyfile_path().write_text(build_caddyfile(config.url))
    env_file_path().write_text(build_env_file(config))

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


def _llama_model_states(port: int) -> dict[str, str]:
    """Return each served model id mapped to its router status.

    Parameters
    ----------
    port : int
        Host port of the llama-server router.

    Returns
    -------
    dict of str to str
        Model id to status (``"loaded"``, ``"unloaded"``, ``"loading"``, ...);
        empty when the listing cannot be read.
    """
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        entries = r.json()["data"] if r.status_code == 200 else []
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return {}
    if not isinstance(entries, list):
        return {}

    states: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        status = entry.get("status")
        if isinstance(status, dict):
            status = status.get("value")
        states[str(entry["id"])] = "unknown" if status is None else str(status)
    return states


def _build_llama_image(pull: bool = False) -> bool:
    """Build the llama-server image through Compose.

    Parameters
    ----------
    pull : bool, optional
        Refresh the ROCm base image before building. Default is ``False``.

    Returns
    -------
    bool
        ``True`` if ``docker compose build`` exited with code 0.
    """
    args = ["build", "--pull", "llama"] if pull else ["build", "llama"]
    return subprocess.run(_compose(*args), check=False).returncode == 0


@click.group()
def cli() -> None:
    """Manage the loki local LLM and offline knowledge server."""


@cli.command()
def setup() -> None:
    """Run the interactive loki setup wizard.

    Walks through config review, system-package installation (aria2,
    avahi-daemon, avahi-utils), Docker installation, an AMD GPU check, and the
    ``LOKI_ROOT`` shell-profile export. Then writes the Caddyfile, ``.env``, and
    ``models.ini``, offers to build the llama-server image, and downloads any
    ZIM files listed in ``config.yaml``.
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
        f"Port configuration written: caddy={ports.caddy}, kiwix={ports.kiwix}, llama={ports.llama}"
    )

    # Build the llama-server image for the configured GPU and commit
    if is_installed("docker"):
        tag = _llama_image_tag(config)
        if click.confirm(
            f"\nBuild the llama-server image {LLAMA_IMAGE}:{tag} now (10-20 minutes)?",
            default=True,
        ):
            if _build_llama_image():
                click.echo("llama-server image built.")
            else:
                click.echo("Warning: llama-server image build failed.", err=True)
        else:
            click.echo("Skipping; `loki start` builds the image if it is missing.")

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
    """Update system packages, Docker Compose images, and the llama-server image.

    Upgrades aria2, avahi-daemon, and avahi-utils via the system package manager,
    pulls the latest published images, rebuilds the llama-server image on a
    refreshed ROCm base for the configured commit, and restarts the Compose
    stack if it is running.
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

    # Locally built llama-server image
    if not _write_generated_files(config):
        return
    click.echo(f"\nRebuilding {LLAMA_IMAGE}:{_llama_image_tag(config)} ...")
    if not _build_llama_image(pull=True):
        click.echo("Warning: llama-server image build failed.", err=True)
        return

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

    click.echo("Starting Docker Compose stack ...")
    subprocess.run(_compose("up", "-d"), check=False)
    click.echo(f"llama-server API: http://{config.url}:{config.ports.llama}/v1")

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

    # llama-server router health
    llama_url = f"http://localhost:{config.ports.llama}/health"
    try:
        r = requests.get(llama_url, timeout=5)
        llama_online = r.status_code == 200
        if llama_online:
            click.echo(f"  llama-server: ONLINE ({llama_url})")
        else:
            click.echo(f"  llama-server: OFFLINE — HTTP {r.status_code} ({llama_url})")
    except requests.exceptions.RequestException as exc:
        llama_online = False
        click.echo(f"  llama-server: OFFLINE — {exc} ({llama_url})")

    # Served models and which are in VRAM
    if llama_online:
        for model_id, state in _llama_model_states(config.ports.llama).items():
            click.echo(f"    {model_id}: {state}")

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
        for name in ("loki-llama", "loki-open-webui", "loki-caddy", "loki-kiwix"):
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
    """Remove ZIM files and llama-server images no longer matching the config."""
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

    # llama-server images built for another commit or GPU target
    result = subprocess.run(
        ["docker", "images", LLAMA_IMAGE, "--format", "{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    current = _llama_image_tag(config)
    stale_tags = sorted({tag for tag in result.stdout.split() if tag != current})

    if stale_tags:
        click.echo(f"Old {LLAMA_IMAGE} images not matching config.yaml:")
        for tag in stale_tags:
            click.echo(f"  {LLAMA_IMAGE}:{tag}")
        if click.confirm(f"Remove {len(stale_tags)} image(s)?", default=False):
            for tag in stale_tags:
                subprocess.run(["docker", "image", "rm", f"{LLAMA_IMAGE}:{tag}"], check=False)
                click.echo(f"Removed {LLAMA_IMAGE}:{tag}.")
        else:
            click.echo("Skipping image removal.")
    else:
        click.echo(f"No old {LLAMA_IMAGE} images found.")


def main() -> None:
    """Entry point for the loki CLI."""
    cli()
