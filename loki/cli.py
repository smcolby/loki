"""CLI entry point for the loki management tool."""

import os
import shutil
import subprocess
from pathlib import Path

import click
import requests

from loki.config import (
    load_config, kiwix_dir, caddyfile_path, build_caddyfile,
    env_file_path, build_env_file, loki_root, avahi_pid_file,
)
from loki.system import (
    detect_package_manager, is_installed, install_packages, PACKAGE_MAP,
    install_docker, install_ollama, is_ollama_binding_configured,
    configure_ollama_binding, detect_shell_profile, loki_root_already_exported,
    add_loki_root_to_profile, get_local_ip, start_avahi_publish, stop_avahi_publish,
)


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


def _parse_ollama_list(output: str) -> list[str]:
    """Parse ``ollama list`` stdout and return the list of installed model names.

    Parameters
    ----------
    output : str
        Raw stdout captured from ``ollama list``.

    Returns
    -------
    list of str
        Model name tags extracted from the output (e.g. ``["llama3:8b"]``).
    """
    lines = output.strip().splitlines()
    if len(lines) < 2:
        return []
    return [line.split()[0] for line in lines[1:] if line.strip()]


def _aria2c_threads() -> int:
    """Return half the number of logical CPU cores, with a minimum of 1."""
    return max(1, (os.cpu_count() or 2) // 2)


def _ollama_warning(port: int) -> str:
    """Return a warning message when the Ollama service is unreachable.

    Parameters
    ----------
    port : int
        The port on which Ollama was expected to be listening.

    Returns
    -------
    str
        Multi-line warning with instructions for resolving connectivity issues.
    """
    return (
        f"Warning: Ollama does not appear to be running at http://localhost:{port}.\n"
        "Ensure the Ollama service is active: sudo systemctl start ollama\n"
        "\n"
        "Ollama binds to 127.0.0.1 by default; Docker containers reach it via\n"
        "host.docker.internal, which requires Ollama to listen on all interfaces.\n"
        "Run `loki setup` to configure the OLLAMA_HOST systemd override automatically.\n"
    )


@click.group()
def cli() -> None:
    """Manage the loki local LLM and offline knowledge server."""


@cli.command()
def setup() -> None:
    """Run the interactive loki setup wizard.

    Walks through six steps in order: config review, system-package installation
    (aria2, avahi-daemon, avahi-utils), Docker installation, Ollama installation,
    Ollama network-binding configuration, and ``LOKI_ROOT`` shell-profile export.
    After all system steps, writes the Caddyfile and ``.env`` port configuration
    and downloads any ZIM files listed in ``config.yaml``.
    """
    config = load_config()

    # Review configuration before proceeding
    config_path = loki_root() / "config.yaml"
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
            if click.confirm(
                f"Install with sudo {manager}?", default=True
            ):
                if not install_packages(missing_pkgs, manager):
                    click.echo(
                        "Warning: package installation failed. "
                        "See README for manual instructions.",
                        err=True,
                    )
                else:
                    click.echo("Packages installed.")
            else:
                click.echo("Skipping — see README for manual instructions.")
    else:
        click.echo(
            "Warning: no supported package manager found (apt-get, dnf). "
            "Install aria2, avahi-daemon, and avahi-utils manually.",
            err=True,
        )

    # Install Docker
    if not is_installed("docker"):
        click.echo("\nDocker is not installed.")
        if click.confirm(
            "Install Docker via the official script with sudo?", default=True
        ):
            if install_docker():
                click.echo("Docker installed.")
                click.echo(
                    "Note: log out and back in (or run `newgrp docker`) "
                    "before using docker without sudo."
                )
            else:
                click.echo(
                    "Warning: Docker installation failed. "
                    "See README for manual instructions.",
                    err=True,
                )
        else:
            click.echo("Skipping — see README for manual instructions.")

    # Install Ollama
    if not is_installed("ollama"):
        click.echo("\nOllama is not installed.")
        if click.confirm("Install Ollama via the official script?", default=True):
            if install_ollama():
                click.echo("Ollama installed.")
            else:
                click.echo(
                    "Warning: Ollama installation failed. "
                    "See README for manual instructions.",
                    err=True,
                )
        else:
            click.echo("Skipping — see README for manual instructions.")

    # Configure Ollama network binding
    if not is_ollama_binding_configured():
        click.echo(
            "\nOllama is not configured to bind to all interfaces "
            "(required for Docker)."
        )
        if click.confirm(
            "Configure Ollama binding with sudo (creates systemd override)?",
            default=True,
        ):
            if configure_ollama_binding():
                click.echo("Ollama binding configured.")
            else:
                click.echo(
                    "Warning: Ollama binding configuration failed. "
                    "See README for manual instructions.",
                    err=True,
                )
        else:
            click.echo("Skipping — see README for manual instructions.")

    # Add LOKI_ROOT to shell profile
    current_root = loki_root()
    if os.environ.get("LOKI_ROOT") != str(current_root):
        profile = detect_shell_profile()
        if not loki_root_already_exported(profile, current_root):
            click.echo(f"\nLOKI_ROOT is not set to {current_root}.")
            if click.confirm(
                f"Add `export LOKI_ROOT={current_root}` to {profile}?", default=True
            ):
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

    click.echo("")

    caddyfile_path().write_text(build_caddyfile(config.url))
    click.echo(f"Caddyfile written for http://{config.url}")

    ports = config.ports
    env_file_path().write_text(build_env_file(ports))
    click.echo(
        f"Port configuration written: caddy={ports.caddy}, "
        f"kiwix={ports.kiwix}, ollama={ports.ollama}"
    )

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
def start() -> None:
    """Pull Ollama models, start the Docker Compose stack, and broadcast mDNS."""
    _require_tool("ollama")
    _require_tool("docker")
    config = load_config()
    ollama_url = f"http://localhost:{config.ports.ollama}/api/tags"

    try:
        response = requests.get(ollama_url, timeout=5)
        ollama_online = response.status_code == 200
    except requests.exceptions.RequestException:
        ollama_online = False

    if not ollama_online:
        click.echo(_ollama_warning(config.ports.ollama), err=True)

    for model in config.ollama_models:
        click.echo(f"Pulling Ollama model: {model}")
        result = subprocess.run(["ollama", "pull", model], check=False)
        if result.returncode != 0:
            click.echo(
                f"Warning: failed to pull {model} (exit code {result.returncode}).",
                err=True,
            )

    click.echo("Starting Docker Compose stack ...")
    subprocess.run(["docker", "compose", "up", "-d"], check=False)

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
        click.echo(
            f"Note: {hostname} does not use .local TLD; skipping mDNS broadcast."
        )


@cli.command()
def stop() -> None:
    """Stop the Docker Compose stack."""
    _require_tool("docker")
    stop_avahi_publish(avahi_pid_file())
    click.echo("Stopping Docker Compose stack ...")
    subprocess.run(["docker", "compose", "down"], check=False)


@cli.command()
def status() -> None:
    """Check the health of running services."""
    config = load_config()
    endpoints = [
        ("Ollama", f"http://localhost:{config.ports.ollama}/api/tags"),
        ("Kiwix", f"http://localhost:{config.ports.kiwix}"),
    ]
    click.echo("Checking service status ...")
    for label, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                click.echo(f"  {label}: ONLINE ({url})")
            else:
                click.echo(f"  {label}: OFFLINE — HTTP {response.status_code} ({url})")
        except requests.exceptions.RequestException as exc:
            click.echo(f"  {label}: OFFLINE — {exc} ({url})")


@cli.command()
def cleanup() -> None:
    """Remove ZIM files and Ollama models no longer listed in config."""
    _require_tool("ollama")
    config = load_config()

    # ZIM files
    kiwix = kiwix_dir()
    if kiwix.exists():
        expected_zims = {Path(entry.url).name for entry in config.kiwix_files}
        orphaned_zims = sorted(
            f for f in kiwix.iterdir()
            if f.suffix == ".zim" and f.name not in expected_zims
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

    # Ollama models
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=False
    )
    installed = _parse_ollama_list(result.stdout)
    orphaned_models = sorted(set(installed) - set(config.ollama_models))

    if orphaned_models:
        click.echo("Orphaned Ollama models not in config.yaml:")
        for model in orphaned_models:
            click.echo(f"  {model}")
        if click.confirm(f"Remove {len(orphaned_models)} Ollama model(s)?", default=False):
            for model in orphaned_models:
                subprocess.run(["ollama", "rm", model], check=False)
                click.echo(f"Removed {model}.")
        else:
            click.echo("Skipping Ollama model removal.")
    else:
        click.echo("No orphaned Ollama models found.")


def main() -> None:
    """Entry point for the loki CLI."""
    cli()
