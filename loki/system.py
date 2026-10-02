"""System-level operations for loki setup on Linux."""

import os
import shutil
import signal
import subprocess
from pathlib import Path

# Device nodes the ROCm runtime opens inside the llama-server container
_AMD_GPU_DEVICES = (Path("/dev/kfd"), Path("/dev/dri"))

# Package names indexed by package manager and the command they provide
PACKAGE_MAP: dict[str, dict[str, str]] = {
    "apt-get": {
        "aria2c": "aria2",
        "avahi-daemon": "avahi-daemon",
        "avahi-publish-address": "avahi-utils",
    },
    "dnf": {
        "aria2c": "aria2",
        "avahi-daemon": "avahi",
        "avahi-publish-address": "avahi-tools",
    },
}


def detect_package_manager() -> str | None:
    """Return the first supported package manager found on PATH.

    Returns
    -------
    str or None
        ``'apt-get'`` or ``'dnf'`` if found, otherwise ``None``.
    """
    for mgr in ("apt-get", "dnf"):
        if shutil.which(mgr):
            return mgr
    return None


def is_installed(cmd: str) -> bool:
    """Return whether ``cmd`` is available on PATH.

    Parameters
    ----------
    cmd : str
        The command name to look up (e.g. ``"docker"``).

    Returns
    -------
    bool
        ``True`` if ``cmd`` resolves to an executable on PATH, ``False`` otherwise.
    """
    return shutil.which(cmd) is not None


def install_packages(pkgs: list[str], manager: str) -> bool:
    """Install ``pkgs`` using ``manager`` under sudo.

    Parameters
    ----------
    pkgs : list of str
        Package names to install (e.g. ``["aria2", "avahi-utils"]``).
    manager : str
        The package manager to use (e.g. ``"apt-get"`` or ``"dnf"``).

    Returns
    -------
    bool
        ``True`` if the installation command exited with code 0, ``False`` otherwise.
    """
    result = subprocess.run(["sudo", manager, "install", "-y"] + pkgs, check=False)
    return result.returncode == 0


def upgrade_packages(pkgs: list[str], manager: str) -> bool:
    """Upgrade ``pkgs`` using ``manager`` under sudo.

    Parameters
    ----------
    pkgs : list of str
        Package names to upgrade (e.g. ``["aria2", "avahi-utils"]``).
    manager : str
        The package manager to use (e.g. ``"apt-get"`` or ``"dnf"``).

    Returns
    -------
    bool
        ``True`` if the upgrade command exited with code 0, ``False`` otherwise.
    """
    if manager == "apt-get":
        result = subprocess.run(
            ["sudo", "apt-get", "install", "--only-upgrade", "-y"] + pkgs, check=False
        )
    else:
        result = subprocess.run(["sudo", manager, "upgrade", "-y"] + pkgs, check=False)
    return result.returncode == 0


def install_docker() -> bool:
    """Install Docker via the official convenience script.

    Uses ``shell=True`` intentionally — the official Docker installer is a shell
    pipeline from a vendor-controlled URL with no user-supplied input.
    After installation, adds the current user to the ``docker`` group.

    Returns
    -------
    bool
        ``True`` if the install script exited with code 0, ``False`` otherwise.
    """
    result = subprocess.run(  # noqa: S602
        "curl -fsSL https://get.docker.com | sh",
        shell=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    user = os.environ.get("USER") or os.environ.get("LOGNAME", "")
    if user:
        subprocess.run(["sudo", "usermod", "-aG", "docker", user], check=False)
    return True


def amd_gpu_present() -> bool:
    """Return whether the amdgpu kernel driver exposes the devices ROCm needs.

    Returns
    -------
    bool
        ``True`` if both ``/dev/kfd`` and ``/dev/dri`` exist, ``False`` otherwise.
    """
    return all(device.exists() for device in _AMD_GPU_DEVICES)


def detect_shell_profile() -> Path:
    """Return the user's shell profile path inferred from ``$SHELL``.

    Prefers ``~/.zshrc`` for zsh, ``~/.bashrc`` for bash, and falls back to
    ``~/.profile`` for any other shell.

    Returns
    -------
    Path
        Absolute path to the shell profile file for the current user.
    """
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "bash" in shell:
        return home / ".bashrc"
    return home / ".profile"


def loki_root_already_exported(profile: Path, root: Path) -> bool:
    """Return whether the exact ``LOKI_ROOT`` export line is already in ``profile``.

    Parameters
    ----------
    profile : Path
        Path to the shell profile file to inspect.
    root : Path
        The ``LOKI_ROOT`` value whose export line to search for.

    Returns
    -------
    bool
        ``True`` if ``export LOKI_ROOT=<root>`` is found in ``profile``,
        ``False`` otherwise (including on missing or unreadable files).
    """
    export_line = f"export LOKI_ROOT={root}"
    try:
        return export_line in profile.read_text()
    except (FileNotFoundError, PermissionError):
        return False


def add_loki_root_to_profile(profile: Path, root: Path) -> bool:
    """Append ``export LOKI_ROOT=<root>`` to ``profile``.

    Parameters
    ----------
    profile : Path
        Path to the shell profile file to append to.
    root : Path
        The directory path to export as ``LOKI_ROOT``.

    Returns
    -------
    bool
        ``True`` if the line was written successfully, ``False`` on any
        ``OSError`` (e.g., permission denied).
    """
    export_line = f"\nexport LOKI_ROOT={root}\n"
    try:
        with open(profile, "a") as f:
            f.write(export_line)
        return True
    except OSError:
        return False


def get_local_ip() -> str:
    """Return the first IP address reported by ``hostname -I``.

    Returns
    -------
    str
        The primary local IP address, or an empty string if ``hostname -I``
        produces no output or raises an ``OSError``.
    """
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=False)
        ips = result.stdout.strip().split()
        return ips[0] if ips else ""
    except OSError:
        return ""


def start_avahi_publish(hostname: str, ip: str, pid_file: Path) -> None:
    """Spawn ``avahi-publish-address`` in the background and record its PID.

    Kills any existing process from a stale PID file before spawning a new one.
    stdout and stderr are redirected to ``/dev/null`` to suppress terminal output.

    Parameters
    ----------
    hostname : str
        The mDNS hostname to advertise (e.g. ``"loki.local"``).
    ip : str
        The IP address to associate with ``hostname``.
    pid_file : Path
        File path where the spawned process PID is written.

    Notes
    -----
    Uses ``avahi-publish-address -R <hostname> <ip>``, which publishes a DNS A
    record (hostname → IP). This is distinct from ``avahi-publish-host-name``,
    which registers the machine's own system hostname. Using address publication
    keeps the announced name fully controlled by ``config.url``.
    """
    stop_avahi_publish(pid_file)
    proc = subprocess.Popen(
        ["avahi-publish-address", "-R", hostname, ip],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid))


def stop_avahi_publish(pid_file: Path) -> None:
    """Send SIGTERM to the avahi-publish process recorded in ``pid_file``.

    Checks liveness with ``os.kill(pid, 0)`` before sending the signal.
    Handles stale or missing PID files gracefully without raising.

    Parameters
    ----------
    pid_file : Path
        Path to the PID file written by ``start_avahi_publish``.
    """
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # Raises ProcessLookupError if process is dead
        os.kill(pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pass
    finally:
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
