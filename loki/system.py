"""System-level operations for loki setup on Linux."""

import os
import shutil
import signal
import subprocess
from pathlib import Path

_OLLAMA_OVERRIDE_DIR = Path("/etc/systemd/system/ollama.service.d")
_OLLAMA_OVERRIDE_FILE = _OLLAMA_OVERRIDE_DIR / "override.conf"
_OLLAMA_OVERRIDE_CONTENT = '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n'

# Package names indexed by package manager and the command they provide.
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
    """Return 'apt-get' or 'dnf' if found on PATH, else None."""
    for mgr in ("apt-get", "dnf"):
        if shutil.which(mgr):
            return mgr
    return None


def is_installed(cmd: str) -> bool:
    """Return True if ``cmd`` is found on PATH."""
    return shutil.which(cmd) is not None


def install_packages(pkgs: list[str], manager: str) -> bool:
    """Install ``pkgs`` using ``manager`` with sudo. Returns True on success."""
    result = subprocess.run(["sudo", manager, "install", "-y"] + pkgs, check=False)
    return result.returncode == 0


def install_docker() -> bool:
    """Install Docker via the official convenience script. Returns True on success.

    ``shell=True`` is intentional — the official Docker installer is a shell
    pipeline from a vendor-controlled URL with no user-supplied input.
    After installation, adds the current user to the ``docker`` group.
    """
    result = subprocess.run(
        "curl -fsSL https://get.docker.com | sh", shell=True, check=False  # noqa: S602
    )
    if result.returncode != 0:
        return False
    user = os.environ.get("USER") or os.environ.get("LOGNAME", "")
    if user:
        subprocess.run(["sudo", "usermod", "-aG", "docker", user], check=False)
    return True


def install_ollama() -> bool:
    """Install Ollama via the official install script. Returns True on success.

    ``shell=True`` is intentional — the official Ollama installer is a shell
    pipeline from a vendor-controlled URL with no user-supplied input.
    """
    result = subprocess.run(
        "curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=False  # noqa: S602
    )
    return result.returncode == 0


def is_ollama_binding_configured() -> bool:
    """Return True if the Ollama systemd override already sets OLLAMA_HOST."""
    try:
        return "OLLAMA_HOST=0.0.0.0" in _OLLAMA_OVERRIDE_FILE.read_text()
    except (FileNotFoundError, PermissionError):
        return False


def configure_ollama_binding() -> bool:
    """Create the Ollama systemd override and restart the service.

    Uses ``sudo mkdir`` and ``sudo tee`` to write the override file without
    requiring a shell redirect. Returns True on success.
    """
    result = subprocess.run(
        ["sudo", "mkdir", "-p", str(_OLLAMA_OVERRIDE_DIR)], check=False
    )
    if result.returncode != 0:
        return False
    result = subprocess.run(
        ["sudo", "tee", str(_OLLAMA_OVERRIDE_FILE)],
        input=_OLLAMA_OVERRIDE_CONTENT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=False)
    result = subprocess.run(["sudo", "systemctl", "restart", "ollama"], check=False)
    return result.returncode == 0


def detect_shell_profile() -> Path:
    """Return the user's shell profile path inferred from ``$SHELL``.

    Prefers ``~/.zshrc`` for zsh, ``~/.bashrc`` for bash, and falls back to
    ``~/.profile`` for any other shell.
    """
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "bash" in shell:
        return home / ".bashrc"
    return home / ".profile"


def loki_root_already_exported(profile: Path, root: Path) -> bool:
    """Return True if the exact LOKI_ROOT export line is already in ``profile``."""
    export_line = f"export LOKI_ROOT={root}"
    try:
        return export_line in profile.read_text()
    except (FileNotFoundError, PermissionError):
        return False


def add_loki_root_to_profile(profile: Path, root: Path) -> bool:
    """Append ``export LOKI_ROOT=<root>`` to ``profile``. Returns True on success."""
    export_line = f"\nexport LOKI_ROOT={root}\n"
    try:
        with open(profile, "a") as f:
            f.write(export_line)
        return True
    except OSError:
        return False


def get_local_ip() -> str:
    """Return the first IP address from ``hostname -I``, or empty string on failure."""
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, check=False
        )
        ips = result.stdout.strip().split()
        return ips[0] if ips else ""
    except OSError:
        return ""


def start_avahi_publish(hostname: str, ip: str, pid_file: Path) -> None:
    """Spawn ``avahi-publish-address`` in the background and save its PID.

    Kills any existing process from a stale PID file before spawning a new one.
    stdout and stderr are redirected to ``/dev/null`` to suppress terminal output.

    Note: ``avahi-publish-address -R <hostname> <ip>`` publishes an A record
    (hostname → IP), distinct from ``avahi-publish-host-name`` which registers
    the machine's own system hostname. Using address publication keeps the
    announced name fully controlled by ``config.url``.
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
    """Send SIGTERM to the avahi-publish process if it is still alive.

    Checks liveness with ``os.kill(pid, 0)`` before sending the signal.
    Handles stale or missing PID files gracefully.
    """
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # Raises ProcessLookupError if process is dead.
        os.kill(pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pass
    finally:
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
