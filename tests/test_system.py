"""Tests for loki.system — OS-level operations for loki setup on Linux."""

import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import loki.system as system
from loki.system import (
    PACKAGE_MAP,
    add_loki_root_to_profile,
    configure_ollama_binding,
    detect_package_manager,
    detect_shell_profile,
    get_local_ip,
    install_docker,
    install_ollama,
    install_packages,
    is_installed,
    is_ollama_binding_configured,
    loki_root_already_exported,
    start_avahi_publish,
    stop_avahi_publish,
)

# ---------------------------------------------------------------------------
# detect_package_manager
# ---------------------------------------------------------------------------


def test_detect_package_manager_returns_apt_get(mocker):
    """Returns 'apt-get' when apt-get is on PATH."""
    mocker.patch("loki.system.shutil.which", side_effect=lambda cmd: cmd == "apt-get")
    assert detect_package_manager() == "apt-get"


def test_detect_package_manager_returns_dnf(mocker):
    """Returns 'dnf' when dnf is on PATH but apt-get is not."""
    mocker.patch("loki.system.shutil.which", side_effect=lambda cmd: cmd == "dnf")
    assert detect_package_manager() == "dnf"


def test_detect_package_manager_prefers_apt_over_dnf(mocker):
    """Returns 'apt-get' when both apt-get and dnf are available."""
    mocker.patch("loki.system.shutil.which", return_value="/usr/bin/stub")
    assert detect_package_manager() == "apt-get"


def test_detect_package_manager_returns_none_when_none_found(mocker):
    """Returns None when neither apt-get nor dnf is on PATH."""
    mocker.patch("loki.system.shutil.which", return_value=None)
    assert detect_package_manager() is None


# ---------------------------------------------------------------------------
# is_installed
# ---------------------------------------------------------------------------


def test_is_installed_returns_true_when_found(mocker):
    """Returns True when the command is found on PATH."""
    mocker.patch("loki.system.shutil.which", return_value="/usr/bin/aria2c")
    assert is_installed("aria2c") is True


def test_is_installed_returns_false_when_not_found(mocker):
    """Returns False when the command is not on PATH."""
    mocker.patch("loki.system.shutil.which", return_value=None)
    assert is_installed("aria2c") is False


# ---------------------------------------------------------------------------
# PACKAGE_MAP
# ---------------------------------------------------------------------------


def test_package_map_contains_apt_and_dnf():
    """PACKAGE_MAP covers apt-get and dnf managers."""
    assert "apt-get" in PACKAGE_MAP
    assert "dnf" in PACKAGE_MAP


def test_package_map_apt_maps_aria2c():
    """apt-get map resolves aria2c to the 'aria2' package name."""
    assert PACKAGE_MAP["apt-get"]["aria2c"] == "aria2"


def test_package_map_dnf_maps_avahi_daemon():
    """Dnf map resolves avahi-daemon to 'avahi'."""
    assert PACKAGE_MAP["dnf"]["avahi-daemon"] == "avahi"


# ---------------------------------------------------------------------------
# install_packages
# ---------------------------------------------------------------------------


def test_install_packages_runs_correct_command(mocker):
    """install_packages calls sudo apt-get install -y with the given packages."""
    mock_run = mocker.patch(
        "loki.system.subprocess.run", autospec=True, return_value=MagicMock(returncode=0)
    )
    install_packages(["aria2", "avahi-daemon"], "apt-get")
    mock_run.assert_called_once_with(
        ["sudo", "apt-get", "install", "-y", "aria2", "avahi-daemon"], check=False
    )


def test_install_packages_returns_true_on_success(mocker):
    """install_packages returns True when subprocess exits with 0."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
    )
    assert install_packages(["aria2"], "apt-get") is True


def test_install_packages_returns_false_on_failure(mocker):
    """install_packages returns False when subprocess exits non-zero."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=1),
    )
    assert install_packages(["aria2"], "apt-get") is False


# ---------------------------------------------------------------------------
# install_docker
# ---------------------------------------------------------------------------


def test_install_docker_runs_convenience_script(mocker):
    """install_docker executes the official Docker convenience script."""
    mock_run = mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
    )
    mocker.patch.dict(os.environ, {"USER": "testuser"})
    install_docker()
    first_call = mock_run.call_args_list[0]
    assert "get.docker.com" in first_call[0][0]
    assert first_call[1].get("shell") is True


def test_install_docker_adds_user_to_docker_group(mocker):
    """install_docker runs usermod to add the current user to the docker group."""
    mock_run = mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
    )
    mocker.patch.dict(os.environ, {"USER": "testuser"})
    install_docker()
    calls_args = [call[0][0] for call in mock_run.call_args_list]
    assert any("usermod" in str(args) for args in calls_args)


def test_install_docker_returns_false_on_failure(mocker):
    """install_docker returns False when the script exits non-zero."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=1),
    )
    assert install_docker() is False


# ---------------------------------------------------------------------------
# install_ollama
# ---------------------------------------------------------------------------


def test_install_ollama_runs_official_script(mocker):
    """install_ollama executes the official Ollama install script."""
    mock_run = mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
    )
    install_ollama()
    call_args = mock_run.call_args
    assert "ollama.com/install.sh" in call_args[0][0]
    assert call_args[1].get("shell") is True


def test_install_ollama_returns_true_on_success(mocker):
    """install_ollama returns True when the script exits with 0."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
    )
    assert install_ollama() is True


def test_install_ollama_returns_false_on_failure(mocker):
    """install_ollama returns False when the script exits non-zero."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=1),
    )
    assert install_ollama() is False


# ---------------------------------------------------------------------------
# is_ollama_binding_configured
# ---------------------------------------------------------------------------


def test_is_ollama_binding_configured_true_when_override_exists(tmp_path, mocker):
    """Returns True when the override file contains OLLAMA_HOST=0.0.0.0."""
    override_file = tmp_path / "override.conf"
    override_file.write_text('[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n')
    mocker.patch.object(system, "_OLLAMA_OVERRIDE_FILE", override_file)
    assert is_ollama_binding_configured() is True


def test_is_ollama_binding_configured_false_when_file_missing(tmp_path, mocker):
    """Returns False when the override file does not exist."""
    mocker.patch.object(system, "_OLLAMA_OVERRIDE_FILE", tmp_path / "missing.conf")
    assert is_ollama_binding_configured() is False


def test_is_ollama_binding_configured_false_when_content_differs(tmp_path, mocker):
    """Returns False when the override file exists but does not set OLLAMA_HOST."""
    override_file = tmp_path / "override.conf"
    override_file.write_text("[Service]\nEnvironment=OTHER=value\n")
    mocker.patch.object(system, "_OLLAMA_OVERRIDE_FILE", override_file)
    assert is_ollama_binding_configured() is False


# ---------------------------------------------------------------------------
# configure_ollama_binding
# ---------------------------------------------------------------------------


def test_configure_ollama_binding_creates_override_and_restarts(mocker):
    """configure_ollama_binding creates the override dir, writes the file, and restarts ollama."""
    mock_run = mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
    )
    configure_ollama_binding()
    calls_args = [call[0][0] for call in mock_run.call_args_list]
    assert any("mkdir" in str(args) for args in calls_args)
    assert any("tee" in str(args) for args in calls_args)
    assert any("systemctl" in str(args) and "restart" in str(args) for args in calls_args)


def test_configure_ollama_binding_returns_false_on_mkdir_failure(mocker):
    """configure_ollama_binding returns False when mkdir fails."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=1),
    )
    assert configure_ollama_binding() is False


def test_configure_ollama_binding_returns_false_on_daemon_reload_failure(mocker):
    """configure_ollama_binding returns False when daemon-reload fails."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        side_effect=[
            MagicMock(spec=subprocess.CompletedProcess, returncode=0),  # mkdir
            MagicMock(spec=subprocess.CompletedProcess, returncode=0),  # tee
            MagicMock(spec=subprocess.CompletedProcess, returncode=1),  # daemon-reload
        ],
    )
    assert configure_ollama_binding() is False


# ---------------------------------------------------------------------------
# detect_shell_profile
# ---------------------------------------------------------------------------


def test_detect_shell_profile_zsh(mocker):
    """Returns ~/.zshrc for zsh users."""
    mocker.patch.dict(os.environ, {"SHELL": "/bin/zsh"})
    assert detect_shell_profile() == Path.home() / ".zshrc"


def test_detect_shell_profile_bash(mocker):
    """Returns ~/.bashrc for bash users."""
    mocker.patch.dict(os.environ, {"SHELL": "/bin/bash"})
    assert detect_shell_profile() == Path.home() / ".bashrc"


def test_detect_shell_profile_fallback(mocker):
    """Returns ~/.profile for unknown shells."""
    mocker.patch.dict(os.environ, {"SHELL": "/bin/sh"})
    assert detect_shell_profile() == Path.home() / ".profile"


# ---------------------------------------------------------------------------
# loki_root_already_exported
# ---------------------------------------------------------------------------


def test_loki_root_already_exported_true_when_present(tmp_path):
    """Returns True when the export line is already in the profile."""
    root = Path("/srv/loki")
    profile = tmp_path / ".bashrc"
    profile.write_text(f"export LOKI_ROOT={root}\n")
    assert loki_root_already_exported(profile, root) is True


def test_loki_root_already_exported_false_when_absent(tmp_path):
    """Returns False when the export line is not in the profile."""
    profile = tmp_path / ".bashrc"
    profile.write_text("# other config\n")
    assert loki_root_already_exported(profile, Path("/srv/loki")) is False


def test_loki_root_already_exported_false_when_file_missing(tmp_path):
    """Returns False when the profile file does not exist."""
    assert loki_root_already_exported(tmp_path / ".bashrc", Path("/srv/loki")) is False


# ---------------------------------------------------------------------------
# add_loki_root_to_profile
# ---------------------------------------------------------------------------


def test_add_loki_root_to_profile_appends_export_line(tmp_path):
    """Appends the export LOKI_ROOT line to the profile file."""
    profile = tmp_path / ".bashrc"
    profile.write_text("# existing config\n")
    root = Path("/srv/loki")
    add_loki_root_to_profile(profile, root)
    assert f"export LOKI_ROOT={root}" in profile.read_text()


def test_add_loki_root_to_profile_returns_true_on_success(tmp_path):
    """Returns True when the write succeeds."""
    profile = tmp_path / ".bashrc"
    profile.write_text("")
    assert add_loki_root_to_profile(profile, Path("/srv/loki")) is True


def test_add_loki_root_to_profile_returns_false_on_error(tmp_path):
    """Returns False when the file cannot be written."""
    profile = tmp_path / "readonly.bashrc"
    profile.write_text("")
    profile.chmod(0o444)
    result = add_loki_root_to_profile(profile, Path("/srv/loki"))
    assert result is False
    profile.chmod(0o644)  # restore for cleanup


# ---------------------------------------------------------------------------
# get_local_ip
# ---------------------------------------------------------------------------


def test_get_local_ip_returns_first_ip(mocker):
    """Returns the first IP address from hostname -I output."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, stdout="192.168.1.5 10.0.0.1 "),
    )
    assert get_local_ip() == "192.168.1.5"


def test_get_local_ip_returns_empty_on_blank_output(mocker):
    """Returns an empty string when hostname -I produces no output."""
    mocker.patch(
        "loki.system.subprocess.run",
        autospec=True,
        return_value=MagicMock(spec=subprocess.CompletedProcess, stdout=""),
    )
    assert get_local_ip() == ""


def test_get_local_ip_returns_empty_on_oserror(mocker):
    """Returns an empty string when the subprocess raises OSError."""
    mocker.patch("loki.system.subprocess.run", side_effect=OSError)
    assert get_local_ip() == ""


# ---------------------------------------------------------------------------
# start_avahi_publish
# ---------------------------------------------------------------------------


def test_start_avahi_publish_spawns_process(mocker, tmp_path):
    """start_avahi_publish spawns avahi-publish-address with correct arguments."""
    mocker.patch("loki.system.stop_avahi_publish")
    mock_popen = mocker.patch("loki.system.subprocess.Popen", return_value=MagicMock(pid=12345))
    pid_file = tmp_path / ".avahi.pid"

    start_avahi_publish("loki.local", "192.168.1.5", pid_file)

    mock_popen.assert_called_once_with(
        ["avahi-publish-address", "-R", "loki.local", "192.168.1.5"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_start_avahi_publish_writes_pid_file(mocker, tmp_path):
    """start_avahi_publish saves the spawned process PID to the pid file."""
    mocker.patch("loki.system.stop_avahi_publish")
    mocker.patch("loki.system.subprocess.Popen", return_value=MagicMock(pid=12345))
    pid_file = tmp_path / ".avahi.pid"

    start_avahi_publish("loki.local", "192.168.1.5", pid_file)

    assert pid_file.read_text() == "12345"


def test_start_avahi_publish_kills_stale_process_first(mocker, tmp_path):
    """start_avahi_publish calls stop_avahi_publish before spawning a new process."""
    mock_stop = mocker.patch("loki.system.stop_avahi_publish")
    mocker.patch("loki.system.subprocess.Popen", return_value=MagicMock(pid=99))
    pid_file = tmp_path / ".avahi.pid"

    start_avahi_publish("loki.local", "192.168.1.5", pid_file)

    mock_stop.assert_called_once_with(pid_file)


# ---------------------------------------------------------------------------
# stop_avahi_publish
# ---------------------------------------------------------------------------


def test_stop_avahi_publish_sends_sigterm(mocker, tmp_path):
    """stop_avahi_publish sends SIGTERM to the process recorded in the pid file."""
    pid_file = tmp_path / ".avahi.pid"
    pid_file.write_text("9999")
    mock_kill = mocker.patch("loki.system.os.kill")

    stop_avahi_publish(pid_file)

    mock_kill.assert_any_call(9999, signal.SIGTERM)


def test_stop_avahi_publish_removes_pid_file(mocker, tmp_path):
    """stop_avahi_publish deletes the pid file after sending the signal."""
    pid_file = tmp_path / ".avahi.pid"
    pid_file.write_text("9999")
    mocker.patch("loki.system.os.kill")

    stop_avahi_publish(pid_file)

    assert not pid_file.exists()


def test_stop_avahi_publish_noop_when_pid_file_missing(tmp_path):
    """stop_avahi_publish does nothing when the pid file does not exist."""
    stop_avahi_publish(tmp_path / ".avahi.pid")  # Should not raise.


def test_stop_avahi_publish_handles_stale_pid(mocker, tmp_path):
    """stop_avahi_publish handles a stale PID gracefully (process already dead)."""
    pid_file = tmp_path / ".avahi.pid"
    pid_file.write_text("9999")
    mocker.patch("loki.system.os.kill", side_effect=ProcessLookupError)

    stop_avahi_publish(pid_file)  # Should not raise.
    assert not pid_file.exists()
