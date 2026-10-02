"""Shared pytest fixtures for the loki test suite."""

import subprocess
from pathlib import Path

import pytest

from loki.config import GATEWAY_SOURCES, KiwixFile, LlamaConfig, LokiConfig, PortsConfig


@pytest.fixture
def sample_config(tmp_path) -> LokiConfig:
    """Return a synthetic LokiConfig instance for use across tests."""
    return LokiConfig(
        url="loki.local",
        ports=PortsConfig(caddy=80, kiwix=8080, api=8090),
        llama=LlamaConfig(models_dir=tmp_path / "models", ref="abc123", gpu_targets="gfx1100"),
        kiwix_files=[
            KiwixFile(
                name="wikipedia_en_test",
                url="https://download.kiwix.org/zim/wikipedia/wikipedia_en_test_2024-01.zim",
            )
        ],
    )


@pytest.fixture(autouse=True)
def _isolate_paths(mocker, monkeypatch, tmp_path):
    """Keep generated files and the default models directory inside ``tmp_path``.

    ``HOME`` points at an empty directory so a default ``~/.llms`` never reads
    the developer's real models, and the three generated files land in
    ``tmp_path``. Tests can re-patch any of these paths.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mocker.patch("loki.cli.caddyfile_path", return_value=tmp_path / "Caddyfile")
    mocker.patch("loki.cli.env_file_path", return_value=tmp_path / ".env")
    mocker.patch("loki.cli.models_preset_path", return_value=tmp_path / "models.ini")


@pytest.fixture(autouse=True)
def gateway_sources(mocker, tmp_path) -> Path:
    """Give the gateway image tag fixed build inputs under ``tmp_path``."""
    source_dir = tmp_path / "gateway"
    source_dir.mkdir()
    for name in GATEWAY_SOURCES:
        (source_dir / name).write_text(f"{name}\n")
    mocker.patch("loki.cli.gateway_dir", return_value=source_dir)
    return source_dir


@pytest.fixture(autouse=True)
def missing_services(mocker):
    """Report every local image as built; image tests set ``return_value`` or re-patch."""
    return mocker.patch("loki.cli._missing_services", return_value=[])


@pytest.fixture(autouse=True)
def _stub_subprocess(mocker):
    """Stop any test from launching a real docker, aria2c, or package-manager process.

    Tests that inspect commands re-patch ``loki.cli.subprocess.run`` themselves.
    """

    def _succeed(args, *_, **__):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    mocker.patch("loki.cli.subprocess.run", new=_succeed)


@pytest.fixture(autouse=True)
def _stub_shutil_which(mocker):
    """Prevent _require_tool from failing when external tools are not on PATH.

    All CLI command tests mock subprocess.run, so external tools never execute.
    This stub ensures shutil.which returns a truthy value so the pre-flight
    check passes in every test by default. Tests that specifically exercise
    _require_tool can override this by re-patching shutil.which to None.
    """
    mocker.patch("loki.cli.shutil.which", return_value="/usr/bin/stub")


@pytest.fixture(autouse=True)
def _stub_system(mocker):
    """Stub loki.system functions so setup tests run without real system changes.

    Functions are patched at their usage site (loki.cli.*) since cli.py imports
    them with ``from loki.system import ...``. Individual tests can override
    these stubs to exercise specific prompt branches.
    """
    mocker.patch("loki.cli.is_installed", return_value=True)
    mocker.patch("loki.cli.detect_package_manager", return_value="apt-get")
    mocker.patch("loki.cli.amd_gpu_present", return_value=True)
    mocker.patch("loki.cli.detect_shell_profile", return_value=Path.home() / ".bashrc")
    mocker.patch("loki.cli.loki_root_already_exported", return_value=True)
    mocker.patch("loki.cli.get_local_ip", return_value="192.168.1.100")
    mocker.patch("loki.cli.start_avahi_publish")
    mocker.patch("loki.cli.stop_avahi_publish")


@pytest.fixture
def write_preset():
    """Return a helper that creates ``<models_dir>/<name>/`` with a GGUF and a preset.ini."""

    def _write(models_dir: Path, name: str, body: str | None = None) -> Path:
        model_dir = models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{name}.gguf").touch()
        text = body if body is not None else f"[{name}]\nmodel = {name}.gguf\n"
        preset = model_dir / "preset.ini"
        preset.write_text(text)
        return preset

    return _write
