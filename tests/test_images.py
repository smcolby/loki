"""Tests for local image detection and builds.

``_missing_services`` is imported at module load, before the autouse fixture
replaces ``loki.cli._missing_services`` with a stub.
"""

import subprocess

from loki.cli import _build_services, _missing_services
from loki.config import ImageConfig, ImageEngineConfig, gateway_tag

LLAMA = "loki-llama:gfx1100-rocm7.2.4-abc123"
STRATA = "loki-strata:gfx1100-rocm7.10.0a20251120-6f32ec070f23ced9f50e704d854d775da52591ab"


def _inspect(mocker, present: set[str]):
    """Patch ``subprocess.run`` so ``docker image inspect`` succeeds only for ``present``."""

    def _run(args, *_, **__):
        found = args[:3] == ["docker", "image", "inspect"] and args[3] in present
        return subprocess.CompletedProcess(args=args, returncode=0 if found else 1)

    return mocker.patch("loki.cli.subprocess.run", autospec=True, side_effect=_run)


def test_missing_services_lists_absent_images(mocker, sample_config):
    """Only services whose configured image is absent are reported, gateway first."""
    _inspect(mocker, {LLAMA})

    assert _missing_services(sample_config) == ["gateway"]


def test_missing_services_empty_when_all_present(mocker, sample_config, gateway_sources):
    """Nothing is missing when every configured image exists."""
    _inspect(mocker, {LLAMA, f"loki-gateway:{gateway_tag(gateway_sources)}"})

    assert _missing_services(sample_config) == []


def test_missing_services_includes_strata_only_with_engines(
    mocker, sample_config, add_strata_engines
):
    """Strata's image is checked only when Strata has engines, and once for all of them."""
    mock_run = _inspect(mocker, set())

    assert _missing_services(sample_config) == ["gateway", "llama"]
    assert all(call.args[0][3] != STRATA for call in mock_run.call_args_list)

    add_strata_engines(sample_config, "qwen", "swift")
    assert _missing_services(sample_config) == ["gateway", "llama", "strata"]


def test_missing_services_follows_gateway_source(mocker, sample_config, gateway_sources):
    """Editing a gateway build input names a new gateway image, which is then missing."""
    _inspect(mocker, {LLAMA, f"loki-gateway:{gateway_tag(gateway_sources)}"})
    (gateway_sources / "gateway.py").write_text("changed\n")

    assert _missing_services(sample_config) == ["gateway"]


def test_build_services_builds_strata_through_its_first_engine(
    mocker, sample_config, add_strata_engines
):
    """The shared Strata image is built once, through the first engine's service."""
    add_strata_engines(sample_config, "qwen", "swift")
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        return_value=subprocess.CompletedProcess(args=[], returncode=0),
    )

    assert _build_services(sample_config, ["gateway", "strata"]) is True
    assert mock_run.call_args.args[0][-3:] == ["build", "gateway", "strata-qwen"]


def test_build_services_builds_image_through_its_first_engine(mocker, sample_config):
    """The shared sd-server image is built once, through the first image engine's service."""
    sample_config.image = ImageConfig(
        engines={name: ImageEngineConfig(model=name) for name in ("qwen", "flux")}
    )
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        return_value=subprocess.CompletedProcess(args=[], returncode=0),
    )

    assert _build_services(sample_config, ["image"]) is True
    assert mock_run.call_args.args[0][-2:] == ["build", "image-qwen"]


def test_missing_services_includes_image_only_with_engines(mocker, sample_config):
    """The sd-server image is checked only when image engines are configured."""
    _inspect(mocker, set())

    assert "image" not in _missing_services(sample_config)

    sample_config.image = ImageConfig(engines={"qwen": ImageEngineConfig(model="q")})
    assert _missing_services(sample_config) == ["gateway", "llama", "image"]


def test_build_services_reports_failure(mocker, sample_config):
    """A non-zero build exit is reported as failure."""
    mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        return_value=subprocess.CompletedProcess(args=[], returncode=1),
    )

    assert _build_services(sample_config, ["llama"]) is False
