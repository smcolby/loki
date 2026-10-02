"""Tests for the cleanup subcommand — orphaned ZIM files and old llama-server images."""

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from loki.cli import cli


def _images(*tags: str) -> subprocess.CompletedProcess:
    """Return ``docker images loki-llama --format {{.Tag}}`` output listing ``tags``."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="".join(f"{t}\n" for t in tags)
    )


@pytest.fixture(autouse=True)
def _config(mocker, sample_config, tmp_path):
    """Serve the sample config and an empty kiwix directory to every cleanup test."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)


# --- ZIM file cleanup tests ---


def test_cleanup_no_orphaned_zims_prints_message(mocker):
    """Cleanup prints a message when there are no orphaned ZIM files on disk."""
    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No orphaned ZIM files found." in result.output


def test_cleanup_skips_configured_zim_file(sample_config, tmp_path):
    """Cleanup does not treat a ZIM file that matches a config entry as an orphan."""
    (tmp_path / Path(sample_config.kiwix_files[0].url).name).touch()

    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No orphaned ZIM files found." in result.output


def test_cleanup_lists_orphaned_zim_file(tmp_path):
    """Cleanup lists the names of ZIM files on disk that are not in the config."""
    (tmp_path / "old_encyclopedia.zim").touch()

    result = CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert "old_encyclopedia.zim" in result.output


def test_cleanup_deletes_orphaned_zim_on_confirm(tmp_path):
    """Cleanup removes an orphaned ZIM file from disk when the user confirms."""
    orphan = tmp_path / "old_encyclopedia.zim"
    orphan.touch()

    CliRunner().invoke(cli, ["cleanup"], input="y\n")

    assert not orphan.exists()


def test_cleanup_preserves_orphaned_zim_on_deny(tmp_path):
    """Cleanup leaves an orphaned ZIM file on disk when the user denies."""
    orphan = tmp_path / "old_encyclopedia.zim"
    orphan.touch()

    CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert orphan.exists()


def test_cleanup_handles_missing_kiwix_dir(mocker, tmp_path):
    """Cleanup handles the case where the kiwix data directory does not exist."""
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "nonexistent")

    result = CliRunner().invoke(cli, ["cleanup"])

    assert result.exit_code == 0
    assert "No orphaned ZIM files found." in result.output


# --- llama-server image cleanup tests ---


def test_cleanup_keeps_current_image(mocker):
    """Cleanup reports nothing to remove when only the configured image exists."""
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=_images("gfx1100-abc123"))

    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No old loki-llama images found." in result.output


def test_cleanup_lists_images_for_other_commits(mocker):
    """Cleanup lists images built for a commit or target the config no longer names."""
    mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        return_value=_images("gfx1100-abc123", "gfx1100-old999", "gfx1201-abc123"),
    )

    result = CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert "loki-llama:gfx1100-old999" in result.output
    assert "loki-llama:gfx1201-abc123" in result.output
    assert "loki-llama:gfx1100-abc123" not in result.output


def test_cleanup_removes_old_images_on_confirm(mocker):
    """Cleanup runs `docker image rm` for each old image when the user confirms."""
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=[_images("gfx1100-abc123", "gfx1100-old999"), _images()],
    )

    CliRunner().invoke(cli, ["cleanup"], input="y\n")

    mock_run.assert_any_call(["docker", "image", "rm", "loki-llama:gfx1100-old999"], check=False)


def test_cleanup_skips_image_removal_on_deny(mocker):
    """Cleanup does not remove images when the user denies the prompt."""
    mock_run = mocker.patch(
        "loki.cli.subprocess.run", autospec=True, return_value=_images("gfx1100-old999")
    )

    CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert all(call.args[0][:3] != ["docker", "image", "rm"] for call in mock_run.call_args_list)


def test_cleanup_exits_when_docker_not_found(mocker):
    """Cleanup exits with an error message when docker is not on PATH."""
    mocker.patch("loki.cli.shutil.which", return_value=None)

    result = CliRunner().invoke(cli, ["cleanup"])

    assert result.exit_code != 0
    assert "docker" in result.output
