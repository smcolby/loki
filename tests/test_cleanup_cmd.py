"""Tests for the cleanup subcommand — orphaned ZIM files and Ollama models."""

from pathlib import Path

from click.testing import CliRunner

from loki.cli import cli, _parse_ollama_list
from loki.config import LokiConfig


def _ollama_list_output(*model_names: str) -> str:
    """Return synthetic `ollama list` output for the given model names."""
    header = "NAME            ID              SIZE    MODIFIED"
    rows = "\n".join(f"{m}       abc123def456    4.7 GB  1 day ago" for m in model_names)
    return f"{header}\n{rows}\n" if rows else f"{header}\n"


# --- _parse_ollama_list unit tests ---

def test_parse_ollama_list_returns_model_names():
    """Model names are extracted from the first column of each data row."""
    output = _ollama_list_output("llama3:8b", "mistral:7b")
    assert _parse_ollama_list(output) == ["llama3:8b", "mistral:7b"]


def test_parse_ollama_list_returns_empty_for_header_only():
    """Only a header row returns an empty list."""
    assert _parse_ollama_list("NAME    ID    SIZE    MODIFIED\n") == []


def test_parse_ollama_list_returns_empty_for_blank_output():
    """Blank output (e.g., ollama not running) returns an empty list."""
    assert _parse_ollama_list("") == []


# --- ZIM file cleanup tests ---

def test_cleanup_no_orphaned_zims_prints_message(mocker, sample_config, tmp_path):
    """Cleanup prints a message when there are no orphaned ZIM files on disk."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No orphaned ZIM files found." in result.output


def test_cleanup_skips_configured_zim_file(mocker, sample_config, tmp_path):
    """Cleanup does not treat a ZIM file that matches a config entry as an orphan."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    filename = Path(sample_config.kiwix_files[0].url).name
    (tmp_path / filename).touch()
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output()
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No orphaned ZIM files found." in result.output


def test_cleanup_lists_orphaned_zim_file(mocker, sample_config, tmp_path):
    """Cleanup lists the names of ZIM files on disk that are not in the config."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    (tmp_path / "old_encyclopedia.zim").touch()
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert "old_encyclopedia.zim" in result.output


def test_cleanup_deletes_orphaned_zim_on_confirm(mocker, sample_config, tmp_path):
    """Cleanup removes an orphaned ZIM file from disk when the user confirms."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    orphan = tmp_path / "old_encyclopedia.zim"
    orphan.touch()
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    CliRunner().invoke(cli, ["cleanup"], input="y\n")

    assert not orphan.exists()


def test_cleanup_preserves_orphaned_zim_on_deny(mocker, sample_config, tmp_path):
    """Cleanup leaves an orphaned ZIM file on disk when the user denies."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    orphan = tmp_path / "old_encyclopedia.zim"
    orphan.touch()
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert orphan.exists()


def test_cleanup_handles_missing_kiwix_dir(mocker, sample_config, tmp_path):
    """Cleanup handles the case where the kiwix data directory does not exist."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path / "nonexistent")
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"])

    assert result.exit_code == 0
    assert "No orphaned ZIM files found." in result.output


# --- Ollama model cleanup tests ---

def test_cleanup_no_orphaned_models_prints_message(mocker, sample_config, tmp_path):
    """Cleanup prints a message when all installed models are in the config."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No orphaned Ollama models found." in result.output


def test_cleanup_skips_configured_ollama_model(mocker, sample_config, tmp_path):
    """Cleanup does not treat a model that matches a config entry as an orphan."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"])

    assert "No orphaned Ollama models found." in result.output


def test_cleanup_lists_orphaned_ollama_model(mocker, sample_config, tmp_path):
    """Cleanup lists installed models that are not present in the config."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b", "mistral:7b")
    mocker.patch("loki.cli.subprocess.run", autospec=True, return_value=list_result)

    result = CliRunner().invoke(cli, ["cleanup"], input="n\n")

    assert "mistral:7b" in result.output


def test_cleanup_removes_orphaned_model_on_confirm(mocker, sample_config, tmp_path):
    """Cleanup runs `ollama rm` for each orphaned model when the user confirms."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b", "mistral:7b")
    rm_result = mocker.MagicMock()
    rm_result.returncode = 0
    mock_run = mocker.patch(
        "loki.cli.subprocess.run",
        autospec=True,
        side_effect=[list_result, rm_result],
    )

    CliRunner().invoke(cli, ["cleanup"], input="y\n")

    mock_run.assert_any_call(["ollama", "rm", "mistral:7b"], check=False)


def test_cleanup_skips_model_removal_on_deny(mocker, sample_config, tmp_path):
    """Cleanup does not run `ollama rm` when the user denies the prompt."""
    mocker.patch("loki.cli.load_config", return_value=sample_config)
    mocker.patch("loki.cli.kiwix_dir", return_value=tmp_path)
    list_result = mocker.MagicMock()
    list_result.stdout = _ollama_list_output("llama3:8b", "mistral:7b")
    mock_run = mocker.patch(
        "loki.cli.subprocess.run", autospec=True, return_value=list_result
    )

    CliRunner().invoke(cli, ["cleanup"], input="n\n")

    for call in mock_run.call_args_list:
        assert call.args[0][:2] != ["ollama", "rm"]
