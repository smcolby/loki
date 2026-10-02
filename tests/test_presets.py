"""Tests for presets.py — assembling the llama-server models preset."""

import configparser

import pytest

from loki.presets import PresetError, build_models_preset, discover_presets


def _parse(text: str) -> configparser.ConfigParser:
    """Parse generated preset text the way a strict INI reader would."""
    parser = configparser.ConfigParser(
        delimiters=("=",), comment_prefixes=(";",), interpolation=None, default_section="__none__"
    )
    parser.optionxform = str  # type: ignore[method-assign, assignment]
    parser.read_string("[__top__]\n" + text)
    return parser


def test_resolves_relative_model_path_inside_preset_dir(tmp_path, write_preset):
    """A relative model path becomes the absolute path inside the preset's directory."""
    write_preset(tmp_path, "alpha")

    text, _ = build_models_preset(tmp_path, {})

    assert _parse(text)["alpha"]["model"] == str(tmp_path / "alpha" / "alpha.gguf")


def test_resolves_every_file_option(tmp_path, write_preset):
    """Draft, projector, and template paths resolve the same way as the model path."""
    model_dir = tmp_path / "beta"
    model_dir.mkdir()
    for name in ("beta.gguf", "draft.gguf", "mmproj.gguf", "chat.jinja"):
        (model_dir / name).touch()
    write_preset(
        tmp_path,
        "beta",
        "[beta]\nmodel = beta.gguf\nmodel-draft = draft.gguf\n"
        "mmproj = mmproj.gguf\nchat-template-file = chat.jinja\n",
    )

    section = _parse(build_models_preset(tmp_path, {})[0])["beta"]

    assert section["model-draft"] == str(model_dir / "draft.gguf")
    assert section["mmproj"] == str(model_dir / "mmproj.gguf")
    assert section["chat-template-file"] == str(model_dir / "chat.jinja")


def test_keeps_absolute_model_path(tmp_path, write_preset):
    """An absolute model path is written unchanged."""
    elsewhere = tmp_path / "shared" / "gamma.gguf"
    elsewhere.parent.mkdir()
    elsewhere.touch()
    write_preset(tmp_path / "models", "gamma", f"[gamma]\nmodel = {elsewhere}\n")

    text, _ = build_models_preset(tmp_path / "models", {})

    assert _parse(text)["gamma"]["model"] == str(elsewhere)


def test_copies_non_path_options_verbatim(tmp_path, write_preset):
    """Options other than file paths pass through with their values untouched."""
    write_preset(tmp_path, "delta", "[delta]\nmodel = delta.gguf\nctx-size = 262144\ntemp = 1.0\n")

    section = _parse(build_models_preset(tmp_path, {})[0])["delta"]

    assert section["ctx-size"] == "262144"
    assert section["temp"] == "1.0"


def test_writes_defaults_as_global_section(tmp_path, write_preset):
    """Defaults land in the [*] section with booleans spelled true/false."""
    write_preset(tmp_path, "alpha")

    text, _ = build_models_preset(tmp_path, {"n-gpu-layers": 99, "jinja": True, "mmap": False})

    globals_ = _parse(text)["*"]
    assert dict(globals_) == {"n-gpu-layers": "99", "jinja": "true", "mmap": "false"}


def test_declares_preset_version(tmp_path):
    """The generated file starts with the version key llama-server reserves."""
    text, _ = build_models_preset(tmp_path, {})

    assert _parse(text)["__top__"]["version"] == "1"


def test_returns_model_ids_sorted_by_directory(tmp_path, write_preset):
    """Model ids come back in directory order regardless of creation order."""
    write_preset(tmp_path, "zeta")
    write_preset(tmp_path, "alpha")

    _, model_ids = build_models_preset(tmp_path, {})

    assert model_ids == ["alpha", "zeta"]


def test_one_preset_can_declare_several_models(tmp_path, write_preset):
    """Two sections in one preset.ini become two models sharing the directory."""
    write_preset(tmp_path, "eta", "[eta-fast]\nmodel = eta.gguf\n\n[eta-long]\nmodel = eta.gguf\n")

    _, model_ids = build_models_preset(tmp_path, {})

    assert model_ids == ["eta-fast", "eta-long"]


def test_ignores_directories_without_preset(tmp_path, write_preset):
    """Subdirectories lacking preset.ini, and presets nested deeper, are not served."""
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "loose.gguf").touch()
    write_preset(tmp_path / "outer", "inner")

    _, model_ids = build_models_preset(tmp_path, {})

    assert model_ids == []


def test_missing_models_dir_yields_no_models(tmp_path):
    """A models directory that does not exist produces a preset with only defaults."""
    text, model_ids = build_models_preset(tmp_path / "absent", {"parallel": 1})

    assert model_ids == []
    assert "[*]" in text


def test_discover_presets_lists_one_level_deep(tmp_path, write_preset):
    """discover_presets finds preset.ini files only in immediate subdirectories."""
    first = write_preset(tmp_path, "a")
    second = write_preset(tmp_path, "b")
    write_preset(tmp_path / "b", "nested")

    assert discover_presets(tmp_path) == [first, second]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[theta]\nmodel = missing.gguf\n", "model not found"),
        ("[theta]\nmodel = theta.gguf\nmodel-draft = gone.gguf\n", "model-draft not found"),
        ("[theta]\nctx-size = 4096\n", "no 'model' option"),
        ("[*]\nn-gpu-layers = 99\n", "global settings belong in config.yaml"),
        ("model = theta.gguf\n", "File contains no section headers"),
        ("[theta]\nmodel = theta.gguf\n[theta]\nmodel = theta.gguf\n", "already exists"),
    ],
    ids=[
        "missing-model",
        "missing-draft",
        "no-model",
        "global-section",
        "no-header",
        "dup-in-file",
    ],
)
def test_rejects_invalid_preset(tmp_path, write_preset, body, message):
    """Malformed presets and dangling file references raise PresetError naming the file."""
    preset = write_preset(tmp_path, "theta", body)

    with pytest.raises(PresetError, match=message) as excinfo:
        build_models_preset(tmp_path, {})

    assert str(preset) in str(excinfo.value)


def test_rejects_model_id_repeated_across_presets(tmp_path, write_preset):
    """Two preset files declaring the same model id raise PresetError."""
    write_preset(tmp_path, "one", "[shared]\nmodel = one.gguf\n")
    write_preset(tmp_path, "two", "[shared]\nmodel = two.gguf\n")

    with pytest.raises(PresetError, match="'shared' is already defined"):
        build_models_preset(tmp_path, {})
