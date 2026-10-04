"""Tests for tools/image_studio.py: the Open WebUI pipe that drives Qwen Image 2.1."""

import base64
import json

import image_studio as studio_mod
import pytest
from aiohttp import web


def _message(role: str, *urls: str, text: str = "") -> dict:
    """Build a chat message with image files at ``urls``."""
    files = [{"type": "file", "content_type": "image/png", "url": url} for url in urls]
    return {"role": role, "content": text, "files": files}


def test_pipes_lists_the_three_studios():
    """The manifold offers Draft, Production, and Refine."""
    assert [p["id"] for p in studio_mod.Pipe().pipes()] == ["draft", "production", "refine"]


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1920, 1280), (1920, 1280)), ((1248, 832), (1248, 832)), ((4032, 3024), (1824, 1344))],
    ids=["production", "draft", "phone-photo"],
)
def test_refine_size_keeps_aspect_and_caps_pixels(size, expected):
    """Refine keeps the image's size up to 1920x1280 worth of pixels, in multiples of 32."""
    assert studio_mod.refine_size(*size) == expected


def test_draft_uses_only_current_attachments():
    """Draft and Production ignore images from earlier messages."""
    messages = [_message("user", "old"), _message("assistant", "gen"), _message("user", "a", "b")]

    assert studio_mod.select_images(studio_mod.STUDIOS["draft"], messages) == ["a", "b"]
    assert (
        studio_mod.select_images(
            studio_mod.STUDIOS["production"], messages[:2] + [_message("user")]
        )
        == []
    )


def test_refine_edits_newest_earlier_image_with_new_attachments_as_extras():
    """Refine targets the latest earlier image and adds this message's attachments after it."""
    messages = [
        _message("user", "p1", "p2"),
        _message("assistant", "gen1"),
        _message("assistant", "gen2"),
        _message("user", "extra"),
    ]

    assert studio_mod.select_images(studio_mod.STUDIOS["refine"], messages) == ["gen2", "extra"]


def test_refine_without_earlier_image_edits_first_attachment():
    """With nothing earlier, Refine edits the first image attached to this message."""
    messages = [_message("user", "upload", "ref")]

    assert studio_mod.select_images(studio_mod.STUDIOS["refine"], messages) == ["upload", "ref"]


def test_refine_without_any_image_raises():
    """Refine reports that it needs an image instead of generating from text."""
    with pytest.raises(ValueError, match="needs an image"):
        studio_mod.select_images(studio_mod.STUDIOS["refine"], [_message("user", text="fix it")])


def test_references_are_capped_at_ten():
    """Qwen Image 2.1 takes at most ten references."""
    messages = [_message("user", *[f"img{i}" for i in range(12)])]

    assert len(studio_mod.select_images(studio_mod.STUDIOS["draft"], messages)) == 10


@pytest.mark.parametrize(
    ("messages_map", "expected"),
    [
        ({"u": {"childrenIds": ["a"]}, "a": {"parentId": "u"}}, "u"),
        ({"u": {"childrenIds": ["a"]}, "a": {"parentId": None}}, "u"),
        ({"x": {"childrenIds": []}}, "current"),
    ],
    ids=["parent-link", "children-link", "fallback"],
)
def test_thread_end_finds_the_answered_user_message(messages_map, expected):
    """The thread ends at the reply's parent, found by parentId or by childrenIds."""
    assert studio_mod.thread_end(messages_map, "a", "current") == expected


def test_build_prompt_appends_step_count():
    """The prompt carries the studio's step count as sd-server extra args."""
    prompt = studio_mod.build_prompt("a fox", 40)

    assert prompt.startswith("a fox\n<sd_cpp_extra_args>")
    tag = prompt.split("<sd_cpp_extra_args>")[1].split("</sd_cpp_extra_args>")[0]
    assert json.loads(tag) == {"sample_params": {"sample_steps": 40}}


def test_message_text_joins_text_parts():
    """Multipart content contributes only its text parts."""
    message = {
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image_url"},
            {"type": "text", "text": "b"},
        ]
    }

    assert studio_mod.message_text(message) == "a\nb"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("title_generation", {"title": "A red fox in the snow"}),
        ("tags_generation", {"tags": []}),
        ("follow_up_generation", {"follow_ups": []}),
    ],
)
def test_task_reply_answers_background_tasks(task, expected):
    """Background tasks get valid JSON instead of starting an image generation."""
    assert json.loads(studio_mod.task_reply(task, "A red fox in the snow at dawn")) == expected


@pytest.fixture
async def gateway(aiohttp_server, monkeypatch):
    """Serve a fake gateway image API and point the pipe at it, recording each request."""
    seen: list[dict] = []
    png = base64.b64encode(b"png-bytes").decode()

    async def edits(request: web.Request) -> web.Response:
        form = await request.post()
        images = [f.file.read() for f in form.getall("image[]")]
        seen.append(
            {"path": "edits", "size": form["size"], "prompt": form["prompt"], "images": images}
        )
        return web.json_response({"data": [{"b64_json": png}]})

    async def generations(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append({"path": "generations", "size": body["size"], "model": body["model"]})
        if body["prompt"] == "fail":
            return web.json_response({"error": "boom"}, status=500)
        return web.json_response({"data": [{"b64_json": png}]})

    app = web.Application()
    app.router.add_post("/v1/images/edits", edits)
    app.router.add_post("/v1/images/generations", generations)
    server = await aiohttp_server(app)
    monkeypatch.setattr(studio_mod, "GATEWAY_URL", str(server.make_url("/v1")))
    return seen


async def test_generate_with_references_sends_a_multipart_edit(gateway):
    """References go to /images/edits as image[] parts in order, with the studio's size."""
    result = await studio_mod.Pipe()._generate(
        "prompt", (1248, 832), [(b"one", "image/png"), (b"two", "image/jpeg")]
    )

    assert result == b"png-bytes"
    assert gateway == [
        {"path": "edits", "size": "1248x832", "prompt": "prompt", "images": [b"one", b"two"]}
    ]


async def test_generate_without_references_sends_a_generation(gateway):
    """A text-only request goes to /images/generations for the image model."""
    result = await studio_mod.Pipe()._generate("prompt", (1920, 1280), [])

    assert result == b"png-bytes"
    assert gateway == [{"path": "generations", "size": "1920x1280", "model": "qwen-image-2.1-q8_0"}]


async def test_generate_raises_on_gateway_error(gateway):
    """A failed request raises ValueError naming the HTTP status."""
    with pytest.raises(ValueError, match="HTTP 500"):
        await studio_mod.Pipe()._generate("fail", (1920, 1280), [])
