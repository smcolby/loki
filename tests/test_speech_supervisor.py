"""Tests for speech/speech_supervisor.py: starting Breeze, the voice library, and generation."""

import asyncio
import io
import json
import socket
import sys
import time
import wave
import zipfile
from pathlib import Path

import av
import narration
import numpy as np
import pytest
import soundfile as sf
import speech_supervisor as sup
from aiohttp import ClientSession, FormData

from tests.epub_builder import make_epub, page

FAKE = Path(__file__).parent / "fake_breeze.py"
STUDIO = {sup.STUDIO_HEADER: "1"}


def free_port() -> int:
    """Return a loopback port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wav_bytes(seconds: float, rate: int = 44100, channels: int = 2) -> bytes:
    """Return a WAV file holding a quiet tone."""
    t = np.arange(int(seconds * rate)) / rate
    mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([mono] * channels, axis=1) if channels > 1 else mono
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def voice_form(seconds: float = 5.0, label: str = "Narrator", **fields: str) -> FormData:
    """Build the Studio's request to save a voice."""
    form = FormData()
    form.add_field("label", label)
    form.add_field("ref_text", "The exact words spoken in the clip.")
    form.add_field("ref_audio", wav_bytes(seconds), filename="ref.wav", content_type="audio/wav")
    for key, value in fields.items():
        form.add_field(key, value)
    return form


def speak_form(text: str, **fields: str) -> FormData:
    """Build the Studio's generation request."""
    form = FormData()
    form.add_field("text", text)
    for key, value in fields.items():
        form.add_field(key, value)
    return form


@pytest.fixture
def breeze_env(monkeypatch, tmp_path):
    """Environment for the fake Breeze process, with a request log it appends to."""
    log = tmp_path / "breeze-requests.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_SECONDS", "4")
    return log


def logged(log: Path) -> list[dict]:
    """Return the requests the fake Breeze received, in order."""
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.fixture
async def child(breeze_env):
    """A supervisor child that runs the fake Breeze, stopped at the end of the test."""
    port = free_port()
    process = sup.Child([sys.executable, str(FAKE), str(port)], port, start_timeout=30)
    yield process
    await process.stop()


@pytest.fixture
async def make_client(aiohttp_client, child, tmp_path):
    """Start the supervisor app over the fake Breeze and a library in ``tmp_path``."""
    session = ClientSession()

    async def make():
        return await aiohttp_client(sup.build_app(child, sup.Library(tmp_path / "data"), session))

    yield make
    await session.close()


@pytest.fixture
async def client(make_client):
    """Supervisor client over the fake Breeze."""
    return await make_client()


async def generate(client, text: str, **fields: str) -> tuple[int, bytes, dict]:
    """Post a generation request and return its status, body, and headers."""
    response = await client.post("/api/generate", data=speak_form(text, **fields), headers=STUDIO)
    return response.status, await response.read(), dict(response.headers)


# ---- text splitting ----


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("One. Two.", 20, [("One. Two.", 0.0)]),
        ("One. Two.\n\nThree.", 9, [("One. Two.", sup.PARAGRAPH_PAUSE), ("Three.", 0.0)]),
        ("Alpha. Beta. Gamma.", 12, [("Alpha. Beta.", sup.SENTENCE_PAUSE), ("Gamma.", 0.0)]),
        ("  \n\n  ", 20, []),
        ("one   spaced\nline.", 40, [("one spaced line.", 0.0)]),
    ],
    ids=["one chunk", "paragraph pause", "sentences packed to the limit", "blank", "whitespace"],
)
def test_split_text_cuts_at_sentence_ends(text, limit, expected):
    """Chunks end at sentence boundaries, with a longer pause after a paragraph."""
    assert sup.split_text(text, limit) == expected


def test_split_text_wraps_a_sentence_longer_than_the_limit_at_a_comma():
    """A sentence over the limit is cut at the last comma or space that fits."""
    chunks = sup.split_text("first part here, second part here and the rest.", 20)

    assert [text for text, _ in chunks] == ["first part here,", "second part here", "and the rest."]
    assert all(len(text) <= 20 for text, _ in chunks)


def test_split_text_keeps_every_word():
    """No word is lost or reordered by chunking."""
    text = " ".join(f"Sentence number {i} has some words." for i in range(60))

    chunks = sup.split_text(text, 120)

    assert " ".join(piece for piece, _ in chunks).split() == text.split()


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        (
            "Alpha beta gamma delta."
            ' He wrote: "I love to sail forbidden seas . . ."'
            " To the Greeks the world was small.",
            66,
            [
                "Alpha beta gamma delta.",
                'He wrote: "I love to sail forbidden seas . . ."',
                "To the Greeks the world was small.",
            ],
        ),
        ('She said "Go." Then she left.', 20, ['She said "Go."', "Then she left."]),
        ("Hi there. Wait. . . . Then go.", 20, ["Hi there.", "Wait. . . . Then go."]),
    ],
    ids=[
        "closing quote after a spaced ellipsis",
        "closing quote after a period",
        "spaced ellipsis",
    ],
)
def test_split_text_keeps_closing_quotes_and_ellipses_with_their_sentence(text, limit, expected):
    """A sentence ends after its closing quote, and the dots of a spaced ellipsis stay together."""
    assert [piece for piece, _ in sup.split_text(text, limit)] == expected


LONG = "a" * 19 + "."


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("One.\n\nTwo.", 20, [("One. Two.", 0.0)]),
        ("Aaaa.\n\nBbbb.", 6, [("Aaaa.", sup.PARAGRAPH_PAUSE), ("Bbbb.", 0.0)]),
        ("One.\n\n\n\nTwo.\n\nThree.", 11, [("One. Two.", sup.PARAGRAPH_PAUSE), ("Three.", 0.0)]),
        (
            "First line.\nSecond line.",
            15,
            [("First line.", sup.PARAGRAPH_PAUSE), ("Second line.", 0.0)],
        ),
        (
            "A wrapped\nline here.\nNext one.",
            25,
            [("A wrapped line here.", sup.PARAGRAPH_PAUSE), ("Next one.", 0.0)],
        ),
        ("", 20, []),
    ],
    ids=[
        "short paragraphs share a chunk",
        "paragraphs that do not fit together stay apart",
        "extra blank lines",
        "a line that ends a sentence ends a paragraph",
        "a line break inside a sentence joins the lines",
        "empty",
    ],
)
def test_plan_chunks_follows_paragraphs(text, limit, expected):
    """Paragraphs that fit together share a chunk, and a long paragraph is cut evenly."""
    assert sup.plan_chunks(text, limit) == expected


def test_plan_chunks_cuts_a_long_paragraph_into_pieces_of_even_length():
    """Four long sentences and a short one split as 2 and 3 sentences, not 3 and 2."""
    text = f"{LONG} {LONG} {LONG} {LONG} Zzzz."

    assert sup.plan_chunks(text, 63) == [
        (f"{LONG} {LONG}", sup.SENTENCE_PAUSE),
        (f"{LONG} {LONG} Zzzz.", 0.0),
    ]


def test_plan_chunks_does_not_join_the_pieces_of_a_cut_paragraph_with_its_neighbors():
    """Only whole paragraphs combine; the pieces of a paragraph cut at sentences stand alone."""
    middle = ["Aaaaaaaaaa.", "Bbbbbbbbbb.", "Cccccccccc.", "Dddddddddd."]
    text = "Hi.\n\n" + " ".join(middle) + "\n\nBye."

    assert sup.plan_chunks(text, 30) == [
        ("Hi.", sup.PARAGRAPH_PAUSE),
        ("Aaaaaaaaaa. Bbbbbbbbbb.", sup.SENTENCE_PAUSE),
        ("Cccccccccc. Dddddddddd.", sup.PARAGRAPH_PAUSE),
        ("Bye.", 0.0),
    ]


def test_plan_chunks_keeps_every_word_within_the_limit():
    """No word is lost or reordered and no chunk is over the limit, whatever the paragraph sizes."""
    paragraphs = [
        " ".join(f"Sentence {i}-{j} has a few words in it." for j in range(i % 7 + 1))
        for i in range(40)
    ]
    text = "\n\n".join(paragraphs)

    chunks = sup.plan_chunks(text, 200)

    assert " ".join(piece for piece, _ in chunks).split() == text.split()
    assert all(len(piece) <= 200 for piece, _ in chunks)


def test_job_chunk_limit_ends_before_breeze_stops_at_the_slowest_measured_pace():
    """A chunk of the job limit, spoken at 0.089 s per character, is shorter than Breeze's cap."""
    assert sup.JOB_CHUNK_CHARS * 0.089 < sup.CAP_FRAMES * 0.08


# ---- import ----


def upload_form(data: bytes, filename: str, **fields: str) -> FormData:
    """Return a multipart form with a file and extra fields."""
    form = FormData()
    form.add_field("file", data, filename=filename, content_type="application/octet-stream")
    for key, value in fields.items():
        form.add_field(key, value)
    return form


async def import_upload(client, data: bytes, filename: str, **fields: str):
    """Post a file to /api/import and return the response."""
    return await client.post(
        "/api/import", data=upload_form(data, filename, **fields), headers=STUDIO
    )


async def test_import_turns_an_epub_into_chapter_text_with_warnings(client):
    """The response has the book title, chapter-format text, chapter sizes, and what was removed."""
    files = {"a.xhtml": page("<h1>One</h1><p>It works [3] here.</p>")}
    epub = make_epub(files, [("a.xhtml", "One")], title="My Book")

    response = await import_upload(client, epub, "book.epub")

    assert response.status == 200
    body = await response.json()
    assert body["title"] == "My Book"
    assert body["text"] == "# One\n\nIt works here.\n"
    assert body["chapters"] == [{"title": "One", "chars": 14}]
    assert body["warnings"] == [{"kind": "numeric citation", "count": 1, "sample": "[3]"}]


async def test_import_reads_plain_text_and_applies_the_options(client):
    """A text file is imported as written, and a field of 0 turns a removal off."""
    text = b"Chapter 1\n\nIt works [3] here.\n"

    on = await (await import_upload(client, text, "a.txt")).json()
    off = await (await import_upload(client, text, "a.txt", citations="0")).json()

    assert "It works here." in on["text"]
    assert "It works [3] here." in off["text"]


@pytest.mark.parametrize(
    ("data", "filename", "message"),
    [
        (b"%PDF-1.7 abc", "paper.pdf", "PDF"),
        (b"PK\x03\x04 broken", "book.epub", "valid EPUB"),
        (b"abc\x00def", "a.txt", "not text"),
    ],
    ids=["pdf", "damaged epub", "binary"],
)
async def test_import_explains_why_a_file_cannot_be_read(client, data, filename, message):
    """An unsupported or damaged file gets a 400 with a reason, not a server error."""
    response = await import_upload(client, data, filename)

    assert response.status == 400
    assert message in (await response.json())["error"]["message"]


async def test_import_refuses_a_request_with_no_file_or_a_file_over_the_limit(client, monkeypatch):
    """A missing file field is a 400, and a file over the limit is a 413."""
    missing = await client.post("/api/import", data={"citations": "1"}, headers=STUDIO)
    monkeypatch.setattr(sup, "MAX_IMPORT_BYTES", 10)
    large = await import_upload(client, b"x" * 11, "a.txt")

    assert missing.status == 400
    assert large.status == 413


# ---- child process ----


def test_child_command_adds_the_private_listen_address():
    """Breeze listens on loopback at the supervisor's port, after the checkpoint path."""
    command = sup.child_command(["--fast-depth-decoder"], 8081)

    assert command[1:] == [
        "-m",
        "breeze_infer.api",
        "/model",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--fast-depth-decoder",
    ]


@pytest.mark.parametrize("flag", ["--host=0.0.0.0", "--port"])
def test_child_command_rejects_a_listen_address(flag):
    """Engine args may not move Breeze off the private port the supervisor proxies to."""
    with pytest.raises(ValueError, match="the image provides"):
        sup.child_command([flag, "9000"], 8081)


async def test_child_starts_on_demand_and_stop_kills_its_process_group(child):
    """Breeze reports loaded once started, and stopping ends the process."""
    async with ClientSession() as session:
        assert not child.loaded

        await child.ensure(session)
        process = child.process

        assert child.loaded
        assert child.state == "ready"
        assert await child._healthy(session)

    assert await child.stop()
    assert not child.loaded
    assert process is not None
    assert process.returncode is not None


async def test_child_refuses_to_stop_while_a_request_is_in_flight(child):
    """An in-flight request keeps Breeze loaded, so the gateway's unload is refused."""
    async with ClientSession() as session:
        await child.ensure(session)
    child.inflight = 1

    assert not await child.stop()
    assert child.loaded

    child.inflight = 0


async def test_child_start_failure_leaves_it_stopped(breeze_env):
    """A process that exits during start raises and does not count as loaded."""
    failing = sup.Child([sys.executable, "-c", "raise SystemExit(3)"], free_port(), 5)

    async with ClientSession() as session:
        with pytest.raises(RuntimeError, match="exited with code 3"):
            await failing.ensure(session)

    assert not failing.loaded


async def test_child_start_timeout_kills_the_process(breeze_env):
    """A process that never reports healthy is killed when the start timeout passes."""
    silent = sup.Child([sys.executable, "-c", "import time; time.sleep(60)"], free_port(), 1)

    async with ClientSession() as session:
        with pytest.raises(RuntimeError, match="did not become healthy"):
            await silent.ensure(session)

    assert not silent.loaded


# ---- supervisor endpoints ----


async def test_health_reports_loaded_only_after_breeze_starts(client, child):
    """The gateway sees an unloaded engine until the first generation starts Breeze."""
    before = await (await client.get("/health")).json()
    await generate(client, "Hello there.")
    after = await (await client.get("/health")).json()

    assert before == {"status": "ok", "loaded": False}
    assert after == {"status": "ok", "loaded": True}


async def test_models_lists_the_speech_model_without_starting_breeze(client, child):
    """The model listing the gateway polls does not load the GPU."""
    listing = await (await client.get("/v1/models")).json()

    assert [model["id"] for model in listing["data"]] == [sup.MODEL_ID]
    assert not child.loaded


async def test_unload_stops_breeze_and_is_refused_while_busy(client, child):
    """POST /unload frees Breeze, answers 409 while busy, and needs no Studio header."""
    await generate(client, "Hello there.")

    child.inflight = 1
    busy = await client.post("/unload", json={})
    child.inflight = 0
    freed = await client.post("/unload", json={})

    assert busy.status == 409
    assert freed.status == 200
    assert await (await client.get("/health")).json() == {"status": "ok", "loaded": False}


@pytest.mark.parametrize(
    "path", ["/api/generate", "/api/jobs", "/api/import", f"/api/jobs/{'0' * 32}/start"]
)
async def test_changes_without_the_studio_header_are_refused(client, path):
    """A cross-site form post cannot add the header, so changes without it get 403."""
    response = await client.post(path, data=speak_form("Hello there."))

    assert response.status == 403


async def test_responses_forbid_content_type_guessing(client):
    """Saved audio and metadata are never sniffed as another type."""
    response = await client.get("/api/voices")

    assert response.headers["X-Content-Type-Options"] == "nosniff"


async def test_studio_page_is_served(client):
    """The root path serves the Studio page."""
    response = await client.get("/")

    assert response.status == 200
    assert "Breeze Studio" in await response.text()


# ---- generation ----


async def test_generate_streams_pcm_and_saves_the_output(client, breeze_env):
    """A short text streams one chunk of PCM, and the same audio is saved with its settings."""
    status, body, headers = await generate(client, "Hello there.", seed="7")

    assert status == 200
    assert headers["X-Sample-Rate"] == "24000"
    assert headers["X-Chunk-Count"] == "1"
    assert len(body) == 4 * 24000 * 2
    output_id = headers["X-Output-Id"]
    with wave.open(str(client.app["library"].outputs_dir / f"{output_id}.wav")) as saved:
        assert saved.getframerate() == 24000
        assert saved.readframes(saved.getnframes()) == body
    listing = (await (await client.get("/api/outputs")).json())["outputs"]
    assert [item["id"] for item in listing] == [output_id]
    assert listing[0]["seed"] == 7
    assert listing[0]["duration_s"] == 4.0
    assert logged(breeze_env)[0]["text"] == "Hello there."


async def test_generate_speaks_long_text_in_chunks_with_pauses_and_new_seeds(client, breeze_env):
    """Each chunk is a separate request with the next seed, and silence separates them."""
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = sup.split_text(text)

    status, body, headers = await generate(client, text, seed="100")

    assert status == 200
    assert len(chunks) > 1
    assert headers["X-Chunk-Count"] == str(len(chunks))
    requests = logged(breeze_env)
    assert [item["text"] for item in requests] == [piece for piece, _ in chunks]
    assert [int(item["seed"]) for item in requests] == [100 + i for i in range(len(chunks))]
    pauses = sum(int(pause * 24000) * 2 for _, pause in chunks)
    assert len(body) == len(chunks) * 4 * 24000 * 2 + pauses


async def test_generate_with_a_voice_sends_its_clip_and_transcript(client, breeze_env):
    """A saved voice is cloned from its stored clip and exact transcript."""
    voice = await (await client.post("/api/voices", data=voice_form(), headers=STUDIO)).json()

    status, _, _ = await generate(client, "Hello there.", voice_id=voice["id"])

    assert status == 200
    request = logged(breeze_env)[0]
    assert request["ref_text"] == "The exact words spoken in the clip."
    assert request["ref_audio_bytes"] > 1000


async def test_generate_sends_guidance_only_with_an_instruction(client, breeze_env):
    """An instruction travels with its strength; without one neither field is sent."""
    await generate(client, "One.")
    await generate(client, "Two.", instruction="Speak slowly.", cfg_scale="4")

    plain, directed = logged(breeze_env)
    assert "instruction" not in plain
    assert "cfg_scale" not in plain
    assert directed["instruction"] == "Speak slowly."
    assert directed["cfg_scale"] == "4.0"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"cfg_scale": "4"}, "needs an instruction"),
        ({"cfg_scale": "7", "instruction": "x"}, "between 1 and 6"),
        ({"seed": "-1"}, "non-negative"),
        ({"seed": "abc"}, "must be a number"),
        ({"voice_id": "../../etc/passwd"}, "unknown voice_id"),
        ({"voice_id": "0" * 32}, "unknown voice_id"),
    ],
)
async def test_generate_rejects_bad_settings(client, breeze_env, fields, message):
    """Settings Breeze would answer with a 500, or that name no voice, are refused up front."""
    status, body, _ = await generate(client, "Hello there.", **fields)

    assert status == 400
    assert message in json.loads(body)["error"]["message"]
    assert not breeze_env.exists()


@pytest.mark.parametrize("text", ["", "   ", "x" * (sup.MAX_TEXT_CHARS + 1)])
async def test_generate_rejects_empty_or_oversized_text(client, text):
    """Text must have 1 to 60,000 characters."""
    status, _, _ = await generate(client, text)

    assert status == 400


async def test_a_second_generation_while_one_runs_is_refused(
    client, monkeypatch, tmp_path, child, breeze_env
):
    """Breeze serves one request at a time, so a second generation gets 409 at once."""
    gate = tmp_path / "gate"
    monkeypatch.setenv("FAKE_GATE", str(gate))
    first = await client.post("/api/generate", data=speak_form("First."), headers=STUDIO)
    await first.content.readany()

    second_status, _, _ = await generate(client, "Second.")
    gate.touch()
    await first.read()

    assert second_status == 409
    assert first.status == 200


async def test_generate_retries_a_chunk_while_breeze_is_busy(client, monkeypatch, breeze_env):
    """A 409 from Breeze, such as a cancelled request still finishing, is retried."""
    monkeypatch.setattr(sup.asyncio, "sleep", _no_sleep)
    monkeypatch.setenv("FAKE_BUSY_FIRST", "2")

    status, body, _ = await generate(client, "Hello there.")

    assert status == 200
    assert len(body) == 4 * 24000 * 2
    assert len(logged(breeze_env)) == 3


async def test_a_failed_chunk_closes_the_stream_and_saves_nothing(client, monkeypatch, breeze_env):
    """When Breeze fails after streaming began, the client sees a broken stream and no output."""
    monkeypatch.setenv("FAKE_FAIL", "1")
    response = await client.post("/api/generate", data=speak_form("Hello there."), headers=STUDIO)

    with pytest.raises(Exception, match="payload|closed|connection|incomplete"):
        await response.read()

    assert (await (await client.get("/api/outputs")).json())["outputs"] == []
    assert not list((client.app["library"].outputs_dir).glob("*.part"))


async def test_generate_reports_a_breeze_that_cannot_start(make_client, child, breeze_env):
    """A start failure is a 502 before any audio is streamed."""
    child.command = [sys.executable, "-c", "raise SystemExit(4)"]
    client = await make_client()

    status, body, _ = await generate(client, "Hello there.")

    assert status == 502
    assert "exited with code 4" in json.loads(body)["error"]["message"]


async def _no_sleep(_: float) -> None:
    """Skip a retry delay."""


# ---- voices ----


async def test_voice_upload_is_converted_to_a_24khz_mono_clip(client):
    """A stereo 44.1 kHz upload is saved as a 24 kHz mono clip with its transcript."""
    response = await client.post("/api/voices", data=voice_form(5.0), headers=STUDIO)
    voice = await response.json()

    assert response.status == 201
    assert voice["label"] == "Narrator"
    assert voice["duration_s"] == 5.0
    assert voice["source"] == "upload"
    clip = await (await client.get(f"/api/voices/{voice['id']}/audio")).read()
    samples, rate = sf.read(io.BytesIO(clip))
    assert rate == 24000
    assert samples.ndim == 1
    assert abs(len(samples) / rate - 5.0) < 0.01


@pytest.mark.parametrize("seconds", [1.0, 40.0])
async def test_voice_upload_outside_three_to_thirty_seconds_is_refused(client, seconds):
    """Clips too short to clone from, or long enough to crowd out the output, are refused."""
    response = await client.post("/api/voices", data=voice_form(seconds), headers=STUDIO)

    assert response.status == 400
    assert "use 3 to 30 s" in (await response.json())["error"]["message"]


async def test_voice_upload_that_is_not_audio_is_refused(client):
    """A file libsndfile cannot read gets a clear 400."""
    form = FormData()
    form.add_field("label", "Bad")
    form.add_field("ref_text", "words")
    form.add_field("ref_audio", b"not audio at all", filename="x.wav")

    response = await client.post("/api/voices", data=form, headers=STUDIO)

    assert response.status == 400
    assert "cannot read the audio file" in (await response.json())["error"]["message"]


async def test_voice_upload_needs_a_label_and_a_transcript(client):
    """A voice without a name or its exact transcript is refused."""
    form = FormData()
    form.add_field("ref_audio", wav_bytes(5), filename="ref.wav")

    response = await client.post("/api/voices", data=form, headers=STUDIO)

    assert response.status == 400


async def test_deleting_a_voice_removes_it_and_its_clip(client):
    """A deleted voice leaves the list and its audio is gone."""
    voice = await (await client.post("/api/voices", data=voice_form(), headers=STUDIO)).json()

    deleted = await client.delete(f"/api/voices/{voice['id']}", headers=STUDIO)
    listing = await (await client.get("/api/voices")).json()
    audio = await client.get(f"/api/voices/{voice['id']}/audio")

    assert deleted.status == 200
    assert listing == {"voices": []}
    assert audio.status == 404


@pytest.mark.parametrize("voice_id", ["..%2Foutputs", "..%2F..%2Fetc", "G" * 32, "a" * 31])
async def test_malformed_voice_ids_never_touch_the_filesystem(client, voice_id):
    """Ids that are not 32 hex digits are treated as unknown, so no path is built from them."""
    audio = await client.get(f"/api/voices/{voice_id}/audio")
    deleted = await client.delete(f"/api/voices/{voice_id}", headers=STUDIO)

    assert audio.status == 404
    assert deleted.status == 404


async def test_a_voice_can_be_saved_from_the_first_chunk_of_an_output(client):
    """Saving a designed output as a voice uses its first chunk and that chunk's text."""
    _, _, headers = await generate(client, "Design a voice. Then speak.", instruction="Warm.")

    response = await client.post(
        "/api/voices/from-output",
        json={"output_id": headers["X-Output-Id"], "label": "Designed"},
        headers=STUDIO,
    )
    voice = await response.json()

    assert response.status == 201
    assert voice["source"] == "output"
    assert voice["ref_text"] == "Design a voice. Then speak."
    assert voice["instruction"] == "Warm."
    assert voice["duration_s"] == 4.0


async def test_a_first_chunk_too_long_to_be_a_reference_is_refused(client, monkeypatch):
    """A sample longer than 30 s is refused with advice, instead of cutting its transcript."""
    monkeypatch.setenv("FAKE_SECONDS", "35")
    _, _, headers = await generate(client, "A long sample.")

    response = await client.post(
        "/api/voices/from-output",
        json={"output_id": headers["X-Output-Id"], "label": "Too long"},
        headers=STUDIO,
    )

    assert response.status == 400
    assert "about two sentences" in (await response.json())["error"]["message"]


# ---- saved audio ----


async def test_saved_audio_is_served_as_wav_and_encoded_to_aac_and_mp3(client):
    """Each format comes back with its own type, and the encoded copies carry the title tag."""
    _, _, headers = await generate(client, "Hello there.")
    base = f"/api/outputs/{headers['X-Output-Id']}/audio"

    wav = await client.get(base)
    m4a = await client.get(base + "?format=m4a")
    mp3 = await client.get(base + "?format=mp3")

    assert (wav.status, wav.content_type) == (200, "audio/wav")
    assert (await wav.read())[:4] == b"RIFF"
    assert (m4a.status, m4a.content_type) == (200, "audio/mp4")
    assert b"ftyp" in (await m4a.read())[:16]
    assert (mp3.status, mp3.content_type) == (200, "audio/mpeg")
    assert b"Hello there." in (await mp3.read())[:400]


async def test_encoded_audio_keeps_its_length_at_44khz(client, tmp_path):
    """The AAC copy is 4 s long, within one encoder frame, and 44.1 kHz."""
    import av

    _, _, headers = await generate(client, "Hello there.")
    body = await (
        await client.get(f"/api/outputs/{headers['X-Output-Id']}/audio?format=m4a")
    ).read()
    path = tmp_path / "copy.m4a"
    path.write_bytes(body)

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        assert stream.rate == 44100
        assert abs(float(container.duration) / 1e6 - 4.0) < 0.1


async def test_unknown_format_or_output_is_a_404(client):
    """Only known outputs and the three formats are served."""
    _, _, headers = await generate(client, "Hello there.")

    bad_format = await client.get(f"/api/outputs/{headers['X-Output-Id']}/audio?format=ogg")
    missing = await client.get(f"/api/outputs/{'0' * 32}/audio")

    assert bad_format.status == 404
    assert missing.status == 404


async def test_deleting_an_output_removes_every_copy(client):
    """The WAV, its metadata, and any encoded copy are all deleted."""
    _, _, headers = await generate(client, "Hello there.")
    output_id = headers["X-Output-Id"]
    await client.get(f"/api/outputs/{output_id}/audio?format=mp3")
    outputs = client.app["library"].outputs_dir
    assert len(list(outputs.glob(f"{output_id}.*"))) == 3

    response = await client.delete(f"/api/outputs/{output_id}", headers=STUDIO)

    assert response.status == 200
    assert list(outputs.glob(f"{output_id}.*")) == []


def test_encode_audio_rejects_an_unknown_format(tmp_path):
    """Only AAC and MP3 are encoded here."""
    with pytest.raises(ValueError, match="cannot encode 'flac'"):
        sup.encode_audio(tmp_path / "a.wav", tmp_path / "a.flac", "flac", "t")


def test_library_lists_newest_first(tmp_path):
    """Voices come back newest first, as the Studio shows them."""
    library = sup.Library(tmp_path)
    samples = np.zeros(24000 * 4, dtype=np.float32)
    first = library.add_voice("first", "text", samples, {})
    time.sleep(0.01)
    second = library.add_voice("second", "text", samples, {})

    assert [voice["id"] for voice in library.voices()] == [second["id"], first["id"]]


def _decoded_levels(path: Path) -> tuple[float, float]:
    """Return the RMS and peak, in dBFS, of an encoded file decoded to samples."""
    import av

    with av.open(str(path)) as container:
        samples = np.concatenate(
            [frame.to_ndarray().astype(np.float64).ravel() for frame in container.decode(audio=0)]
        )
    samples = samples / (32768.0 if np.abs(samples).max() > 4 else 1.0)
    return (
        20 * np.log10(np.sqrt(np.mean(samples**2))),
        20 * np.log10(np.abs(samples).max()),
    )


def _write_master(path: Path, amplitude: float, spike: float = 0.0) -> None:
    """Write a 10 s mono 24 kHz sine at ``amplitude``, with one tall sample if ``spike`` is set."""
    t = np.arange(24000 * 10) / 24000
    samples = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    samples[12000] = spike or samples[12000]
    sf.write(str(path), samples, 24000, subtype="PCM_16")


@pytest.mark.parametrize(
    ("amplitude", "spike", "gain"),
    [(0.03, 0.0, 13.5), (0.3, 0.0, -6.5), (0.05, 0.5, 2.5)],
    ids=["quiet master is raised", "hot master is lowered", "tall peak limits the gain"],
)
def test_export_gain_reaches_the_target_without_passing_the_peak_ceiling(
    tmp_path, amplitude, spike, gain
):
    """RMS goes to -20 dB, unless that would put a peak above -3.5 dB."""
    _write_master(tmp_path / "master.wav", amplitude, spike)

    assert sup.export_gain_db(tmp_path / "master.wav") == pytest.approx(gain, abs=0.2)


def test_export_gain_of_silence_is_zero(tmp_path):
    """A silent file has no level to adjust."""
    sf.write(str(tmp_path / "silent.wav"), np.zeros(24000, dtype=np.float32), 24000)

    assert sup.export_gain_db(tmp_path / "silent.wav") == 0.0


@pytest.mark.parametrize("fmt", ["m4a", "mp3"])
def test_encoded_copies_meet_the_acx_levels(tmp_path, fmt):
    """A quiet master comes out at about -20 dB RMS with peaks under -3 dB in both formats."""
    _write_master(tmp_path / "master.wav", 0.03)
    target = tmp_path / f"copy.{fmt}"

    sup.encode_audio(tmp_path / "master.wav", target, fmt, "Quiet")
    rms_db, peak_db = _decoded_levels(target)

    assert rms_db == pytest.approx(-20.0, abs=0.7)
    assert peak_db < -3.0


# ---- narration jobs ----

BOOK = "# Chapter One\n\nFirst chapter words here.\n\n# Chapter Two\n\nSecond chapter words here."
PLAIN = "Plain words long enough to pass."


async def create_job(client, text: str = BOOK, **fields) -> tuple[int, dict]:
    """Post a narration job and return the status and the response body."""
    response = await client.post("/api/jobs", json={"text": text, **fields}, headers=STUDIO)
    return response.status, await response.json()


async def start_job(client, job_id: str):
    """Ask the supervisor to start or resume a job."""
    return await client.post(f"/api/jobs/{job_id}/start", headers=STUDIO)


async def wait_for_state(client, job_id: str, states: tuple[str, ...]) -> dict:
    """Poll a job until its state is one of ``states``, and return its record."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = await (await client.get(f"/api/jobs/{job_id}")).json()
        if job["state"] in states:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job did not reach {states}: {job['state']}")


async def wait_for_requests(log: Path, count: int) -> None:
    """Wait until the fake Breeze has logged ``count`` requests."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if log.exists() and len(logged(log)) >= count:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"fake Breeze saw fewer than {count} requests")


@pytest.fixture
def gate(monkeypatch, tmp_path) -> Path:
    """A file that, until it exists, holds every fake Breeze response before its last block."""
    path = tmp_path / "gate"
    monkeypatch.setenv("FAKE_GATE", str(path))
    return path


@pytest.fixture
def paced(monkeypatch):
    """Make the fake Breeze speak at the pace the quality check expects."""
    monkeypatch.setenv("FAKE_PER_CHAR", str(narration.SECONDS_PER_CHAR))


async def test_create_job_splits_chapters_and_reads_each_heading_first(client, tmp_path):
    """A heading starts a chapter, and its title is the first chunk spoken in it."""
    status, job = await create_job(client, title="My book")

    assert status == 201
    assert job["state"] == "ready"
    assert [chapter["title"] for chapter in job["chapters"]] == ["Chapter One", "Chapter Two"]
    assert [chapter["chunks_total"] for chapter in job["chapters"]] == [2, 2]
    plan = narration.JobStore(tmp_path / "data").plan(job["id"])
    assert plan[0] == [("Chapter One", sup.PARAGRAPH_PAUSE), ("First chapter words here.", 0.0)]
    listed = await (await client.get("/api/jobs")).json()
    assert [entry["id"] for entry in listed["jobs"]] == [job["id"]]


async def test_create_job_can_leave_headings_unspoken(client, tmp_path):
    """With speak_titles off, chapters hold only their text."""
    _, job = await create_job(client, speak_titles=False)

    plan = narration.JobStore(tmp_path / "data").plan(job["id"])
    assert [len(chunks) for chunks in plan] == [1, 1]
    assert job["chunks_total"] == 2


async def test_create_job_does_not_read_a_title_nobody_wrote(client, tmp_path):
    """Text before the first heading is a chapter named for the job, and is not announced."""
    _, job = await create_job(
        client, "Opening words here.\n# Next\nMore words here.", title="Paper"
    )

    plan = narration.JobStore(tmp_path / "data").plan(job["id"])
    assert [chapter["title"] for chapter in job["chapters"]] == ["Paper", "Next"]
    assert [text for text, _ in plan[0]] == ["Opening words here."]
    assert [text for text, _ in plan[1]] == ["Next", "More words here."]


async def test_create_job_keeps_a_paragraph_whole_up_to_the_job_limit(client, tmp_path):
    """A paragraph under the job limit is one chunk, though it is over the generate limit."""
    sentence = "This sentence has some words in it. "
    paragraph = sentence * (sup.JOB_CHUNK_CHARS // len(sentence))
    assert sup.CHUNK_CHARS < len(paragraph) <= sup.JOB_CHUNK_CHARS

    _, job = await create_job(client, f"{paragraph}\n\n{paragraph}", speak_titles=False)

    plan = narration.JobStore(tmp_path / "data").plan(job["id"])
    assert [(len(text), pause) for text, pause in plan[0]] == [
        (len(paragraph.strip()), sup.PARAGRAPH_PAUSE),
        (len(paragraph.strip()), 0.0),
    ]


async def test_create_job_joins_short_paragraphs_into_one_chunk(client, tmp_path):
    """Paragraphs that fit together within the job limit are spoken as one chunk."""
    _, job = await create_job(client, "One wrapped\nline.\n\nNext one.", speak_titles=False)

    plan = narration.JobStore(tmp_path / "data").plan(job["id"])
    assert plan[0] == [("One wrapped line. Next one.", 0.0)]


async def test_create_job_titles_itself_after_the_first_chapter(client):
    """Without a title, the first chapter's title names the job, cut to 80 characters."""
    _, job = await create_job(client, "# " + "Long title " * 20 + "\nBody words here.")

    assert 0 < len(job["title"]) <= 80


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"text": ""}, "send 1 to"),
        ({"text": "# Heading only"}, "chapters"),
        ({"text": "x", "voice_id": "0" * 32}, "unknown voice_id"),
        ({"text": "x", "cfg_scale": 4}, "needs an instruction"),
        ({"text": "x", "speak_titles": "no"}, "speak_titles"),
    ],
)
async def test_create_job_rejects_bad_requests(client, fields, message):
    """Empty text, no chapter text, a missing voice, and bad settings are refused with 400."""
    response = await client.post("/api/jobs", json=fields, headers=STUDIO)

    assert response.status == 400
    assert message in (await response.json())["error"]["message"]


async def test_create_job_rejects_a_body_that_is_not_json(client):
    """A form post to /api/jobs gets 400 instead of an error page."""
    response = await client.post("/api/jobs", data="not json", headers=STUDIO)

    assert response.status == 400


async def test_a_job_speaks_every_chapter_into_a_library_output(
    client, breeze_env, paced, tmp_path
):
    """Each chapter becomes one output of its text's length plus pauses, and chunks are cleared."""
    _, job = await create_job(client, title="My book")

    started = await start_job(client, job["id"])
    done = await wait_for_state(client, job["id"], ("done", "failed"))

    assert started.status == 202
    assert done["state"] == "done"
    assert (done["chunks_done"], done["chars_done"]) == (4, done["chars_total"])
    outputs = (await (await client.get("/api/outputs")).json())["outputs"]
    by_title = {output["title"]: output for output in outputs}
    assert sorted(by_title) == ["My book - 01 Chapter One", "My book - 02 Chapter Two"]
    # 11 characters, a 0.8 s pause, then 25 characters at 0.075 s each
    assert by_title["My book - 01 Chapter One"]["duration_s"] == pytest.approx(3.5, abs=0.06)
    assert done["chapters"][0]["output_id"] == by_title["My book - 01 Chapter One"]["id"]
    assert [entry["seed"] for entry in logged(breeze_env)] == ["42", "43", "42", "43"]
    assert done["retries"] == 0
    assert [chapter["retries"] for chapter in done["chapters"]] == [[], []]
    assert not list((tmp_path / "data" / "jobs" / job["id"] / "chunks").iterdir())
    exported = await client.get(
        f"/api/outputs/{by_title['My book - 01 Chapter One']['id']}/audio?format=m4a"
    )
    assert exported.status == 200


async def test_a_job_clones_every_chunk_from_its_voice_and_instruction(client, breeze_env, paced):
    """The voice clip and the instruction go with every chunk's request."""
    voice = await (await client.post("/api/voices", data=voice_form(), headers=STUDIO)).json()
    _, job = await create_job(
        client, voice_id=voice["id"], instruction="Calm.", cfg_scale=4, seed=9
    )

    await start_job(client, job["id"])
    await wait_for_state(client, job["id"], ("done", "failed"))

    requests = logged(breeze_env)
    assert len(requests) == 4
    assert {request["ref_text"] for request in requests} == {voice["ref_text"]}
    assert all(request["ref_audio_bytes"] > 0 for request in requests)
    assert {(request["instruction"], request["cfg_scale"]) for request in requests} == {
        ("Calm.", "4.0")
    }


async def test_a_job_resumes_after_the_chunks_already_saved(client, breeze_env, paced, tmp_path):
    """Chunks on disk are not spoken again, and their audio is part of the chapter."""
    _, job = await create_job(client)
    first_title = np.frombuffer(bytes(19800 * 2), dtype=np.int16)
    store = narration.JobStore(tmp_path / "data")
    store.write_chunk(job["id"], 0, 0, (first_title + 3000).tobytes())

    await start_job(client, job["id"])
    done = await wait_for_state(client, job["id"], ("done", "failed"))

    assert done["state"] == "done"
    assert [entry["text"] for entry in logged(breeze_env)] == [
        "First chapter words here.",
        "Chapter Two",
        "Second chapter words here.",
    ]
    outputs = (await (await client.get("/api/outputs")).json())["outputs"]
    first = next(output for output in outputs if output["chapter"] == 1)
    assert first["duration_s"] == pytest.approx(3.5, abs=0.06)


async def test_a_chunk_that_fails_the_check_is_spoken_again_with_a_new_seed(
    client, breeze_env, paced, monkeypatch
):
    """A clip far shorter than its text is replaced by a second attempt with another seed."""
    monkeypatch.setenv("FAKE_SHORT_FIRST", "1")
    _, job = await create_job(client, PLAIN)

    await start_job(client, job["id"])
    done = await wait_for_state(client, job["id"], ("done", "failed"))

    assert done["state"] == "done"
    assert [entry["seed"] for entry in logged(breeze_env)] == ["42", str(42 + sup.RETRY_SEED_STEP)]
    assert done["chapters"][0]["warnings"] == []
    (retry,) = done["chapters"][0]["retries"]
    assert retry.startswith("part 1, attempt 1: too short")
    assert done["retries"] == 1


async def test_a_chunk_that_never_passes_is_used_with_a_warning(
    client, breeze_env, paced, monkeypatch
):
    """After three attempts the closest clip is used, and the chapter records why."""
    monkeypatch.setenv("FAKE_SHORT_FIRST", "9")
    _, job = await create_job(client, PLAIN)

    await start_job(client, job["id"])
    done = await wait_for_state(client, job["id"], ("done", "failed"))

    assert done["state"] == "done"
    assert len(logged(breeze_env)) == sup.JOB_ATTEMPTS
    (warning,) = done["chapters"][0]["warnings"]
    assert "too short" in warning
    assert f"all {sup.JOB_ATTEMPTS} attempts" in warning
    assert len(done["chapters"][0]["retries"]) == sup.JOB_ATTEMPTS
    assert done["retries"] == sup.JOB_ATTEMPTS


async def test_a_job_fails_with_the_error_when_breeze_keeps_failing(
    client, breeze_env, monkeypatch
):
    """Three failed attempts at one chunk end the job as failed, with the saved chunks intact."""
    monkeypatch.setenv("FAKE_FAIL", "1")
    _, job = await create_job(client, PLAIN)

    await start_job(client, job["id"])
    failed = await wait_for_state(client, job["id"], ("done", "failed"))

    assert failed["state"] == "failed"
    assert f"Breeze failed {sup.JOB_ATTEMPTS} times" in failed["error"]
    assert failed["chapters"][0]["state"] == "pending"
    status = await (await client.get("/api/status")).json()
    assert status["job"] == ""


async def test_stop_pauses_a_job_and_start_resumes_it(client, breeze_env, paced, gate):
    """Stopping discards the part in progress and frees Breeze; starting again finishes the job."""
    _, job = await create_job(client)
    await start_job(client, job["id"])
    await wait_for_requests(breeze_env, 1)

    stopped = await client.post(f"/api/jobs/{job['id']}/stop", headers=STUDIO)
    status = await (await client.get("/api/status")).json()

    assert (await stopped.json())["state"] == "paused"
    assert (status["job"], status["generating"]) == ("", False)
    gate.write_text("open")
    await start_job(client, job["id"])
    done = await wait_for_state(client, job["id"], ("done", "failed"))
    assert done["state"] == "done"
    assert done["chunks_done"] == 4


async def test_a_running_job_refuses_other_generation_unload_and_deletion(
    client, breeze_env, paced, gate
):
    """While a job runs, a second job, a Studio generation, an unload, and a delete get 409."""
    _, job = await create_job(client)
    _, other = await create_job(client, PLAIN)
    await start_job(client, job["id"])
    await wait_for_requests(breeze_env, 1)

    second = await start_job(client, other["id"])
    generation, *_ = await generate(client, "Hello there.")
    unload = await client.post("/unload", json={})
    deletion = await client.delete(f"/api/jobs/{job['id']}", headers=STUDIO)

    assert (second.status, generation, unload.status, deletion.status) == (409, 409, 409, 409)
    await client.post(f"/api/jobs/{job['id']}/stop", headers=STUDIO)
    assert (await client.post("/unload", json={})).status == 200


async def test_a_job_left_running_by_a_restart_comes_back_interrupted(make_client):
    """A supervisor that starts and finds a running job marks it interrupted so it can resume."""
    first = await make_client()
    _, job = await create_job(first)
    on_disk = {**job, "state": "running"}
    narration.JobStore(Path(first.app["library"].root)).save(on_disk)

    second = await make_client()
    after = await (await second.get(f"/api/jobs/{job['id']}")).json()

    assert after["state"] == "interrupted"


async def test_start_refuses_a_finished_job_and_one_whose_voice_was_deleted(
    client, breeze_env, paced
):
    """A finished job and a job whose voice is gone cannot be started."""
    voice = await (await client.post("/api/voices", data=voice_form(), headers=STUDIO)).json()
    _, voiced = await create_job(client, PLAIN, voice_id=voice["id"])
    _, plain = await create_job(client, PLAIN)
    await start_job(client, plain["id"])
    await wait_for_state(client, plain["id"], ("done", "failed"))

    await client.delete(f"/api/voices/{voice['id']}", headers=STUDIO)
    finished = await start_job(client, plain["id"])
    orphaned = await start_job(client, voiced["id"])

    assert (finished.status, orphaned.status) == (409, 409)
    assert "voice" in (await orphaned.json())["error"]["message"]


async def test_unknown_jobs_are_not_found(client):
    """Reading, starting, stopping, or deleting a job that does not exist is a 404 or 409."""
    unknown = "0" * 32

    assert (await client.get(f"/api/jobs/{unknown}")).status == 404
    assert (await client.get("/api/jobs/not-an-id")).status == 404
    assert (await start_job(client, unknown)).status == 404
    assert (await client.post(f"/api/jobs/{unknown}/stop", headers=STUDIO)).status == 409
    assert (await client.delete(f"/api/jobs/{unknown}", headers=STUDIO)).status == 404


def test_zip_member_name_numbers_the_chapter_and_removes_characters_files_reject():
    """The filename is the padded number and a title without path or reserved characters."""
    assert sup.zip_member_name(3, 2, 'Why: a "test"?', "m4a") == "03 Why a test.m4a"
    assert sup.zip_member_name(7, 3, "a/b\\c", "mp3") == "007 a b c.mp3"
    assert sup.zip_member_name(1, 2, "...", "wav") == "01 Chapter.wav"
    assert len(sup.zip_member_name(1, 2, "x" * 500, "wav")) < 120


async def finished_job(client, breeze_env) -> dict:
    """Run a two-chapter job to completion and return its record."""
    _, job = await create_job(client, title="My book")
    await start_job(client, job["id"])
    return await wait_for_state(client, job["id"], ("done", "failed"))


async def test_a_zip_holds_each_finished_chapter_in_order_with_album_and_track_tags(
    client, breeze_env, paced, tmp_path
):
    """The zip has one numbered file per chapter, tagged so a player groups and orders them."""
    done = await finished_job(client, breeze_env)

    response = await client.get(f"/api/jobs/{done['id']}/zip?format=m4a")

    assert response.status == 200
    assert response.headers["Content-Type"] == "application/zip"
    assert 'filename="my-book.zip"' in response.headers["Content-Disposition"]
    archive = zipfile.ZipFile(io.BytesIO(await response.read()))
    assert archive.namelist() == ["01 Chapter One.m4a", "02 Chapter Two.m4a"]
    assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    tagged = tmp_path / "chapter.m4a"
    tagged.write_bytes(archive.read("02 Chapter Two.m4a"))
    with av.open(str(tagged)) as container:
        assert container.metadata["album"] == "My book"
        assert container.metadata["track"] == "2/2"

    # The handler removes the temporary zip as it returns, just after the client has the last byte
    leftover = tmp_path / "data" / "tmp"
    for _ in range(40):
        if not list(leftover.iterdir()):
            break
        await asyncio.sleep(0.05)
    assert not list(leftover.iterdir())


async def test_a_zip_lists_chapters_that_have_no_audio(client, breeze_env, paced):
    """A chapter whose output was deleted is left out and listed in MISSING.txt."""
    done = await finished_job(client, breeze_env)
    await client.delete(f"/api/outputs/{done['chapters'][1]['output_id']}", headers=STUDIO)

    response = await client.get(f"/api/jobs/{done['id']}/zip?format=wav")

    archive = zipfile.ZipFile(io.BytesIO(await response.read()))
    assert archive.namelist() == ["01 Chapter One.wav", "MISSING.txt"]
    assert "02 Chapter Two" in archive.read("MISSING.txt").decode()


async def test_a_zip_is_refused_for_a_job_with_no_finished_chapter_or_a_bad_request(client):
    """A new job has nothing to zip (409), and an unknown job or format is a 404."""
    _, job = await create_job(client)

    assert (await client.get(f"/api/jobs/{job['id']}/zip")).status == 409
    assert (await client.get(f"/api/jobs/{job['id']}/zip?format=ogg")).status == 404
    assert (await client.get(f"/api/jobs/{'0' * 32}/zip")).status == 404


async def test_the_library_clears_temporary_zip_files_left_by_a_restart(tmp_path):
    """A zip a cut-off download left behind is removed when the library opens."""
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "left.zip").write_bytes(b"PK")

    sup.Library(tmp_path)

    assert list((tmp_path / "tmp").iterdir()) == []


async def test_deleting_a_job_removes_its_files_but_not_its_finished_chapters(
    client, breeze_env, paced, tmp_path
):
    """Chapter outputs stay in the library after their job is deleted."""
    _, job = await create_job(client, PLAIN)
    await start_job(client, job["id"])
    await wait_for_state(client, job["id"], ("done", "failed"))

    deleted = await client.delete(f"/api/jobs/{job['id']}", headers=STUDIO)

    assert deleted.status == 200
    assert not (tmp_path / "data" / "jobs" / job["id"]).exists()
    assert len((await (await client.get("/api/outputs")).json())["outputs"]) == 1
