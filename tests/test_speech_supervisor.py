"""Tests for speech/speech_supervisor.py: starting Breeze, the voice library, and generation."""

import io
import json
import socket
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import speech_supervisor as sup
from aiohttp import ClientSession, FormData

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


async def test_changes_without_the_studio_header_are_refused(client):
    """A cross-site form post cannot add the header, so changes without it get 403."""
    response = await client.post("/api/generate", data=speak_form("Hello there."))

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
        ({"cfg_scale": "11", "instruction": "x"}, "between 1 and 10"),
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
