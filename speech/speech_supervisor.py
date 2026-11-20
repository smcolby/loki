"""Run Breeze TTS 2 on demand and serve its Studio, voice library, and saved audio.

Breeze keeps about 9.5 GiB of GPU memory while it runs and takes 40 to 95 s to
start, so this supervisor owns the public port and starts Breeze for the first
generation request. ``POST /unload`` stops it, and ``GET /health`` reports
whether it is starting or running, matching Strata's API so the gateway
unloads it the same way. The supervisor itself stays up while another model
holds the GPU, so the voice library and saved audio remain reachable.

A generation request carries text of any length. The supervisor splits it into
chunks Breeze can speak in one request, streams the audio to the browser as it
is made, and saves the result to the library with its settings.
"""

import asyncio
import contextlib
import io
import json
import logging
import math
import os
import re
import secrets
import signal
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import soundfile as sf
import soxr
from aiohttp import ClientError, ClientSession, ClientTimeout, FormData, web
from av.audio.stream import AudioStream
from av.error import FFmpegError

log = logging.getLogger("speech")

PUBLIC_PORT = 8080
CHILD_PORT = 8081
START_TIMEOUT = 600.0
OWNED_FLAGS = ("--host", "--port")
MODEL_ID = "breeze-tts-2"
SAMPLE_RATE = 24000
FRAME_BYTES = 1920 * 2
# Export levels from ACX's audiobook requirements: RMS between -23 and -18 dB, peaks under -3 dB
TARGET_RMS_DB = -20.0
PEAK_CEILING_DB = -3.0
CAP_FRAMES = 1499
MAX_TEXT_CHARS = 60_000
CHUNK_CHARS = 500
SENTENCE_PAUSE = 0.3
PARAGRAPH_PAUSE = 0.8
SPEAK_ATTEMPTS = 6
REFERENCE_SECONDS = (3.0, 30.0)
AUDIO_FORMATS = {
    "wav": ("audio/wav", "wav"),
    "m4a": ("audio/mp4", "m4a"),
    "mp3": ("audio/mpeg", "mp3"),
}
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
STUDIO_HEADER = "X-Loki-Studio"
STUDIO_PAGE = Path(__file__).parent / "studio" / "index.html"


class ChunkError(RuntimeError):
    """Breeze could not speak one chunk of the request."""


def child_command(args: list[str], port: int) -> list[str]:
    """Return the command line that starts Breeze's API on a private port.

    Parameters
    ----------
    args : list[str]
        Extra ``breeze_infer.api`` options from the engine config.
    port : int
        Loopback port Breeze listens on.

    Returns
    -------
    list[str]
        Command line for ``asyncio.create_subprocess_exec``.

    Raises
    ------
    ValueError
        If ``args`` sets a listen address, which the supervisor owns.

    Examples
    --------
    >>> child_command(["--fast-depth-decoder"], 8081)[-3:]
    ['--port', '8081', '--fast-depth-decoder']
    """
    owned = sorted({arg.split("=")[0] for arg in args} & set(OWNED_FLAGS))
    if owned:
        raise ValueError(f"engine args set {', '.join(owned)}; the image provides these")
    return [
        sys.executable,
        "-m",
        "breeze_infer.api",
        "/model",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *args,
    ]


def _wrap(sentence: str, limit: int) -> list[str]:
    """Split a sentence over ``limit`` characters at commas or spaces into parts that fit."""
    parts = []
    while len(sentence) > limit:
        cut = max(sentence.rfind(", ", 0, limit), sentence.rfind(" ", 0, limit))
        cut = limit if cut <= 0 else cut + 1
        parts.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    return [*parts, sentence] if sentence else parts


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[tuple[str, float]]:
    r"""Cut text into chunks that end at sentence boundaries, with the pause after each.

    Paragraph breaks get a longer pause than sentence breaks, and the last chunk
    has none.

    Parameters
    ----------
    text : str
        Text to speak; blank lines separate paragraphs.
    limit : int, optional
        Most characters in one chunk. Default is ``CHUNK_CHARS``.

    Returns
    -------
    list of tuple of (str, float)
        Each chunk and the seconds of silence to add after it.

    Examples
    --------
    >>> split_text("One. Two.\n\nThree.", limit=9)
    [('One. Two.', 0.8), ('Three.', 0.0)]
    """
    chunks: list[tuple[str, float]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        pieces, current = [], ""
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            for part in _wrap(sentence, limit):
                if current and len(current) + 1 + len(part) > limit:
                    pieces.append(current)
                    current = part
                else:
                    current = f"{current} {part}".strip()
        if current:
            pieces.append(current)
        chunks += [
            (piece, PARAGRAPH_PAUSE if index == len(pieces) - 1 else SENTENCE_PAUSE)
            for index, piece in enumerate(pieces)
        ]
    if chunks:
        chunks[-1] = (chunks[-1][0], 0.0)
    return chunks


def decode_reference(data: bytes) -> np.ndarray:
    """Decode an uploaded reference clip to 24 kHz mono float samples.

    Parameters
    ----------
    data : bytes
        Audio file contents in any format libsndfile reads.

    Returns
    -------
    numpy.ndarray
        Mono float32 samples at 24 kHz.

    Raises
    ------
    ValueError
        If the file cannot be decoded or its length is outside ``REFERENCE_SECONDS``.
    """
    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError(f"cannot read the audio file: {exc}") from exc
    mono = samples.mean(axis=1)
    seconds = len(mono) / rate
    low, high = REFERENCE_SECONDS
    if not low <= seconds <= high:
        raise ValueError(f"the reference clip is {seconds:.1f} s; use {low:.0f} to {high:.0f} s")
    if rate != SAMPLE_RATE:
        mono = soxr.resample(mono, rate, SAMPLE_RATE)
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


def export_gain_db(source: Path) -> float:
    """Return the gain that brings a WAV file's RMS level to ``TARGET_RMS_DB``.

    The gain is limited so the loudest sample stays under ``PEAK_CEILING_DB``, which
    means a file with tall peaks ends up quieter than the target instead of clipping.

    Parameters
    ----------
    source : Path
        WAV file to measure; read in blocks.

    Returns
    -------
    float
        Gain in decibels; 0.0 for a silent file.
    """
    energy, count, peak = 0.0, 0, 0.0
    for block in sf.blocks(str(source), blocksize=SAMPLE_RATE * 30, dtype="float32"):
        energy += float(np.square(block, dtype=np.float64).sum())
        count += len(block)
        peak = max(peak, float(np.abs(block).max(initial=0.0)))
    if not count or energy == 0.0 or peak == 0.0:
        return 0.0
    rms_db = 10 * math.log10(energy / count)
    peak_db = 20 * math.log10(peak)
    return min(TARGET_RMS_DB - rms_db, PEAK_CEILING_DB - peak_db)


def encode_audio(source: Path, target: Path, fmt: str, title: str) -> None:
    """Encode a 24 kHz mono WAV file as AAC (``m4a``) or MP3 at 44.1 kHz, with tags.

    AAC uses 128 kbps, the top tier of Audible's own files, and MP3 uses 128 kbps.
    The level is set by :func:`export_gain_db`, so the copies meet ACX's loudness
    limits whatever the master's level; the WAV master is not changed. The source is
    read and encoded in blocks, so a book-length file stays out of memory.

    Parameters
    ----------
    source : Path
        WAV file to encode.
    target : Path
        File to write; written whole or not at all.
    fmt : str
        ``"m4a"`` or ``"mp3"``.
    title : str
        Title tag.

    Raises
    ------
    ValueError
        If ``fmt`` is not ``"m4a"`` or ``"mp3"``.
    """
    codec, container = {"m4a": ("aac", "ipod"), "mp3": ("libmp3lame", "mp3")}.get(fmt, (None, None))
    if codec is None:
        raise ValueError(f"cannot encode {fmt!r}")
    rate = 44100
    partial = target.with_name(target.name + ".part")
    output = av.open(str(partial), "w", format=container)
    output.metadata.update(
        {"title": title, "artist": "Breeze TTS 2 (synthetic speech)", "album": "loki"}
    )
    stream = cast(AudioStream, output.add_stream(codec, rate=rate))
    stream.bit_rate = 128_000
    stream.layout = "mono"
    frame_size = 1024 if fmt == "m4a" else 1152
    resampler = soxr.ResampleStream(SAMPLE_RATE, rate, 1, dtype="float32")
    pending = np.zeros(0, dtype=np.int16)

    def mux(samples: np.ndarray, final: bool) -> None:
        nonlocal pending
        pending = np.concatenate([pending, (np.clip(samples, -1, 1) * 32767).astype(np.int16)])
        while len(pending) >= frame_size or (final and len(pending)):
            piece, pending = pending[:frame_size], pending[frame_size:]
            frame = av.AudioFrame.from_ndarray(piece[None, :], format="s16", layout="mono")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                output.mux(packet)

    gain = 10 ** (export_gain_db(source) / 20)
    try:
        for block in sf.blocks(str(source), blocksize=SAMPLE_RATE * 30, dtype="float32"):
            mux(resampler.resample_chunk(block * gain, last=False), final=False)
        mux(resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True), final=True)
        for packet in stream.encode(None):
            output.mux(packet)
    finally:
        output.close()
    partial.replace(target)


def _json_write(path: Path, data: dict[str, Any]) -> None:
    """Write JSON beside its final path, then rename it into place so readers see all or none."""
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, indent=1))
    partial.replace(path)


def _json_read(path: Path) -> dict[str, Any] | None:
    """Return the JSON object stored at ``path``, or None when it is missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Library:
    """Saved voices and generated audio under one data directory.

    A voice is a reference clip, its exact transcript, and default settings:
    Breeze clones from the clip on every request instead of training a model.

    Parameters
    ----------
    root : Path
        Directory that holds ``voices/`` and ``outputs/``; created if missing.
    """

    def __init__(self, root: Path):
        self.voices_dir = root / "voices"
        self.outputs_dir = root / "outputs"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def voice(self, voice_id: str) -> dict[str, Any] | None:
        """Return a voice's metadata, or None for an unknown or malformed id."""
        if not ID_PATTERN.fullmatch(voice_id):
            return None
        return _json_read(self.voices_dir / voice_id / "meta.json")

    def voices(self) -> list[dict[str, Any]]:
        """Return every voice, newest first."""
        found = [_json_read(path / "meta.json") for path in self.voices_dir.iterdir()]
        return sorted(
            (meta for meta in found if meta), key=lambda meta: meta["created"], reverse=True
        )

    def reference_path(self, voice_id: str) -> Path:
        """Return the path of a voice's reference clip; the id must come from :meth:`voice`."""
        return self.voices_dir / voice_id / "ref.wav"

    def add_voice(
        self, label: str, ref_text: str, samples: np.ndarray, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Save a voice and return its metadata.

        Parameters
        ----------
        label : str
            Name shown in the Studio.
        ref_text : str
            Exact transcript of the clip.
        samples : numpy.ndarray
            Mono float32 samples at 24 kHz.
        settings : dict
            Default ``instruction``, ``cfg_scale``, ``seed`` and ``source`` for the voice.
        """
        voice_id = secrets.token_hex(16)
        folder = self.voices_dir / voice_id
        folder.mkdir()
        sf.write(str(folder / "ref.wav"), samples, SAMPLE_RATE, subtype="PCM_16")
        meta = {
            "id": voice_id,
            "label": label,
            "ref_text": ref_text,
            "duration_s": round(len(samples) / SAMPLE_RATE, 1),
            "created": time.time(),
            **settings,
        }
        _json_write(folder / "meta.json", meta)
        return meta

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice and its clip; return whether it existed."""
        if self.voice(voice_id) is None:
            return False
        for path in (self.voices_dir / voice_id).iterdir():
            path.unlink()
        (self.voices_dir / voice_id).rmdir()
        return True

    def output(self, output_id: str) -> dict[str, Any] | None:
        """Return a saved output's metadata, or None for an unknown or malformed id."""
        if not ID_PATTERN.fullmatch(output_id):
            return None
        return _json_read(self.outputs_dir / f"{output_id}.json")

    def outputs(self) -> list[dict[str, Any]]:
        """Return every saved output, newest first."""
        found = [_json_read(path) for path in self.outputs_dir.glob("*.json")]
        return sorted(
            (meta for meta in found if meta), key=lambda meta: meta["created"], reverse=True
        )

    def delete_output(self, output_id: str) -> bool:
        """Delete a saved output and every encoded copy; return whether it existed."""
        if self.output(output_id) is None:
            return False
        for path in self.outputs_dir.glob(f"{output_id}.*"):
            path.unlink()
        return True


@dataclass
class Child:
    """The Breeze API process, started for generation and stopped on unload.

    Parameters
    ----------
    command : list[str]
        Command line that starts Breeze on ``port``.
    port : int
        Loopback port the command listens on.
    start_timeout : float
        Seconds to wait for a started process to report healthy.
    """

    command: list[str]
    port: int
    start_timeout: float = START_TIMEOUT
    process: asyncio.subprocess.Process | None = None
    state: str = "stopped"
    inflight: int = 0
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def url(self) -> str:
        """Base URL of Breeze's API."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def loaded(self) -> bool:
        """Report whether Breeze is starting or running, since either holds GPU memory."""
        return self.state != "stopped"

    def _alive(self) -> bool:
        """Report whether the process exists and has not exited."""
        return self.process is not None and self.process.returncode is None

    async def _healthy(self, session: ClientSession) -> bool:
        """Report whether Breeze answers its health check."""
        try:
            async with session.get(self.url + "/health") as response:
                return response.status == 200
        except (ClientError, TimeoutError, OSError):
            return False

    async def ensure(self, session: ClientSession) -> None:
        """Start Breeze if needed and wait until it is healthy.

        Raises
        ------
        RuntimeError
            If the process exits during start, is stopped, or is not healthy in time.
        """
        async with self._start_lock:
            if self.state == "ready" and self._alive():
                return
            self.process = await asyncio.create_subprocess_exec(
                *self.command, start_new_session=True
            )
            self.state = "starting"
            log.info("starting Breeze (pid %d)", self.process.pid)
            began = time.monotonic()
            deadline = began + self.start_timeout
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    self.state = "stopped"
                    raise RuntimeError(f"Breeze exited with code {self.process.returncode}")
                if await self._healthy(session):
                    self.state = "ready"
                    log.info("Breeze ready after %.0f s", time.monotonic() - began)
                    return
                await asyncio.sleep(0.5)
            await self._kill()
            raise RuntimeError(f"Breeze did not become healthy within {self.start_timeout:.0f} s")

    async def _kill(self) -> None:
        """Kill the process group, which includes Breeze's compile workers, and wait for it."""
        process, self.process, self.state = self.process, None, "stopped"
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def stop(self) -> bool:
        """Stop Breeze unless a request is in flight; return whether it is stopped."""
        if self.inflight:
            return False
        await self._kill()
        return True


def _error(status: int, message: str) -> web.Response:
    """Build a JSON error response."""
    return web.json_response({"error": {"message": message}}, status=status)


@web.middleware
async def guard(request: web.Request, handler: Any) -> web.StreamResponse:
    """Require the Studio header on changes and stop browsers from guessing content types.

    A page on another site can post a plain form to a LAN address without
    permission, but it cannot add a custom header, so the header proves the
    request came from the Studio.
    """
    if request.method not in ("GET", "HEAD") and request.path != "/unload":
        if STUDIO_HEADER not in request.headers:
            return _error(403, f"requests that change data need the {STUDIO_HEADER} header")
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        exc.headers["X-Content-Type-Options"] = "nosniff"
        raise
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


class Supervisor:
    """Handlers for the Studio, the library, and generation.

    Parameters
    ----------
    child : Child
        The Breeze process this supervisor starts and stops.
    library : Library
        Saved voices and audio.
    session : ClientSession
        Client session for requests to Breeze.
    """

    def __init__(self, child: Child, library: Library, session: ClientSession):
        self.child = child
        self.library = library
        self.session = session
        self.generating = asyncio.Lock()
        self.progress: dict[str, Any] = {}

    def routes(self) -> list[web.RouteDef]:
        """Return the supervisor's routes."""
        return [
            web.get("/", self.page),
            web.get("/health", self.health),
            web.post("/unload", self.unload),
            web.get("/v1/models", self.models),
            web.get("/api/status", self.status),
            web.get("/api/voices", self.list_voices),
            web.post("/api/voices", self.add_voice),
            web.post("/api/voices/from-output", self.voice_from_output),
            web.delete("/api/voices/{voice_id}", self.delete_voice),
            web.get("/api/voices/{voice_id}/audio", self.voice_audio),
            web.get("/api/outputs", self.list_outputs),
            web.delete("/api/outputs/{output_id}", self.delete_output),
            web.get("/api/outputs/{output_id}/audio", self.output_audio),
            web.post("/api/generate", self.generate),
        ]

    async def page(self, _: web.Request) -> web.StreamResponse:
        """Serve the Studio page."""
        return web.FileResponse(STUDIO_PAGE, headers={"Cache-Control": "no-cache"})

    async def health(self, _: web.Request) -> web.Response:
        """Report whether Breeze is loading or loaded, as Strata's health does."""
        return web.json_response({"status": "ok", "loaded": self.child.loaded})

    async def models(self, _: web.Request) -> web.Response:
        """List the speech model without starting Breeze."""
        return web.json_response({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})

    async def unload(self, request: web.Request) -> web.Response:
        """Stop Breeze, or answer 409 while a request is in flight."""
        await request.read()
        if await self.child.stop():
            return web.json_response({"status": "unloaded"})
        return web.json_response({"status": "busy"}, status=409)

    async def status(self, _: web.Request) -> web.Response:
        """Report Breeze's state and the progress of the running generation."""
        return web.json_response(
            {
                "state": self.child.state,
                "generating": self.generating.locked(),
                "progress": self.progress if self.generating.locked() else {},
            }
        )

    async def list_voices(self, _: web.Request) -> web.Response:
        """List saved voices."""
        return web.json_response({"voices": self.library.voices()})

    async def add_voice(self, request: web.Request) -> web.Response:
        """Save a voice from an uploaded reference clip and its transcript."""
        form = await request.post()
        label = str(form.get("label", "")).strip()[:80]
        ref_text = str(form.get("ref_text", "")).strip()
        upload = form.get("ref_audio")
        if not label or not ref_text or not hasattr(upload, "file"):
            return _error(400, "send a label, ref_text, and ref_audio")
        try:
            settings = self._settings(form)
            samples = decode_reference(upload.file.read())  # type: ignore[union-attr]
        except ValueError as exc:
            return _error(400, str(exc))
        meta = self.library.add_voice(label, ref_text, samples, {**settings, "source": "upload"})
        return web.json_response(meta, status=201)

    async def voice_from_output(self, request: web.Request) -> web.Response:
        """Save the first chunk of a generated output as a voice.

        A designed voice changes on every request, so saving a sample of it as a
        reference clip is how it stays the same across a long narration.
        """
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "send a JSON body")
        meta = self.library.output(str(body.get("output_id", "")))
        label = str(body.get("label", "")).strip()[:80]
        if meta is None or not label:
            return _error(400, "send a known output_id and a label")
        first = meta["chunks"][0]
        start, end = int(first["start_s"] * SAMPLE_RATE), int(first["end_s"] * SAMPLE_RATE)
        samples, _ = sf.read(
            str(self.library.outputs_dir / f"{meta['id']}.wav"),
            start=start,
            stop=end,
            dtype="float32",
        )
        low, high = REFERENCE_SECONDS
        if not low <= len(samples) / SAMPLE_RATE <= high:
            return _error(
                400,
                f"the first chunk is {len(samples) / SAMPLE_RATE:.1f} s; "
                f"generate a sample of {low:.0f} to {high:.0f} s (about two sentences) to save",
            )
        settings = {
            "instruction": meta.get("instruction", ""),
            "cfg_scale": meta.get("cfg_scale", 1.0),
            "seed": meta.get("seed", 42),
            "source": "output",
        }
        saved = self.library.add_voice(label, first["text"], samples, settings)
        return web.json_response(saved, status=201)

    async def delete_voice(self, request: web.Request) -> web.Response:
        """Delete a saved voice."""
        if not self.library.delete_voice(request.match_info["voice_id"]):
            return _error(404, "no such voice")
        return web.json_response({"status": "deleted"})

    async def voice_audio(self, request: web.Request) -> web.StreamResponse:
        """Serve a voice's reference clip."""
        voice_id = request.match_info["voice_id"]
        if self.library.voice(voice_id) is None:
            return _error(404, "no such voice")
        return web.FileResponse(self.library.reference_path(voice_id))

    async def list_outputs(self, _: web.Request) -> web.Response:
        """List saved audio."""
        return web.json_response({"outputs": self.library.outputs()})

    async def delete_output(self, request: web.Request) -> web.Response:
        """Delete saved audio."""
        if not self.library.delete_output(request.match_info["output_id"]):
            return _error(404, "no such output")
        return web.json_response({"status": "deleted"})

    async def output_audio(self, request: web.Request) -> web.StreamResponse:
        """Serve saved audio as WAV, M4A (AAC), or MP3, encoding a copy the first time."""
        output_id = request.match_info["output_id"]
        meta = self.library.output(output_id)
        fmt = request.query.get("format", "wav")
        if meta is None or fmt not in AUDIO_FORMATS:
            return _error(404, "no such output or format")
        content_type, extension = AUDIO_FORMATS[fmt]
        source = self.library.outputs_dir / f"{output_id}.wav"
        target = self.library.outputs_dir / f"{output_id}.{extension}"
        if not target.exists():
            try:
                await asyncio.to_thread(encode_audio, source, target, fmt, meta["title"])
            except (ValueError, OSError, FFmpegError) as exc:
                return _error(500, f"encoding failed: {exc}")
        slug = re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")[:60] or "narration"
        return web.FileResponse(
            target,
            headers={
                "Content-Type": content_type,
                "Content-Disposition": f'inline; filename="{slug}.{extension}"',
            },
        )

    def _settings(self, form: Any) -> dict[str, Any]:
        """Check the instruction, guidance, and seed fields that voices and generation share.

        Raises
        ------
        ValueError
            If a field is malformed, or guidance above 1 has no instruction to follow.
        """
        instruction = str(form.get("instruction", "")).strip()
        try:
            cfg_scale = float(form.get("cfg_scale", 1.0) or 1.0)
            seed = int(form.get("seed", 42) or 42)
        except ValueError as exc:
            raise ValueError("cfg_scale must be a number and seed an integer") from exc
        if len(instruction) > 1000:
            raise ValueError("the instruction is limited to 1000 characters")
        if not 1.0 <= cfg_scale <= 10.0:
            raise ValueError("cfg_scale must be between 1 and 10")
        if not 0 <= seed < 2**31 - 10_000:
            raise ValueError("seed must be a non-negative integer below 2147473647")
        if cfg_scale > 1.0 and not instruction:
            raise ValueError("cfg_scale above 1 needs an instruction to follow")
        return {"instruction": instruction, "cfg_scale": cfg_scale, "seed": seed}

    async def generate(self, request: web.Request) -> web.StreamResponse:
        """Speak the text, streaming 24 kHz mono signed 16-bit PCM and saving the result.

        The response is raw PCM; its headers give the sample rate, chunk count,
        total characters, and the id the saved audio will have. A failure after
        streaming starts closes the connection, so the client sees a truncated
        stream instead of a clean end.
        """
        form = await request.post()
        text = str(form.get("text", "")).strip()
        if not text or len(text) > MAX_TEXT_CHARS:
            return _error(400, f"send 1 to {MAX_TEXT_CHARS} characters of text")
        try:
            settings = self._settings(form)
        except ValueError as exc:
            return _error(400, str(exc))
        voice_id = str(form.get("voice_id", ""))
        voice = self.library.voice(voice_id) if voice_id else None
        if voice_id and voice is None:
            return _error(400, "unknown voice_id")
        if self.generating.locked():
            return _error(409, "Breeze is already generating; wait for it or stop it first")

        async with self.generating:
            self.child.inflight += 1
            try:
                return await self._stream(request, text, settings, voice)
            finally:
                self.child.inflight -= 1
                self.progress = {}

    async def _speak(
        self,
        piece: str,
        seed: int,
        settings: dict[str, Any],
        voice: dict[str, Any] | None,
    ) -> Any:
        """Open Breeze's streaming response for one chunk, retrying while it is busy.

        Raises
        ------
        ChunkError
            If Breeze stays busy or answers with an error.
        """
        reference = self.library.reference_path(voice["id"]).read_bytes() if voice else b""
        for attempt in range(SPEAK_ATTEMPTS):
            data = FormData()
            data.add_field("text", piece)
            data.add_field("seed", str(seed))
            if settings["instruction"]:
                data.add_field("instruction", settings["instruction"])
                data.add_field("cfg_scale", str(settings["cfg_scale"]))
            if voice:
                data.add_field("ref_text", voice["ref_text"])
                data.add_field("ref_audio", reference, filename="ref.wav", content_type="audio/wav")
            response = await self.session.post(self.child.url + "/v1/audio/speech", data=data)
            if response.status == 200:
                return response
            detail = (await response.text())[:200]
            response.release()
            if response.status != 409:
                raise ChunkError(f"Breeze returned HTTP {response.status}: {detail}")
            await asyncio.sleep(1 + attempt)
        raise ChunkError("Breeze stayed busy with an earlier request")

    async def _stream(
        self,
        request: web.Request,
        text: str,
        settings: dict[str, Any],
        voice: dict[str, Any] | None,
    ) -> web.StreamResponse:
        """Start Breeze, then speak each chunk to the client and the saved WAV file."""
        try:
            await self.child.ensure(self.session)
        except RuntimeError as exc:
            return _error(502, str(exc))

        chunks = split_text(text)
        output_id = secrets.token_hex(16)
        response = web.StreamResponse(
            headers={
                "Content-Type": "audio/pcm",
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Sample-Format": "s16le",
                "X-Chunk-Count": str(len(chunks)),
                "X-Total-Chars": str(len(text)),
                "X-Output-Id": output_id,
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)

        partial = self.library.outputs_dir / f"{output_id}.wav.part"
        began = time.monotonic()
        spoken: list[dict[str, Any]] = []
        warnings: list[str] = []
        written = 0
        with wave.open(str(partial), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            try:
                for index, (piece, pause) in enumerate(chunks):
                    self.progress = {"chunk": index + 1, "chunks": len(chunks)}
                    upstream = await self._speak(piece, settings["seed"] + index, settings, voice)
                    start, size, carry = written, 0, b""
                    try:
                        async for block in upstream.content.iter_chunked(16384):
                            buffer = carry + block
                            usable = len(buffer) & ~1
                            carry, buffer = buffer[usable:], buffer[:usable]
                            await response.write(buffer)
                            wav.writeframes(buffer)
                            size += len(buffer)
                    finally:
                        upstream.release()
                    frames = size / FRAME_BYTES
                    if frames >= CAP_FRAMES:
                        warnings.append(f"chunk {index + 1} reached Breeze's length cap")
                    silence = bytes(int(pause * SAMPLE_RATE) * 2)
                    if silence:
                        await response.write(silence)
                        wav.writeframes(silence)
                    written += size + len(silence)
                    spoken.append(
                        {
                            "text": piece,
                            "start_s": round(start / 2 / SAMPLE_RATE, 2),
                            "end_s": round((start + size) / 2 / SAMPLE_RATE, 2),
                            "seed": settings["seed"] + index,
                        }
                    )
            except (ChunkError, ClientError, TimeoutError) as exc:
                log.warning("generation failed: %s", exc)
                partial.unlink(missing_ok=True)
                if request.transport is not None:
                    request.transport.close()
                return response
            except (ConnectionResetError, asyncio.CancelledError):
                partial.unlink(missing_ok=True)
                raise

        partial.replace(self.library.outputs_dir / f"{output_id}.wav")
        elapsed = time.monotonic() - began
        duration = written / 2 / SAMPLE_RATE
        _json_write(
            self.library.outputs_dir / f"{output_id}.json",
            {
                "id": output_id,
                "created": time.time(),
                "title": re.sub(r"\s+", " ", text)[:60],
                "text": text,
                "voice_id": voice["id"] if voice else "",
                "voice_label": voice["label"] if voice else "",
                **settings,
                "duration_s": round(duration, 1),
                "generation_s": round(elapsed, 1),
                "rtf": round(elapsed / duration, 2) if duration else None,
                "chunks": spoken,
                "warnings": warnings,
            },
        )
        await response.write_eof()
        return response


def build_app(child: Child, library: Library, session: ClientSession) -> web.Application:
    """Create the supervisor application.

    Parameters
    ----------
    child : Child
        The Breeze process to supervise.
    library : Library
        Saved voices and audio.
    session : ClientSession
        Client session for requests to Breeze.
    """
    app = web.Application(client_max_size=64 * 1024**2, middlewares=[guard])
    app["library"] = library
    app.add_routes(Supervisor(child, library, session).routes())

    async def stop_child(_: web.Application) -> None:
        child.inflight = 0
        await child.stop()

    app.on_cleanup.append(stop_child)
    return app


def main() -> None:
    """Serve on port 8080 and start Breeze with the options given on the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        command = child_command(sys.argv[1:], CHILD_PORT)
    except ValueError as exc:
        sys.exit(f"Cannot start Breeze: {exc}")
    library = Library(Path(os.environ.get("SPEECH_DATA", "/data")))

    async def factory() -> web.Application:
        session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
        app = build_app(Child(command, CHILD_PORT), library, session)

        async def close_session(_: web.Application) -> None:
            await session.close()

        app.on_cleanup.append(close_session)
        return app

    web.run_app(factory(), host="0.0.0.0", port=PUBLIC_PORT)  # noqa: S104 (compose network only)


if __name__ == "__main__":
    main()
