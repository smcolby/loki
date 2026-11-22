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
import hashlib
import io
import logging
import math
import os
import re
import secrets
import signal
import sys
import time
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import av
import numpy as np
import soundfile as sf
import soxr
from aiohttp import ClientError, ClientSession, ClientTimeout, FormData, web
from av.audio.stream import AudioStream
from av.error import FFmpegError
from narration import (
    ID_PATTERN,
    MAX_CHAPTERS,
    MAX_JOB_CHARS,
    SAMPLE_RATE,
    JobStore,
    check_chunk,
    length_error,
    new_job,
    read_json,
    split_chapters,
    write_json,
)
from textimport import ImportFailureError, Options, import_document

log = logging.getLogger("speech")

PUBLIC_PORT = 8080
CHILD_PORT = 8081
START_TIMEOUT = 600.0
OWNED_FLAGS = ("--host", "--port")
MODEL_ID = "breeze-tts-2"
FRAME_BYTES = 1920 * 2
# Export levels from ACX's audiobook requirements: RMS between -23 and -18 dB, peaks under -3 dB.
# The ceiling is 0.5 dB lower than the limit because AAC and MP3 encoding can overshoot a peak.
TARGET_RMS_DB = -20.0
PEAK_CEILING_DB = -3.5
CAP_FRAMES = 1499
MAX_TEXT_CHARS = 60_000
MAX_IMPORT_BYTES = 50 * 1024**2
CHUNK_CHARS = 500
# A narration job speaks whole paragraphs where it can. Breeze stops at CAP_FRAMES (119.9 s), and
# measured pace ran from 0.066 to 0.089 s per character, so 1200 characters ends at 107 s at worst
JOB_CHUNK_CHARS = 1200
# A line that ends a sentence, optionally followed by a closing quote or bracket
LINE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*$")
# Sentence ends: a period, question mark or exclamation mark, optionally followed by a closing
# quote or bracket, then whitespace and a character that is not another dot of a spaced ellipsis
SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+(?=[^\s.])")
SENTENCE_PAUSE = 0.3
PARAGRAPH_PAUSE = 0.8
SPEAK_ATTEMPTS = 6
REFERENCE_SECONDS = (3.0, 120.0)
# Breeze's README uses 4; the Studio slider stops at 6 because higher values are untested
MAX_CFG_SCALE = 6.0
# A narration job speaks a chunk again with a new seed when it fails the quality check
JOB_ATTEMPTS = 3
RETRY_SEED_STEP = 7919
SEED_MODULUS = 2**31 - 1
AUDIO_FORMATS = {
    "wav": ("audio/wav", "wav"),
    "m4a": ("audio/mp4", "m4a"),
    "mp3": ("audio/mpeg", "mp3"),
}
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
        for sentence in SENTENCE_END.split(paragraph):
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


def _paragraphs(text: str) -> list[str]:
    """Return the paragraphs of ``text``.

    A blank line ends a paragraph, and so does a line that ends a sentence. A line break
    inside a sentence joins the lines, so hard-wrapped text still has whole paragraphs.
    """
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        current = ""
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            current = f"{current} {line}".strip()
            if LINE_END.search(line):
                paragraphs.append(re.sub(r"\s+", " ", current))
                current = ""
        if current:
            paragraphs.append(re.sub(r"\s+", " ", current))
    return paragraphs


def _balanced(parts: list[str], limit: int) -> list[str]:
    """Join consecutive parts into the fewest pieces of at most ``limit`` characters.

    Each part must fit the limit. Among the ways to make the fewest pieces, the one with the
    smallest sum of squared piece lengths wins, which makes the lengths as equal as possible.
    """
    # Count the fewest pieces with a greedy pass
    count, size = 1, len(parts[0])
    for part in parts[1:]:
        if size + 1 + len(part) > limit:
            count, size = count + 1, len(part)
        else:
            size += 1 + len(part)

    # Find the cut points that minimize the sum of squared lengths for that count
    prefix = [0]
    for part in parts:
        prefix.append(prefix[-1] + len(part))

    def width(start: int, end: int) -> int:
        return prefix[end] - prefix[start] + (end - start - 1)

    cost = [[math.inf] * (len(parts) + 1) for _ in range(count + 1)]
    cut = [[0] * (len(parts) + 1) for _ in range(count + 1)]
    cost[0][0] = 0.0
    for pieces in range(1, count + 1):
        for end in range(pieces, len(parts) + 1):
            for start in range(pieces - 1, end):
                if width(start, end) <= limit and cost[pieces - 1][start] < math.inf:
                    total = cost[pieces - 1][start] + width(start, end) ** 2
                    if total < cost[pieces][end]:
                        cost[pieces][end], cut[pieces][end] = total, start

    # Walk the cut points back from the end
    joined: list[str] = []
    end = len(parts)
    for pieces in range(count, 0, -1):
        start = cut[pieces][end]
        joined.insert(0, " ".join(parts[start:end]))
        end = start
    return joined


def plan_chunks(text: str, limit: int = JOB_CHUNK_CHARS) -> list[tuple[str, float]]:
    r"""Cut text into chunks that follow its paragraphs, with the pause after each.

    Neighboring paragraphs that fit together within ``limit`` share a chunk. A paragraph
    over the limit is cut at sentence boundaries into the fewest pieces, of about equal
    length, and a sentence over the limit is cut at a comma or space. The pieces of a cut
    paragraph are not joined with its neighbors.

    Parameters
    ----------
    text : str
        Text to speak. A blank line or a line that ends a sentence ends a paragraph.
    limit : int, optional
        Most characters in one chunk. Default is ``JOB_CHUNK_CHARS``.

    Returns
    -------
    list of tuple of (str, float)
        Each chunk and the seconds of silence to add after it: a paragraph pause where
        the chunk ends a paragraph, a sentence pause where it ends inside one, and none
        after the last chunk.

    Examples
    --------
    >>> plan_chunks("One.\n\nTwo.", limit=20)
    [('One. Two.', 0.0)]
    >>> plan_chunks("One.\n\nTwo.", limit=5)
    [('One.', 0.8), ('Two.', 0.0)]
    """
    # Make one unit per paragraph, or per piece of a paragraph over the limit
    units: list[tuple[str, bool, bool]] = []
    for paragraph in _paragraphs(text):
        if len(paragraph) <= limit:
            units.append((paragraph, True, True))
            continue
        parts = [
            part for sentence in SENTENCE_END.split(paragraph) for part in _wrap(sentence, limit)
        ]
        pieces = _balanced(parts, limit)
        units += [(piece, index == len(pieces) - 1, False) for index, piece in enumerate(pieces)]

    # Join whole paragraphs that fit together
    chunks: list[tuple[str, bool, bool]] = []
    for unit, ends_paragraph, whole in units:
        if chunks and whole and chunks[-1][2] and len(chunks[-1][0]) + 1 + len(unit) <= limit:
            chunks[-1] = (f"{chunks[-1][0]} {unit}", True, True)
        else:
            chunks.append((unit, ends_paragraph, whole))

    planned = [(chunk, PARAGRAPH_PAUSE if ends else SENTENCE_PAUSE) for chunk, ends, _ in chunks]
    if planned:
        planned[-1] = (planned[-1][0], 0.0)
    return planned


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


def encode_audio(
    source: Path,
    target: Path,
    fmt: str,
    title: str,
    album: str = "loki",
    track: tuple[int, int] | None = None,
) -> None:
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
    album : str, optional
        Album tag; a job's chapters share the job title so a player groups them.
    track : tuple of int, optional
        Chapter number and chapter count, written as the track tag so a player orders them.

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
    tags = {"title": title, "artist": "Breeze TTS 2 (synthetic speech)", "album": album}
    if track:
        tags["track"] = f"{track[0]}/{track[1]}"
    output.metadata.update(tags)
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


def zip_member_name(number: int, width: int, title: str, extension: str) -> str:
    """Return a safe filename for a chapter in a zip: its number, then its title.

    Parameters
    ----------
    number : int
        Chapter number, counted from 1.
    width : int
        Digits to pad the number to, so the files sort in chapter order.
    title : str
        Chapter title; characters that file systems reject are replaced with spaces.
    extension : str
        File extension without the dot.

    Examples
    --------
    >>> zip_member_name(3, 2, 'Why: a "test"?', "m4a")
    '03 Why a test.m4a'
    """
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip(" .")[:100] or "Chapter"
    return f"{number:0{width}d} {clean}.{extension}"


def write_zip(path: Path, members: list[tuple[Path, str]], missing: list[str]) -> None:
    """Write audio files into a zip without compressing them, and list unfinished chapters.

    Parameters
    ----------
    path : Path
        Zip file to write.
    members : list of tuple
        Each source file with its filename inside the zip.
    missing : list of str
        Chapters that have no audio yet; they are listed in ``MISSING.txt`` when any.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for source, arcname in members:
            archive.write(source, arcname)
        if missing:
            listing = "These chapters have no audio yet:\n" + "\n".join(missing) + "\n"
            archive.writestr("MISSING.txt", listing)


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
        self.root = root
        self.voices_dir = root / "voices"
        self.outputs_dir = root / "outputs"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Clear zip files left by a download that was cut off by a restart
        self.tmp_dir = root / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        for stale in self.tmp_dir.iterdir():
            stale.unlink(missing_ok=True)

    def voice(self, voice_id: str) -> dict[str, Any] | None:
        """Return a voice's metadata, or None for an unknown or malformed id."""
        if not ID_PATTERN.fullmatch(voice_id):
            return None
        return read_json(self.voices_dir / voice_id / "meta.json")

    def voices(self) -> list[dict[str, Any]]:
        """Return every voice, newest first."""
        found = [read_json(path / "meta.json") for path in self.voices_dir.iterdir()]
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
        write_json(folder / "meta.json", meta)
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
        return read_json(self.outputs_dir / f"{output_id}.json")

    def outputs(self) -> list[dict[str, Any]]:
        """Return every saved output, newest first."""
        found = [read_json(path) for path in self.outputs_dir.glob("*.json")]
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
        self.jobs = JobStore(library.root)
        self.running: dict[str, Any] | None = None
        self.encoding: dict[str, asyncio.Lock] = {}
        self.jobs.interrupt_running()

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
            web.get("/api/jobs", self.list_jobs),
            web.post("/api/import", self.import_file),
            web.post("/api/jobs", self.create_job),
            web.get("/api/jobs/{job_id}", self.get_job),
            web.get("/api/jobs/{job_id}/zip", self.job_zip),
            web.delete("/api/jobs/{job_id}", self.delete_job),
            web.post("/api/jobs/{job_id}/start", self.start_job),
            web.post("/api/jobs/{job_id}/stop", self.stop_job),
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
                "job": self.running["id"] if self.running else "",
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
        try:
            target = await self._ensure_audio(output_id, meta, fmt)
        except (ValueError, OSError, FFmpegError) as exc:
            return _error(500, f"encoding failed: {exc}")
        slug = re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")[:120] or "narration"
        return web.FileResponse(
            target,
            headers={
                "Content-Type": content_type,
                "Content-Disposition": f'inline; filename="{slug}.{extension}"',
            },
        )

    async def _ensure_audio(self, output_id: str, meta: dict[str, Any], fmt: str) -> Path:
        """Return the file for an output in a format, encoding it first when no copy exists.

        Two requests for the same copy wait for one encode, so they do not write the same
        partial file. A chapter of a job has its job title as the album and its number
        as the track.

        Raises
        ------
        ValueError
            If ``fmt`` cannot be encoded.
        OSError
            If the copy cannot be written.
        FFmpegError
            If encoding fails.
        """
        source = self.library.outputs_dir / f"{output_id}.wav"
        target = self.library.outputs_dir / f"{output_id}.{AUDIO_FORMATS[fmt][1]}"
        async with self.encoding.setdefault(output_id, asyncio.Lock()):
            if not target.exists():
                track = (meta["track"], meta["tracks"]) if "track" in meta else None
                album = meta.get("album", "loki")
                await asyncio.to_thread(
                    encode_audio, source, target, fmt, meta["title"], album, track
                )
        return target

    async def job_zip(self, request: web.Request) -> web.StreamResponse:
        """Download a job's finished chapters as one zip with a numbered file per chapter.

        The format is ``m4a`` (the default), ``mp3`` or ``wav``. Chapters without audio are
        listed in ``MISSING.txt`` inside the zip. The zip is built in a temporary file and
        removed after it is sent.
        """
        job = self.jobs.job(request.match_info["job_id"])
        fmt = request.query.get("format", "m4a")
        if job is None or fmt not in AUDIO_FORMATS:
            return _error(404, "no such job or format")
        extension = AUDIO_FORMATS[fmt][1]
        width = max(2, len(str(len(job["chapters"]))))
        members: list[tuple[Path, str]] = []
        missing: list[str] = []
        for number, chapter in enumerate(job["chapters"], start=1):
            meta = self.library.output(chapter["output_id"]) if chapter["output_id"] else None
            if meta is None:
                missing.append(f"{number:0{width}d} {chapter['title']}")
                continue
            try:
                path = await self._ensure_audio(chapter["output_id"], meta, fmt)
            except (ValueError, OSError, FFmpegError) as exc:
                return _error(500, f"encoding failed: {exc}")
            members.append((path, zip_member_name(number, width, chapter["title"], extension)))
        if not members:
            return _error(409, "no chapter of this job is finished yet")

        # Build the zip in a temporary file, send it in blocks, and remove it afterward
        archive = self.library.tmp_dir / f"{secrets.token_hex(8)}.zip"
        slug = re.sub(r"[^a-z0-9]+", "-", job["title"].lower()).strip("-")[:120] or "narration"
        try:
            await asyncio.to_thread(write_zip, archive, members, missing)
            response = web.StreamResponse(
                headers={
                    "Content-Type": "application/zip",
                    "Content-Length": str(archive.stat().st_size),
                    "Content-Disposition": (
                        f'attachment; filename="{slug}.zip"; '
                        f"filename*=UTF-8''{quote(job['title'])}.zip"
                    ),
                }
            )
            await response.prepare(request)
            with archive.open("rb") as stream:
                while block := await asyncio.to_thread(stream.read, 1 << 20):
                    await response.write(block)
            await response.write_eof()
            return response
        finally:
            archive.unlink(missing_ok=True)

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
        if not 1.0 <= cfg_scale <= MAX_CFG_SCALE:
            raise ValueError("cfg_scale must be between 1 and 6")
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
        write_json(
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

    # ---- narration jobs ----

    async def list_jobs(self, _: web.Request) -> web.Response:
        """List narration jobs, newest first."""
        return web.json_response({"jobs": self.jobs.jobs()})

    async def get_job(self, request: web.Request) -> web.Response:
        """Return one job's record with its chapters and progress."""
        job = self.jobs.job(request.match_info["job_id"])
        if job is None:
            return _error(404, "no such job")
        return web.json_response(job)

    async def import_file(self, request: web.Request) -> web.Response:
        """Turn an uploaded EPUB, text, or Markdown file into chapter-format text for review.

        The multipart form has the file as ``file``, and the options ``citations``,
        ``author_year``, ``links`` and ``captions`` as ``1`` or ``0``. The response is
        ``title``, ``text``, ``chapters`` (title and size), and ``warnings`` (what was
        removed, with counts). Nothing is stored: the Studio shows the text for editing,
        and a job is created from whatever text it then sends.
        """
        form = await request.post()
        upload = form.get("file")
        if not hasattr(upload, "file"):
            return _error(400, "send the file as the multipart field 'file'")
        data = upload.file.read(MAX_IMPORT_BYTES + 1)  # type: ignore[union-attr]
        if len(data) > MAX_IMPORT_BYTES:
            return _error(413, f"the file is over {MAX_IMPORT_BYTES // 1024**2} MB")

        def flag(field: str, default: bool) -> bool:
            value = form.get(field)
            return default if value is None else str(value).lower() not in ("0", "false", "off", "")

        options = Options(
            citations=flag("citations", True),
            author_year=flag("author_year", True),
            links=flag("links", True),
            captions=flag("captions", False),
        )
        filename = str(getattr(upload, "filename", "") or "")

        # Parse off the event loop so a large book does not stall status requests or a running job
        try:
            book = await asyncio.to_thread(import_document, data, filename, options)
        except ImportFailureError as exc:
            return _error(400, str(exc))
        return web.json_response(book.as_dict())

    async def create_job(self, request: web.Request) -> web.Response:
        """Save a narration job for long text; it starts when asked and does not need the GPU.

        The JSON body has ``text``, where lines that start with ``#`` begin chapters, and
        optional ``title``, ``voice_id``, ``instruction``, ``cfg_scale``, ``seed``, and
        ``speak_titles`` (default true, which reads each heading before its chapter).
        """
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "send a JSON body")
        if not isinstance(body, dict):
            return _error(400, "send a JSON object")
        text = str(body.get("text", "")).strip()
        if not text or len(text) > MAX_JOB_CHARS:
            return _error(400, f"send 1 to {MAX_JOB_CHARS:,} characters of text")
        try:
            settings = self._settings(body)
        except ValueError as exc:
            return _error(400, str(exc))
        voice_id = str(body.get("voice_id", ""))
        voice = self.library.voice(voice_id) if voice_id else None
        if voice_id and voice is None:
            return _error(400, "unknown voice_id")
        speak_titles = body.get("speak_titles", True)
        if not isinstance(speak_titles, bool):
            return _error(400, "speak_titles must be true or false")

        title = re.sub(r"\s+", " ", str(body.get("title", ""))).strip()[:80]
        chapters = split_chapters(text, title or "Narration")
        if not chapters or len(chapters) > MAX_CHAPTERS:
            return _error(400, f"the text must have 1 to {MAX_CHAPTERS} chapters with text")
        title = title or re.sub(r"\s+", " ", chapters[0].title)[:80]
        plan: list[list[tuple[str, float]]] = []
        for chapter in chapters:
            chunks = plan_chunks(chapter.body)
            if speak_titles and chapter.headed:
                chunks.insert(0, (re.sub(r"\s+", " ", chapter.title)[:60], PARAGRAPH_PAUSE))
            plan.append(chunks)
        titles = [re.sub(r"\s+", " ", chapter.title)[:60] for chapter in chapters]
        job = new_job(title, list(zip(titles, plan, strict=True)), voice, settings, speak_titles)
        self.jobs.create(job, plan)
        return web.json_response(job, status=201)

    async def delete_job(self, request: web.Request) -> web.Response:
        """Delete a job and its unfinished chunks; finished chapters stay in the library."""
        job_id = request.match_info["job_id"]
        if self.running and self.running["id"] == job_id:
            return _error(409, "stop the job before deleting it")
        if not self.jobs.delete(job_id):
            return _error(404, "no such job")
        return web.json_response({"status": "deleted"})

    async def start_job(self, request: web.Request) -> web.Response:
        """Start or resume a job in the background; it continues after this request ends."""
        job = self.jobs.job(request.match_info["job_id"])
        if job is None:
            return _error(404, "no such job")
        if job["state"] == "done":
            return _error(409, "the job is finished")
        if job["voice_id"] and self.library.voice(job["voice_id"]) is None:
            return _error(409, "the voice this job uses was deleted")
        if self.generating.locked():
            return _error(409, "Breeze is already generating; wait for it or stop it first")

        await self.generating.acquire()
        self.child.inflight += 1
        self.running = {"id": job["id"], "reason": "interrupted", "task": None}
        self.running["task"] = asyncio.create_task(self._run_job(job["id"]))
        job["state"] = "running"
        return web.json_response(job, status=202)

    async def stop_job(self, request: web.Request) -> web.Response:
        """Stop the running job after discarding the part in progress; finished parts stay saved."""
        job_id = request.match_info["job_id"]
        run = self.running
        if run is None or run["id"] != job_id:
            return _error(409, "the job is not running")
        run["reason"] = "paused"
        run["task"].cancel()
        await asyncio.wait({run["task"]})
        return web.json_response(self.jobs.job(job_id))

    async def shutdown(self) -> None:
        """Stop a running job so its record says it was interrupted and can resume."""
        run = self.running
        if run is not None and not run["task"].done():
            run["reason"] = "interrupted"
            run["task"].cancel()
            await asyncio.wait({run["task"]})

    def _recount(self, job: dict[str, Any], plan: list[list[tuple[str, float]]]) -> None:
        """Rebuild a job's progress counts from the chunk files on disk."""
        job["chunks_done"], job["chars_done"], job["speech_s"] = 0, 0, 0.0
        for number, (chunks, chapter) in enumerate(zip(plan, job["chapters"], strict=True)):
            if chapter["output_id"]:
                done, chars, speech = chapter["chunks_total"], chapter["chars"], chapter["speech_s"]
            else:
                paths = [self.jobs.chunk_path(job["id"], number, i) for i in range(len(chunks))]
                pairs = zip(chunks, paths, strict=True)
                spoken = [(text, path) for (text, _), path in pairs if path.exists()]
                done, chars = len(spoken), sum(len(text) for text, _ in spoken)
                speech = sum(path.stat().st_size for _, path in spoken) / 2 / SAMPLE_RATE
            chapter["chunks_done"], chapter["speech_s"] = done, round(speech, 2)
            job["chunks_done"] += done
            job["chars_done"] += chars
            job["speech_s"] = round(job["speech_s"] + speech, 2)

    async def _speak_checked(
        self, text: str, index: int, settings: dict[str, Any], voice: dict[str, Any] | None
    ) -> tuple[bytes, str | None, list[str]]:
        """Speak one chunk, speaking it again with a new seed while it fails the quality check.

        Returns
        -------
        tuple of (bytes, str or None, list of str)
            The 16-bit PCM, a warning when every attempt failed the check (the PCM is then
            the attempt closest to the expected length), and why each failed attempt failed.

        Raises
        ------
        ChunkError
            If Breeze returned an error on every attempt.
        RuntimeError
            If Breeze cannot be restarted after it exited.
        """
        best: tuple[float, bytes, str] | None = None
        failure = ""
        failed: list[str] = []
        for attempt in range(JOB_ATTEMPTS):
            seed = (settings["seed"] + index + attempt * RETRY_SEED_STEP) % SEED_MODULUS
            try:
                upstream = await self._speak(text, seed, settings, voice)
                try:
                    pcm = await upstream.read()
                finally:
                    upstream.release()
            except (ChunkError, ClientError, TimeoutError) as exc:
                failure = str(exc) or type(exc).__name__
                log.warning("attempt %d at a chunk failed: %s", attempt + 1, failure)
                failed.append(f"attempt {attempt + 1}: Breeze error, {failure}")
                await self.child.ensure(self.session)
                continue
            pcm = pcm[: len(pcm) & ~1]
            samples = np.frombuffer(pcm, dtype="<i2")
            reason = check_chunk(samples, text, hit_cap=len(pcm) / FRAME_BYTES >= CAP_FRAMES)
            if reason is None:
                return pcm, None, failed
            log.warning("attempt %d at a chunk was %s", attempt + 1, reason)
            failed.append(f"attempt {attempt + 1}: {reason}")
            error = length_error(samples, text)
            if best is None or error < best[0]:
                best = (error, pcm, reason)
        if best is None:
            raise ChunkError(f"Breeze failed {JOB_ATTEMPTS} times: {failure}")
        return best[1], f"{best[2]} on all {JOB_ATTEMPTS} attempts; used the closest", failed

    def _stitch_chapter(
        self, job: dict[str, Any], number: int, chunks: list[tuple[str, float]]
    ) -> str:
        """Join a chapter's chunks and pauses into a library output and return its id.

        The id comes from the job and chapter, so stitching again after an interruption
        replaces the same output instead of adding a second one.
        """
        chapter = job["chapters"][number]
        output_id = hashlib.sha256(f"{job['id']}:{number}".encode()).hexdigest()[:32]
        self.library.delete_output(output_id)
        partial = self.library.outputs_dir / f"{output_id}.wav.part"
        spoken: list[dict[str, Any]] = []
        written = 0
        with wave.open(str(partial), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            for index, (text, pause) in enumerate(chunks):
                pcm = self.jobs.chunk_path(job["id"], number, index).read_bytes()
                silence = bytes(int(pause * SAMPLE_RATE) * 2)
                wav.writeframes(pcm + silence)
                spoken.append(
                    {
                        "text": text,
                        "start_s": round(written / 2 / SAMPLE_RATE, 2),
                        "end_s": round((written + len(pcm)) / 2 / SAMPLE_RATE, 2),
                    }
                )
                written += len(pcm) + len(silence)
        partial.replace(self.library.outputs_dir / f"{output_id}.wav")
        duration = written / 2 / SAMPLE_RATE
        title = job["title"]
        if len(job["chapters"]) > 1:
            title = f"{job['title']} - {number + 1:02d} {chapter['title']}"
        write_json(
            self.library.outputs_dir / f"{output_id}.json",
            {
                "id": output_id,
                "created": time.time(),
                "title": title,
                "text": " ".join(text for text, _ in chunks),
                "voice_id": job["voice_id"],
                "voice_label": job["voice_label"],
                "instruction": job["instruction"],
                "cfg_scale": job["cfg_scale"],
                "seed": job["seed"],
                "duration_s": round(duration, 1),
                "generation_s": round(chapter["generation_s"], 1),
                "rtf": round(chapter["generation_s"] / duration, 2) if duration else None,
                "chunks": spoken,
                "warnings": chapter["warnings"],
                "job_id": job["id"],
                "chapter": number + 1,
                "album": job["title"],
                "track": number + 1,
                "tracks": len(job["chapters"]),
            },
        )
        return output_id

    async def _run_job(self, job_id: str) -> None:
        """Speak every chapter of a job that has no saved audio, saving each chunk as it is made.

        Chunks that already exist on disk are skipped, so a job resumes where it stopped.
        The job's state is ``done``, ``failed`` with an error message, or the reason the
        task was cancelled (``paused`` by the user, ``interrupted`` by a shutdown).
        """
        run = cast(dict[str, Any], self.running)
        job = cast(dict[str, Any], self.jobs.job(job_id))
        job["state"], job["error"] = "running", ""
        try:
            plan = self.jobs.plan(job_id)
            self._recount(job, plan)
            self.jobs.save(job)
            voice = self.library.voice(job["voice_id"]) if job["voice_id"] else None
            settings = {key: job[key] for key in ("instruction", "cfg_scale", "seed")}
            await self.child.ensure(self.session)

            # Speak the chunks of each chapter that has no saved audio, then stitch it
            for number, chunks in enumerate(plan):
                chapter = job["chapters"][number]
                if chapter["output_id"]:
                    continue
                chapter["state"] = "running"
                for index, (text, _) in enumerate(chunks):
                    if self.jobs.chunk_path(job_id, number, index).exists():
                        continue
                    self.progress = {
                        "chunk": job["chunks_done"] + 1,
                        "chunks": job["chunks_total"],
                        "chapter": number + 1,
                        "chapters": len(plan),
                    }
                    started = time.monotonic()
                    pcm, warning, failed = await self._speak_checked(text, index, settings, voice)
                    self.jobs.write_chunk(job_id, number, index, pcm)
                    spent = time.monotonic() - started
                    seconds = len(pcm) / 2 / SAMPLE_RATE
                    for record in (job, chapter):
                        record["chunks_done"] += 1
                        record["speech_s"] = round(record["speech_s"] + seconds, 2)
                        record["generation_s"] = round(record["generation_s"] + spent, 2)
                    job["chars_done"] += len(text)
                    if warning:
                        chapter["warnings"].append(f"part {index + 1}: {warning}")

                    # Record each rejected attempt so a high rate of retries shows in the job
                    if failed:
                        chapter["retries"].extend(f"part {index + 1}, {why}" for why in failed)
                        job["retries"] += len(failed)
                        log.warning(
                            "job %s chapter %d part %d: %d attempt(s) rejected",
                            job_id,
                            number + 1,
                            index + 1,
                            len(failed),
                        )
                    self.jobs.save(job)
                output_id = await asyncio.to_thread(self._stitch_chapter, job, number, chunks)
                chapter["output_id"], chapter["state"] = output_id, "done"
                self.jobs.save(job)
                self.jobs.clear_chunks(job_id, number)
            job["state"] = "done"
        except asyncio.CancelledError:
            job["state"] = run["reason"]
            raise
        except (ChunkError, RuntimeError, OSError) as exc:
            log.warning("job %s failed: %s", job_id, exc)
            job["state"], job["error"] = "failed", str(exc)
        except Exception as exc:
            log.exception("job %s crashed", job_id)
            job["state"], job["error"] = "failed", f"internal error: {exc}"
            raise
        finally:
            for chapter in job["chapters"]:
                if chapter["state"] == "running":
                    chapter["state"] = "pending"
            self.jobs.save(job)
            self.running = None
            self.progress = {}
            self.child.inflight -= 1
            self.generating.release()


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
    supervisor = Supervisor(child, library, session)
    app.add_routes(supervisor.routes())

    async def stop_child(_: web.Application) -> None:
        await supervisor.shutdown()
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
