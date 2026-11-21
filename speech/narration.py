"""Narration jobs for the speech supervisor: chapters, a checkpointed plan, and a quality gate.

A job turns long text (a paper or a book) into one audio file per chapter. The text is
split into chunks Breeze can speak in one request, and every finished chunk is written to
disk. A job stopped by a restart, an unload, or a crash resumes at the first chunk that has
no file. When all chunks of a chapter exist they are joined into a library output and the
chunk files are deleted.

This module contains the parts that need no GPU: chapter parsing, the check that rejects a
bad chunk, and the job files. The supervisor runs the jobs.
"""

import json
import math
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

SAMPLE_RATE = 24000
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
HEADING = re.compile(r"^#{1,2}[ \t]+(\S.*?)[ \t]*#*[ \t]*$")
MAX_JOB_CHARS = 2_000_000
MAX_CHAPTERS = 500

# Chunk check limits, from the measured pace of 0.075 s per character (88 to 139 words per minute)
SECONDS_PER_CHAR = 0.075
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5
MIN_RMS_DB = -50.0
QUIET_FRAME_DB = -55.0
MAX_QUIET_SECONDS = 4.0
FRAME_SECONDS = 0.05

STATES = ("ready", "running", "paused", "interrupted", "failed", "done")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON beside its final path, then rename it into place so readers see all or none."""
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, indent=1))
    partial.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the JSON object stored at ``path``, or None when it is missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Chapter(NamedTuple):
    """One chapter of the text: its title, its body, and whether a heading started it."""

    title: str
    body: str
    headed: bool


def split_chapters(text: str, default_title: str) -> list[Chapter]:
    r"""Cut text into chapters at lines that start with ``#`` or ``##``.

    Text before the first heading becomes a chapter called ``default_title``. A heading
    with no text after it is dropped, and text with no headings is one chapter.

    Parameters
    ----------
    text : str
        Text to narrate, with ``# Title`` lines starting chapters.
    default_title : str
        Title for text that comes before any heading, or for text with no headings.

    Returns
    -------
    list of Chapter
        Each chapter in order; ``headed`` is False for the text before the first heading.

    Examples
    --------
    >>> chapters = split_chapters("Intro words.\n# One\nFirst.\n## Two\nSecond.", "Opening")
    >>> [(chapter.title, chapter.headed) for chapter in chapters]
    [('Opening', False), ('One', True), ('Two', True)]
    """
    found: list[tuple[str, bool, list[str]]] = [(default_title, False, [])]
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            found.append((match.group(1), True, []))
        else:
            found[-1][2].append(line)
    chapters = [Chapter(title, "\n".join(lines).strip(), headed) for title, headed, lines in found]
    return [chapter for chapter in chapters if chapter.body]


def _longest_quiet_seconds(samples: np.ndarray) -> float:
    """Return the longest stretch of near-silence in 16-bit samples, in seconds."""
    frame = int(SAMPLE_RATE * FRAME_SECONDS)
    usable = len(samples) // frame * frame
    if not usable:
        return 0.0
    blocks = samples[:usable].astype(np.float32).reshape(-1, frame) / 32768.0
    level_db = 20 * np.log10(np.sqrt(np.square(blocks).mean(axis=1)) + 1e-9)
    longest = run = 0
    for quiet in level_db < QUIET_FRAME_DB:
        run = run + 1 if quiet else 0
        longest = max(longest, run)
    return longest * FRAME_SECONDS


def length_error(samples: np.ndarray, text: str) -> float:
    """Return how far a chunk's length is from the expected length, as an absolute log ratio.

    Parameters
    ----------
    samples : numpy.ndarray
        16-bit mono samples at 24 kHz.
    text : str
        The text that was spoken.

    Returns
    -------
    float
        0.0 for the expected length; 0.69 for half or double it.
    """
    seconds = max(len(samples) / SAMPLE_RATE, 0.05)
    return abs(math.log(seconds / max(len(text) * SECONDS_PER_CHAR, 0.05)))


def check_chunk(samples: np.ndarray, text: str, *, hit_cap: bool) -> str | None:
    """Return why a spoken chunk should be regenerated, or None when it passes.

    Breeze can stop early, repeat itself until its length cap, or fall silent. The
    checks are the length against the text, the loudness, and the longest silence.

    Parameters
    ----------
    samples : numpy.ndarray
        16-bit mono samples at 24 kHz.
    text : str
        The text that was spoken.
    hit_cap : bool
        Whether Breeze stopped at its per-request length cap.

    Returns
    -------
    str or None
        A short reason, or None.

    Examples
    --------
    >>> check_chunk(np.zeros(24000, dtype=np.int16), "Hello there.", hit_cap=False)
    'silent'
    """
    seconds = len(samples) / SAMPLE_RATE
    expected = len(text) * SECONDS_PER_CHAR
    if hit_cap:
        return "reached Breeze's length cap"
    if seconds < max(0.3, MIN_LENGTH_RATIO * expected):
        return f"too short ({seconds:.1f} s for {len(text)} characters)"
    if seconds > MAX_LENGTH_RATIO * expected + 3.0:
        return f"too long ({seconds:.1f} s for {len(text)} characters)"
    energy = float(np.square(samples.astype(np.float64) / 32768.0).mean())
    if energy <= 0 or 10 * math.log10(energy) < MIN_RMS_DB:
        return "silent"
    quiet = _longest_quiet_seconds(samples)
    if quiet > MAX_QUIET_SECONDS:
        return f"{quiet:.1f} s of silence"
    return None


def new_job(
    title: str,
    chapters: list[tuple[str, list[tuple[str, float]]]],
    voice: dict[str, Any] | None,
    settings: dict[str, Any],
    speak_titles: bool,
) -> dict[str, Any]:
    """Build a job record for chapters that are already split into chunks.

    Parameters
    ----------
    title : str
        Job title, used in each chapter's output title.
    chapters : list
        Each chapter's title and its chunks as ``(text, pause_after_seconds)``.
    voice : dict or None
        The saved voice every chunk is cloned from, or None.
    settings : dict
        ``instruction``, ``cfg_scale`` and ``seed``.
    speak_titles : bool
        Whether each chapter's chunks begin with its title.
    """
    now = time.time()
    return {
        "id": secrets.token_hex(16),
        "title": title,
        "created": now,
        "updated": now,
        "state": "ready",
        "error": "",
        "voice_id": voice["id"] if voice else "",
        "voice_label": voice["label"] if voice else "",
        **settings,
        "speak_titles": speak_titles,
        "chars_total": sum(len(text) for _, chunks in chapters for text, _ in chunks),
        "chars_done": 0,
        "chunks_total": sum(len(chunks) for _, chunks in chapters),
        "chunks_done": 0,
        "speech_s": 0.0,
        "generation_s": 0.0,
        "chapters": [
            {
                "title": chapter_title,
                "state": "pending",
                "chars": sum(len(text) for text, _ in chunks),
                "chunks_total": len(chunks),
                "chunks_done": 0,
                "speech_s": 0.0,
                "generation_s": 0.0,
                "output_id": "",
                "warnings": [],
            }
            for chapter_title, chunks in chapters
        ],
    }


class JobStore:
    """Narration jobs on disk: a record, a plan of chunks, and the chunks spoken so far.

    Parameters
    ----------
    root : Path
        Library directory; jobs live under ``jobs/`` in it and are created if missing.
    """

    def __init__(self, root: Path):
        self.dir = root / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _folder(self, job_id: str) -> Path:
        """Return a job's folder; the id must have passed :meth:`job`."""
        return self.dir / job_id

    def job(self, job_id: str) -> dict[str, Any] | None:
        """Return a job record, or None for an unknown or malformed id."""
        if not ID_PATTERN.fullmatch(job_id):
            return None
        return read_json(self._folder(job_id) / "job.json")

    def jobs(self) -> list[dict[str, Any]]:
        """Return every job, newest first."""
        found = [read_json(path / "job.json") for path in self.dir.iterdir() if path.is_dir()]
        return sorted((job for job in found if job), key=lambda job: job["created"], reverse=True)

    def save(self, job: dict[str, Any]) -> None:
        """Write a job record and stamp its update time."""
        job["updated"] = time.time()
        write_json(self._folder(job["id"]) / "job.json", job)

    def create(self, job: dict[str, Any], plan: list[list[tuple[str, float]]]) -> None:
        """Store a new job with the chunks it will speak, chapter by chapter."""
        folder = self._folder(job["id"])
        (folder / "chunks").mkdir(parents=True)
        write_json(folder / "plan.json", {"chapters": plan})
        self.save(job)

    def plan(self, job_id: str) -> list[list[tuple[str, float]]]:
        """Return each chapter's chunks as ``(text, pause_after_seconds)``."""
        stored = read_json(self._folder(job_id) / "plan.json") or {}
        return [[(text, float(pause)) for text, pause in chapter] for chapter in stored["chapters"]]

    def chunk_path(self, job_id: str, chapter: int, index: int) -> Path:
        """Return where a chunk's 16-bit PCM is stored."""
        return self._folder(job_id) / "chunks" / f"{chapter:04d}-{index:05d}.pcm"

    def write_chunk(self, job_id: str, chapter: int, index: int, pcm: bytes) -> None:
        """Store a spoken chunk, whole or not at all."""
        path = self.chunk_path(job_id, chapter, index)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(pcm)
        partial.replace(path)

    def clear_chunks(self, job_id: str, chapter: int) -> None:
        """Delete a chapter's chunk files once its audio is saved."""
        for path in (self._folder(job_id) / "chunks").glob(f"{chapter:04d}-*"):
            path.unlink()

    def delete(self, job_id: str) -> bool:
        """Delete a job and its chunk files; return whether it existed."""
        if self.job(job_id) is None:
            return False
        shutil.rmtree(self._folder(job_id))
        return True

    def interrupt_running(self) -> int:
        """Mark jobs left running by a stopped supervisor as interrupted; return how many."""
        count = 0
        for job in self.jobs():
            if job["state"] == "running":
                job["state"] = "interrupted"
                for chapter in job["chapters"]:
                    if chapter["state"] == "running":
                        chapter["state"] = "pending"
                self.save(job)
                count += 1
        return count
