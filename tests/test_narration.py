"""Tests for speech/narration.py: chapters, the chunk check, and the job files."""

import narration as nar
import numpy as np
import pytest

RATE = nar.SAMPLE_RATE


def tone(seconds: float, level: float = 0.25) -> np.ndarray:
    """Return a 16-bit sine tone of the given length and peak level."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


# ---- chapters ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Just words.", [("Opening", "Just words.", False)]),
        ("# One\nFirst.\n# Two\nSecond.", [("One", "First.", True), ("Two", "Second.", True)]),
        ("Preface.\n## One\nFirst.", [("Opening", "Preface.", False), ("One", "First.", True)]),
        ("# Empty\n# Real\nBody.", [("Real", "Body.", True)]),
        ("# Title #\nBody.", [("Title", "Body.", True)]),
        (
            "#hashtag is text\n### Deeper\nBody.",
            [("Opening", "#hashtag is text\n### Deeper\nBody.", False)],
        ),
        ("  # indented\nBody.", [("Opening", "# indented\nBody.", False)]),
        ("# Only a heading", []),
    ],
    ids=[
        "no headings",
        "two chapters",
        "text before the first heading",
        "heading without text is dropped",
        "closing hashes trimmed",
        "only one or two hashes followed by a space",
        "indented heading is text",
        "no body",
    ],
)
def test_split_chapters_starts_a_chapter_at_each_heading(text, expected):
    """A line that begins with one or two hashes and a space starts a chapter."""
    assert [tuple(chapter) for chapter in nar.split_chapters(text, "Opening")] == expected


def test_split_chapters_keeps_every_body_line_in_order():
    """Chapter bodies together contain every line that is not a heading, in order."""
    body = [f"Line {i}." for i in range(30)]
    text = "\n".join(["# A", *body[:10], "## B", *body[10:20], "# C", *body[20:]])

    chapters = nar.split_chapters(text, "Opening")

    assert "\n".join(chapter.body for chapter in chapters).split("\n") == body


# ---- chunk check ----


@pytest.mark.parametrize(
    ("samples", "text", "hit_cap", "reason"),
    [
        (tone(3.0), "x" * 40, False, None),
        (tone(0.1), "x" * 40, False, "too short"),
        (tone(40.0), "x" * 40, False, "too long"),
        (np.zeros(3 * RATE, dtype=np.int16), "x" * 40, False, "silent"),
        (tone(3.0, level=0.0001), "x" * 40, False, "silent"),
        (tone(3.0), "x" * 40, True, "length cap"),
    ],
    ids=[
        "speech of the expected length",
        "stopped early",
        "ran on",
        "digital silence",
        "inaudible",
        "length cap",
    ],
)
def test_check_chunk_rejects_audio_that_does_not_match_the_text(samples, text, hit_cap, reason):
    """Audio far shorter or longer than its text, silent, or at the cap fails the check."""
    result = nar.check_chunk(samples, text, hit_cap=hit_cap)

    assert (result is None) if reason is None else (reason in str(result))


def test_check_chunk_rejects_a_long_silence_inside_speech():
    """A speech chunk with a silence over four seconds fails even when its length is plausible."""
    samples = np.concatenate([tone(5.0), np.zeros(5 * RATE, dtype=np.int16), tone(5.0)])

    result = nar.check_chunk(samples, "x" * 200, hit_cap=False)

    assert result is not None
    assert "silence" in result


def test_check_chunk_accepts_a_natural_pause():
    """A pause of about a second is ordinary speech."""
    samples = np.concatenate([tone(3.0), np.zeros(RATE, dtype=np.int16), tone(3.0)])

    assert nar.check_chunk(samples, "x" * 80, hit_cap=False) is None


def test_length_error_is_zero_at_the_expected_length_and_grows_with_distance():
    """The error is the absolute log ratio, so half and double the length are equally far."""
    text = "x" * 40
    expected = len(text) * nar.SECONDS_PER_CHAR

    assert nar.length_error(tone(expected), text) == pytest.approx(0.0, abs=1e-3)
    assert nar.length_error(tone(expected / 2), text) == pytest.approx(0.693, abs=1e-2)
    assert nar.length_error(tone(expected * 2), text) == pytest.approx(0.693, abs=1e-2)


# ---- job files ----


def make_job(title: str = "Book") -> tuple[dict, list[list[tuple[str, float]]]]:
    """Return a job record and plan with two chapters of two and one chunks."""
    plan = [[("Chapter one.", 0.8), ("More words here.", 0.0)], [("Chapter two.", 0.0)]]
    job = nar.new_job(
        title,
        [("One", plan[0]), ("Two", plan[1])],
        None,
        {"instruction": "", "cfg_scale": 1.0, "seed": 42},
        speak_titles=True,
    )
    return job, plan


def test_new_job_counts_chunks_and_characters_per_chapter():
    """The record's totals are sums of the plan, and nothing is done yet."""
    job, _ = make_job()

    assert job["state"] == "ready"
    assert (job["chunks_total"], job["chars_total"]) == (3, 12 + 16 + 12)
    assert [(c["chunks_total"], c["chars"]) for c in job["chapters"]] == [(2, 28), (1, 12)]
    assert job["chunks_done"] == 0
    assert nar.ID_PATTERN.fullmatch(job["id"])


def test_job_store_round_trips_a_job_and_its_plan(tmp_path):
    """A created job reads back with the same record and chunk plan."""
    store = nar.JobStore(tmp_path)
    job, plan = make_job()

    store.create(job, plan)

    assert store.job(job["id"]) == job
    assert store.plan(job["id"]) == plan


@pytest.mark.parametrize("job_id", ["", "../../etc", "z" * 32, "0" * 31])
def test_job_store_rejects_ids_that_are_not_32_hex_digits(tmp_path, job_id):
    """Only 32 hex digits name a job, so an id cannot point outside the jobs folder."""
    assert nar.JobStore(tmp_path).job(job_id) is None


def test_job_store_lists_newest_first(tmp_path):
    """Jobs are listed by creation time, newest first."""
    store = nar.JobStore(tmp_path)
    first, plan = make_job("First")
    second, _ = make_job("Second")
    second["created"] = first["created"] + 10
    store.create(first, plan)
    store.create(second, plan)

    assert [job["title"] for job in store.jobs()] == ["Second", "First"]


def test_job_store_chunk_files_are_written_whole_and_cleared_by_chapter(tmp_path):
    """A chunk is stored under its chapter, and clearing one chapter leaves the others."""
    store = nar.JobStore(tmp_path)
    job, plan = make_job()
    store.create(job, plan)

    store.write_chunk(job["id"], 0, 0, b"\x01\x00")
    store.write_chunk(job["id"], 1, 0, b"\x02\x00")
    store.clear_chunks(job["id"], 0)

    assert not store.chunk_path(job["id"], 0, 0).exists()
    assert store.chunk_path(job["id"], 1, 0).read_bytes() == b"\x02\x00"
    assert not list((tmp_path / "jobs" / job["id"] / "chunks").glob("*.part"))


def test_job_store_delete_removes_the_job_and_reports_unknown_ids(tmp_path):
    """Deleting removes the whole folder; a second delete finds nothing."""
    store = nar.JobStore(tmp_path)
    job, plan = make_job()
    store.create(job, plan)

    assert store.delete(job["id"]) is True
    assert not (tmp_path / "jobs" / job["id"]).exists()
    assert store.delete(job["id"]) is False


def test_interrupt_running_marks_only_running_jobs_and_resets_their_chapter(tmp_path):
    """A job left running by a dead supervisor becomes interrupted; others are unchanged."""
    store = nar.JobStore(tmp_path)
    running, plan = make_job("Running")
    running["state"] = "running"
    running["chapters"][0]["state"] = "running"
    done, _ = make_job("Done")
    done["state"] = "done"
    store.create(running, plan)
    store.create(done, plan)

    count = store.interrupt_running()

    assert count == 1
    after = store.job(running["id"])
    assert after is not None
    assert (after["state"], after["chapters"][0]["state"]) == ("interrupted", "pending")
    done_after = store.job(done["id"])
    assert done_after is not None
    assert done_after["state"] == "done"
