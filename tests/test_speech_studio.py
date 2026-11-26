"""Tests for the Studio page's playback timing, run in Node when it is installed."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parent.parent / "speech" / "studio" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def node(*args: str, source: str | None = None) -> str:
    """Run Node and return its output.

    The suite stubs ``subprocess.run`` for every test, so this uses ``Popen``.
    """
    process = subprocess.Popen(
        ["node", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = process.communicate(source, timeout=30)
    assert process.returncode == 0, err
    return out


def run_logic(expression: str) -> object:
    """Evaluate a JavaScript expression against the page's timing functions."""
    logic = re.search(r"// <logic>(.*?)// </logic>", PAGE.read_text(), re.S)
    assert logic is not None
    script = f"{logic.group(1)}\nconsole.log(JSON.stringify({expression}));"
    return json.loads(node("-e", script))


def test_page_script_has_valid_syntax():
    """The whole page script parses, so a typo cannot ship a dead Studio."""
    script = re.search(r"<script>(.*)</script>", PAGE.read_text(), re.S)
    assert script is not None

    node("--check", "--input-type=commonjs", "-", source=script.group(1))


def test_playback_waits_for_the_share_generation_cannot_finish_in_time():
    """At 0.77x real time and 2,000 characters (195 s planned), 46.5 s must exist before playing."""
    assert run_logic("audioNeeded(2000, 10, 13)") == pytest.approx(46.5, abs=0.1)


def test_playback_starts_after_a_short_cushion_when_generation_outpaces_it():
    """At or above real time only the margin is needed."""
    assert run_logic("audioNeeded(2000, 20, 10)") == 1.5


@pytest.mark.parametrize(
    ("received", "elapsed", "ended", "expected"),
    [
        (0.5, 1, False, False),
        (20, 26, False, False),
        (50, 65, False, True),
        (5, 6, True, True),
        (0, 0, True, False),
    ],
    ids=[
        "under one second",
        "too little made",
        "enough made",
        "short text finished",
        "nothing made",
    ],
)
def test_should_start_decides_from_the_audio_made_so_far(received, elapsed, ended, expected):
    """Playback starts once enough audio exists, or when generation has finished."""
    chars = 2000
    assert (
        run_logic(f"shouldStart({chars}, {received}, {elapsed}, {str(ended).lower()})") is expected
    )


def test_wait_estimate_is_the_missing_audio_over_the_generation_rate():
    """The countdown is (46.5 - 10 s made) at 10/13 of real time, about 47 s."""
    assert run_logic("secondsUntilStart(2000, 10, 13)") == pytest.approx(47.4, abs=0.2)


def test_wait_estimate_is_unknown_before_any_audio_arrives():
    """No rate exists before the first audio, so no countdown is shown."""
    assert run_logic("secondsUntilStart(2000, 0, 5)") is None


@pytest.mark.parametrize("rate", [0.5, 0.77, 1.0, 1.4])
def test_playback_started_by_the_rule_never_runs_dry(rate):
    """Simulating a 150 s narration at several speeds, playback ends no earlier than generation."""
    outcome = run_logic(
        f"""(() => {{
          const chars = 2000, truth = 150, rate = {rate};
          let t = 0, started = null;
          while (started === null && t < 600) {{
            t += 0.25;
            const made = Math.min(truth, rate * t);
            if (shouldStart(chars, made, t, made >= truth)) started = t;
          }}
          return {{ playbackEnds: started + truth, generationEnds: truth / rate }};
        }})()"""
    )

    assert outcome["playbackEnds"] >= outcome["generationEnds"] - 0.01


@pytest.mark.parametrize(
    ("text", "chapters"),
    [
        ("Just words.", 1),
        ("# One\nFirst.\n## Two\nSecond.", 2),
        ("Preface.\n# One\nFirst.", 2),
        ("# Empty\n# Real\nBody.", 1),
        ("#hashtag\n### Deeper\nBody.", 1),
        ("# Only a heading", 0),
        ("", 0),
    ],
    ids=["none", "two", "preface", "empty heading", "not headings", "no body", "blank"],
)
def test_chapter_count_follows_the_servers_heading_rule(text, chapters):
    """The preview counts a chapter at each heading that has text, plus text before the first."""
    assert run_logic(f"countChapters({json.dumps(text)})") == chapters


def test_job_time_left_uses_the_pace_measured_so_far():
    """A quarter of the characters took 50 s, so the other 750 take 150 s."""
    job = {"chars_total": 1000, "chars_done": 250, "generation_s": 50}

    assert run_logic(f"jobSecondsLeft({json.dumps(job)})") == pytest.approx(150.0)


def test_job_time_left_before_any_part_is_done_uses_the_measured_rate():
    """With nothing made yet, 1,000 characters at 0.075 s each and 0.77x real time is 97.4 s."""
    job = {"chars_total": 1000, "chars_done": 0, "generation_s": 0}

    assert run_logic(f"jobSecondsLeft({json.dumps(job)})") == pytest.approx(97.4, abs=0.1)


def test_a_finished_job_has_no_time_left():
    """No characters remain, so no time does."""
    job = {"chars_total": 500, "chars_done": 500, "generation_s": 400}

    assert run_logic(f"jobSecondsLeft({json.dumps(job)})") == 0


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(59, "59 s"), (61, "1 min 1 s"), (3725, "1 h 2 min"), (36000, "10 h 0 min")],
)
def test_durations_read_in_hours_for_book_length_jobs(seconds, text):
    """Narration of a book runs for hours, so durations over an hour show hours and minutes."""
    assert run_logic(f"formatSeconds({seconds})") == text


def test_retry_note_lists_each_rejected_attempt_and_is_empty_without_any():
    """The job card names the chapter and the reason for every attempt that was spoken again."""
    quiet = {"chapters": [{"title": "One", "retries": []}, {"title": "Two"}]}
    retried = {
        "chapters": [
            {"title": "One", "retries": ["part 3, attempt 1: silent"]},
            {"title": "Two", "retries": ["part 1, attempt 1: too short (0.1 s for 40 characters)"]},
        ]
    }

    assert run_logic(f"retryNote({json.dumps(quiet)})") == ""
    note = run_logic(f"retryNote({json.dumps(retried)})")
    assert isinstance(note, str)
    assert note.startswith("2 attempt(s)")
    assert "One, part 3, attempt 1: silent" in note
    assert "Two, part 1, attempt 1: too short" in note


def test_import_summary_reports_the_chapters_and_each_kind_of_change():
    """The first line counts chapters and characters; each warning adds a counted line."""
    result = {
        "chapters": [{"title": "One", "chars": 1200}, {"title": "Two", "chars": 800}],
        "warnings": [
            {"kind": "numeric citation", "count": 12, "sample": "[3]"},
            {"kind": "table skipped", "count": 2, "sample": ""},
        ],
    }

    lines = run_logic(f"importSummary({json.dumps(result)})")

    assert lines == [
        "Read 2 chapter(s), 2,000 characters. Review the text below before creating the job.",
        'numeric citation: 12 (for example "[3]")',
        "table skipped: 2",
    ]


@pytest.mark.parametrize(("seconds", "expected"), [(47.4, 55), (10.0, 12), (0.0, 0), (0.01, 1)])
def test_shown_wait_is_padded_and_rounded_up_to_a_whole_second(seconds, expected):
    """The displayed wait is 15% over the measured one, then rounded up, so it errs long."""
    assert run_logic(f"paddedWait({seconds})") == expected


@pytest.mark.parametrize(
    ("remaining_ms", "expected"), [(12000, 12), (11001, 12), (10999, 11), (400, 1), (-5000, 1)]
)
def test_countdown_falls_one_whole_second_at_a_time_and_stops_at_one(remaining_ms, expected):
    """The shown wait rounds up, so it steps down once per second and never reads zero or less."""
    assert run_logic(f"countdown({remaining_ms}, 0)") == expected


@pytest.mark.parametrize(
    ("ends_at", "candidate", "expected"),
    [
        (0, 50000, 50000),
        (40000, 42500, 40000),
        (40000, 37500, 40000),
        (40000, 29000, 29000),
        (40000, 60000, 60000),
        (-5, 8000, 8000),
    ],
)
def test_deadline_moves_only_for_a_change_beyond_the_tolerance(ends_at, candidate, expected):
    """A candidate within 3 s or 25% of the time left is ignored; a missed deadline is replaced."""
    assert run_logic(f"nextDeadline({ends_at}, {candidate}, 0)") == expected
