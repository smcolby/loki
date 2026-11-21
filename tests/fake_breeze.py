"""A stand-in for Breeze TTS 2's API, run as a subprocess by the speech supervisor tests.

Serves ``/health`` and ``/v1/audio/speech`` on the port given as the first argument.
Every request is appended to the JSON-lines file in ``FAKE_LOG``. Behavior comes from
the environment: ``FAKE_SECONDS`` is the length of audio per request, ``FAKE_PER_CHAR``
sizes the audio by the text instead, ``FAKE_SHORT_FIRST`` the number of requests answered
with a 0.1 s clip before one is normal, ``FAKE_BUSY_FIRST`` the number answered 409
before one succeeds, ``FAKE_FAIL`` makes every request fail with 500, and ``FAKE_GATE``
is a file the server waits to see before it streams the last block, which lets a test
hold a request in flight.
"""

import asyncio
import json
import math
import os
import struct
import sys
from pathlib import Path

from aiohttp import web

RATE = 24000
BUSY_LEFT = int(os.environ.get("FAKE_BUSY_FIRST", "0"))
SHORT_LEFT = int(os.environ.get("FAKE_SHORT_FIRST", "0"))


def tone(seconds: float) -> bytes:
    """Return a quiet 440 Hz tone as signed 16-bit little-endian PCM."""
    count = int(seconds * RATE)
    return struct.pack(f"<{count}h", *(int(8000 * math.sin(i * 0.115)) for i in range(count)))


async def health(_: web.Request) -> web.Response:
    """Report ready."""
    return web.json_response({"status": "ok", "sample_rate": RATE})


async def speech(request: web.Request) -> web.StreamResponse:
    """Record the form fields, then stream PCM in blocks with an odd byte count."""
    global BUSY_LEFT, SHORT_LEFT
    form = await request.post()
    record = {key: str(value) for key, value in form.items() if key != "ref_audio"}
    upload = form.get("ref_audio")
    record["ref_audio_bytes"] = len(upload.file.read()) if hasattr(upload, "file") else 0
    with open(os.environ["FAKE_LOG"], "a") as log:
        log.write(json.dumps(record) + "\n")
    if BUSY_LEFT:
        BUSY_LEFT -= 1
        return web.json_response({"detail": "busy"}, status=409)
    if os.environ.get("FAKE_FAIL"):
        return web.json_response({"detail": "boom"}, status=500)

    seconds = float(os.environ.get("FAKE_SECONDS", "4"))
    if os.environ.get("FAKE_PER_CHAR"):
        seconds = len(str(form.get("text", ""))) * float(os.environ["FAKE_PER_CHAR"])
    if SHORT_LEFT:
        SHORT_LEFT -= 1
        seconds = 0.1
    pcm = tone(seconds)
    response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
    await response.prepare(request)
    cut = len(pcm) // 2 + 1
    await response.write(pcm[:cut])
    gate = os.environ.get("FAKE_GATE")
    while gate and not Path(gate).exists():
        await asyncio.sleep(0.02)
    await response.write(pcm[cut:])
    await response.write_eof()
    return response


app = web.Application()
app.router.add_get("/health", health)
app.router.add_post("/v1/audio/speech", speech)

if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]), print=None)
