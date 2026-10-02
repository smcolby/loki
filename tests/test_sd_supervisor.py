"""Tests for image/sd_supervisor.py: running sd-server on demand behind one port."""

import http.client
import json
import socket
import sys
import threading
from urllib.parse import urlsplit

import pytest
import sd_supervisor as sup

FAKE_SD_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "sd-cpp-local"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"data": [{"b64_json": body["prompt"]}]})

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _free_port() -> int:
    """Return a loopback port nothing is listening on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def supervise(tmp_path):
    """Return a factory that serves a supervisor over a child command, stopping both after."""
    started = []

    def _start(command: list[str] | None = None) -> tuple[sup.Child, str]:
        port = _free_port()
        if command is None:
            script = tmp_path / "fake_sd_server.py"
            script.write_text(FAKE_SD_SERVER)
            command = [sys.executable, str(script), str(port)]
        child = sup.Child(command, port, start_timeout=10)
        server = sup.serve(child, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append((server, child))
        return child, f"http://127.0.0.1:{server.server_address[1]}"

    yield _start
    for server, child in started:
        server.shutdown()
        child.inflight = 0
        child.stop()


def _request(url: str, data: dict | None = None) -> tuple[int, dict]:
    """Send a GET, or a JSON POST when ``data`` is given, and decode the reply."""
    parts = urlsplit(url)
    connection = http.client.HTTPConnection(parts.netloc, timeout=10)
    if data is None:
        connection.request("GET", parts.path)
    else:
        headers = {"Content-Type": "application/json"}
        connection.request("POST", parts.path, body=json.dumps(data), headers=headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def test_child_command_listens_privately():
    """sd-server listens on loopback at the private port, followed by the engine args."""
    command = sup.child_command("/opt/sd/bin/sd-server", ["--steps", "20"], 8081)

    assert command == [
        "/opt/sd/bin/sd-server",
        "--listen-ip",
        "127.0.0.1",
        "--listen-port",
        "8081",
        "--steps",
        "20",
    ]


@pytest.mark.parametrize("arg", ["--listen-port", "--listen-ip=0.0.0.0"])
def test_child_command_rejects_listen_flags(arg):
    """Engine args may not move sd-server off the supervisor's private port."""
    with pytest.raises(ValueError, match="the image provides these"):
        sup.child_command("sd-server", [arg, "x"], 8081)


def test_idle_supervisor_answers_without_starting_sd_server(supervise):
    """Health and the model listing answer while sd-server is stopped."""
    child, url = supervise()

    assert _request(url + "/health") == (200, {"status": "ok", "loaded": False})
    assert _request(url + "/v1/models")[0] == 200
    assert not child.running()


def test_generation_starts_sd_server_and_unload_stops_it(supervise):
    """The first request starts sd-server and is proxied; unload stops it again."""
    child, url = supervise()

    status, body = _request(url + "/v1/images/generations", {"prompt": "fox"})
    assert (status, body) == (200, {"data": [{"b64_json": "fox"}]})
    assert _request(url + "/health")[1]["loaded"] is True

    assert _request(url + "/unload", {}) == (200, {"status": "unloaded"})
    assert _request(url + "/health")[1]["loaded"] is False
    assert not child.running()


def test_unload_while_busy_returns_409(supervise):
    """A request in flight keeps sd-server running and unload reports busy."""
    child, url = supervise()
    child.acquire()

    assert _request(url + "/unload", {}) == (409, {"status": "busy"})
    assert child.running()
    child.release()


def test_failed_start_returns_502(supervise):
    """sd-server exiting before it listens fails the request with its exit code."""
    _, url = supervise([sys.executable, "-c", "raise SystemExit(3)"])

    status, body = _request(url + "/v1/images/generations", {"prompt": "fox"})

    assert status == 502
    assert body["error"]["message"] == "sd-server exited with code 3 on startup"
