"""Run sd-server on demand so an idle image engine holds no GPU memory.

sd-server keeps compute buffers on the GPU after its first generation and has
no way to release them, so this supervisor owns the public port instead. It
starts sd-server on a private port for the first request, proxies requests to
it, and stops it on ``POST /unload``. ``GET /health`` reports whether sd-server
is running, matching Strata's API so the gateway unloads both the same way.
"""

import http.client
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SD_SERVER = "/opt/sd/bin/sd-server"
PUBLIC_PORT = 8080
CHILD_PORT = 8081
START_TIMEOUT = 300.0
OWNED_FLAGS = ("--listen-ip", "--listen-port")
MODELS = {"object": "list", "data": [{"id": "sd-cpp-local", "object": "model"}]}


def child_command(exe: str, args: list[str], port: int) -> list[str]:
    """Return the sd-server command line listening privately on ``port``.

    Parameters
    ----------
    exe : str
        Path to the sd-server binary.
    args : list[str]
        Engine arguments from the image engine config.
    port : int
        Loopback port sd-server listens on.

    Returns
    -------
    list[str]
        Command line for ``subprocess.Popen``.

    Raises
    ------
    ValueError
        If ``args`` sets a listen address, which the supervisor owns.

    Examples
    --------
    >>> child_command("sd-server", ["--steps", "20"], 8081)
    ['sd-server', '--listen-ip', '127.0.0.1', '--listen-port', '8081', '--steps', '20']
    """
    owned = sorted({arg.split("=")[0] for arg in args} & set(OWNED_FLAGS))
    if owned:
        raise ValueError(f"engine args set {', '.join(owned)}; the image provides these")
    return [exe, "--listen-ip", "127.0.0.1", "--listen-port", str(port), *args]


class Child:
    """One sd-server process, started for requests and stopped on unload.

    Parameters
    ----------
    command : list[str]
        sd-server command line.
    port : int
        Loopback port the command listens on.
    start_timeout : float
        Seconds to wait for a started process to answer its model listing.
    """

    def __init__(self, command: list[str], port: int, start_timeout: float = START_TIMEOUT):
        self.command = command
        self.port = port
        self.start_timeout = start_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.inflight = 0
        self._lock = threading.Lock()

    def running(self) -> bool:
        """Report whether the process is alive."""
        return self.process is not None and self.process.poll() is None

    def _wait_ready(self, process: subprocess.Popen[bytes]) -> None:
        """Block until ``process`` answers ``/v1/models``, raising RuntimeError if it cannot."""
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"sd-server exited with code {process.returncode} on startup")
            try:
                connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                connection.request("GET", "/v1/models")
                if connection.getresponse().status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        process.kill()
        process.wait()
        raise RuntimeError(f"sd-server did not start within {self.start_timeout:.0f} s")

    def acquire(self) -> None:
        """Count a request in flight, starting the process first if needed."""
        with self._lock:
            if not self.running():
                process = subprocess.Popen(self.command)  # noqa: S603 (fixed binary, config args)
                self._wait_ready(process)
                self.process = process
            self.inflight += 1

    def release(self) -> None:
        """Mark one request finished."""
        with self._lock:
            self.inflight -= 1

    def stop(self) -> bool:
        """Stop the process unless a request is in flight; return whether it is stopped."""
        with self._lock:
            if self.inflight:
                return False
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
            return True


class Handler(BaseHTTPRequestHandler):
    """Serve health and unload locally and proxy everything else to sd-server."""

    child: Child
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, body: Any) -> None:
        """Write a JSON response."""
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming)
        """Answer health and the model listing without starting sd-server."""
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "loaded": self.child.running()})
        elif self.path == "/v1/models":
            self._send_json(200, MODELS)
        else:
            self._proxy()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming)
        """Stop sd-server on unload and proxy any other request."""
        if self.path == "/unload":
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.child.stop():
                self._send_json(200, {"status": "unloaded"})
            else:
                self._send_json(409, {"status": "busy"})
        else:
            self._proxy()

    def _proxy(self) -> None:
        """Forward the request to sd-server, starting it first, and relay the response."""
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))

        # Start sd-server and report a failed start as a gateway error
        try:
            self.child.acquire()
        except (OSError, RuntimeError) as exc:
            self._send_json(502, {"error": {"message": str(exc), "type": "server_error"}})
            return

        # Relay one request and its whole response, holding the GPU until done
        try:
            connection = http.client.HTTPConnection("127.0.0.1", self.child.port)
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
        except OSError as exc:
            self._send_json(
                502, {"error": {"message": f"sd-server: {exc}", "type": "server_error"}}
            )
            return
        finally:
            self.child.release()
        self.send_response(response.status)
        self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(child: Child, port: int) -> ThreadingHTTPServer:
    """Return a server on all interfaces that supervises ``child``."""
    handler = type("BoundHandler", (Handler,), {"child": child})
    return ThreadingHTTPServer(("0.0.0.0", port), handler)  # noqa: S104 (compose network only)


def main() -> None:
    """Serve until SIGTERM, then stop sd-server."""
    try:
        command = child_command(SD_SERVER, sys.argv[1:], CHILD_PORT)
    except ValueError as exc:
        sys.exit(f"Cannot start sd-server: {exc}")
    child = Child(command, CHILD_PORT)
    server = serve(child, PUBLIC_PORT)

    # Shut down from a thread, since shutdown() blocks until serve_forever returns
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    server.serve_forever()
    child.inflight = 0
    child.stop()


if __name__ == "__main__":
    main()
