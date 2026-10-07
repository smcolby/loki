"""Route OpenAI and Anthropic API requests to whichever engine serves the model.

loki runs llama-server, Strata, and sd-server side by side, but one GPU holds
only one of them at a time. The gateway publishes a single API: it merges the engines'
model lists, sends each request to the engine that owns the requested model,
and unloads every other engine's models before the first request to a
different engine. Requests to the engine that already holds the GPU pass
straight through; a request for another engine waits for in-flight requests to
finish, then swaps. An optional preload loads one text model at startup, and
``/strata/`` serves the live Strata engine's web dashboard read-only.
"""

import asyncio
import contextlib
import email.policy
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from email.parser import BytesParser
from typing import Any, ClassVar

from aiohttp import ClientError, ClientSession, ClientTimeout, web

log = logging.getLogger("gateway")

HOP_BY_HOP = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
TEXT_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
)
IMAGE_PATHS = ("/v1/images/generations", "/v1/images/edits")
# Opens the dashboard on its Monitor tab (it reads the tab from the URL fragment at
# load) and hides the Chat tab, whose requests the read-only route refuses
DASHBOARD_HEAD = (
    b"<head><script>location.hash || history.replaceState(null, '', '#monitor')</script>"
    b'<style>.st-tab[data-tab="chat"] { display: none; }</style>'
)
MODEL_LIST_TTL = 30.0
UNLOAD_TIMEOUT = 120.0
POLL_INTERVAL = 0.5
PRELOAD_RETRY = 5.0
PRELOAD_TIMEOUT = 600.0


class EngineError(RuntimeError):
    """An engine could not list, unload, or report its models."""


def api_error(status: int, message: str, kind: str = "invalid_request_error") -> web.Response:
    """Build an OpenAI-style JSON error response."""
    return web.json_response({"error": {"message": message, "type": kind}}, status=status)


@dataclass
class Engine:
    """One inference server behind the gateway.

    Parameters
    ----------
    name : str
        Engine name shown in logs and the model list; its prefix before the first
        hyphen is the engine type (``llama``, ``strata``, or ``image``).
    url : str
        Base URL of the engine's HTTP API, without a trailing slash.
    session : ClientSession
        Shared client session for engine requests.
    """

    # Request paths the engine serves, whether /v1/models lists its models, and
    # whether it serves Strata's web dashboard
    paths: ClassVar[tuple[str, ...]] = TEXT_PATHS
    listed: ClassVar[bool] = True
    dashboard: ClassVar[bool] = False

    name: str
    url: str
    session: ClientSession

    async def _get_json(self, path: str) -> Any:
        """GET a path and decode its JSON body, raising EngineError on failure."""
        try:
            async with self.session.get(self.url + path) as response:
                if response.status != 200:
                    raise EngineError(f"{self.name} {path} returned HTTP {response.status}")
                return await response.json()
        except (ClientError, TimeoutError, ValueError) as exc:
            raise EngineError(f"{self.name} {path} failed: {exc}") from exc

    async def models(self) -> list[dict[str, Any]]:
        """Return the engine's OpenAI model objects."""
        body = await self._get_json("/v1/models")
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EngineError(f"{self.name} /v1/models has no data list")
        return [model for model in data if isinstance(model, dict) and "id" in model]

    async def loaded(self) -> bool:
        """Report whether any of the engine's models holds GPU memory."""
        raise NotImplementedError

    async def request_unload(self) -> None:
        """Ask the engine to release its models without waiting."""
        raise NotImplementedError

    async def unload(self, timeout: float | None = None) -> None:
        """Release the engine's models and wait until the GPU memory is free.

        An engine that cannot be reached holds no GPU memory (a stopped or
        crashed container frees it), so it counts as unloaded.

        Parameters
        ----------
        timeout : float or None
            Seconds to wait for the engine to unload; ``UNLOAD_TIMEOUT`` when None.

        Raises
        ------
        EngineError
            If the engine still reports a loaded model after ``timeout`` seconds.
        """
        timeout = UNLOAD_TIMEOUT if timeout is None else timeout
        try:
            holding = await self.loaded()
        except EngineError as exc:
            log.warning("treating %s as unloaded: %s", self.name, exc)
            return
        deadline = time.monotonic() + timeout
        while holding:
            if time.monotonic() > deadline:
                raise EngineError(f"{self.name} did not unload within {timeout:.0f} s")
            await self.request_unload()
            await asyncio.sleep(POLL_INTERVAL)
            holding = await self.loaded()


class LlamaEngine(Engine):
    """llama-server in router mode, which loads and unloads models by name."""

    async def _loaded_ids(self) -> list[str]:
        """Return ids of models the router is loading or has loaded."""
        loaded = []
        for model in await self.models():
            status = model.get("status")
            value = status.get("value") if isinstance(status, dict) else status
            if value not in (None, "unloaded"):
                loaded.append(model["id"])
        return loaded

    async def loaded(self) -> bool:
        """Report whether the router holds any model."""
        return bool(await self._loaded_ids())

    async def request_unload(self) -> None:
        """Ask the router to unload every model it holds."""
        for model_id in await self._loaded_ids():
            try:
                async with self.session.post(
                    self.url + "/models/unload", json={"model": model_id}
                ) as response:
                    await response.read()
            except (ClientError, TimeoutError) as exc:
                raise EngineError(f"llama unload of {model_id} failed: {exc}") from exc


class StrataEngine(Engine):
    """Strata's server, which serves one model and unloads it on request."""

    dashboard: ClassVar[bool] = True

    async def loaded(self) -> bool:
        """Report whether the engine process is running."""
        body = await self._get_json("/health")
        return bool(isinstance(body, dict) and body.get("loaded"))

    async def request_unload(self) -> None:
        """Ask Strata to unload; a busy engine answers 409 and is retried.

        Strata refuses control requests that are not JSON, so the request
        carries an empty JSON body.
        """
        try:
            async with self.session.post(self.url + "/unload", json={}) as response:
                await response.read()
        except (ClientError, TimeoutError) as exc:
            raise EngineError(f"{self.name} unload failed: {exc}") from exc


@dataclass
class ImageEngine(StrataEngine):
    """stable-diffusion.cpp's sd-server, which generates images from one model.

    The image's supervisor runs sd-server on demand and offers Strata's
    ``/health`` and ``/unload``, so the engine unloads like Strata. sd-server
    lists a fixed model id and ignores the requested one, so the gateway
    publishes ``model_id`` in its place. Its models stay out of ``/v1/models``,
    which chat clients read as a list of chat models.
    """

    paths: ClassVar[tuple[str, ...]] = IMAGE_PATHS
    listed: ClassVar[bool] = False
    dashboard: ClassVar[bool] = False

    model_id: str

    async def models(self) -> list[dict[str, Any]]:
        """Return the configured model once sd-server answers its model listing."""
        await self._get_json("/v1/models")
        return [{"id": self.model_id, "object": "model"}]


@dataclass
class Arbiter:
    """Grant one engine the GPU at a time.

    An engine keeps the GPU while it has requests in flight. A request for a
    different engine blocks new requests to the current one, waits for the
    in-flight count to reach zero, then unloads every other engine.

    Parameters
    ----------
    engines : list[Engine]
        Every engine that shares the GPU.
    """

    engines: list[Engine]
    active: str | None = None
    recent: list[str] = field(default_factory=list)
    inflight: int = 0
    waiting: dict[str, int] = field(default_factory=dict)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    def _may_enter(self, name: str) -> bool:
        """Decide whether a request for ``name`` may start now."""
        if self.inflight == 0:
            return True
        others_waiting = any(count for other, count in self.waiting.items() if other != name)
        return self.active == name and not others_waiting

    async def acquire(self, engine: Engine) -> None:
        """Wait for the GPU and make ``engine`` the only one holding models.

        Raises
        ------
        EngineError
            If another engine fails to unload; the GPU is left unassigned.
        """
        async with self._cond:
            self.waiting[engine.name] = self.waiting.get(engine.name, 0) + 1
            try:
                await self._cond.wait_for(lambda: self._may_enter(engine.name))
            finally:
                # Wake requests this one was holding back, including when it was cancelled
                self.waiting[engine.name] -= 1
                self._cond.notify_all()

            # Swap engines while holding the lock so no other request starts mid-swap
            if self.active != engine.name:
                self.active = None
                for other in self.engines:
                    if other is not engine:
                        log.info("unloading %s before %s", other.name, engine.name)
                        await other.unload()
                self.active = engine.name

                # Remember the order engines held the GPU, most recent last
                if engine.name in self.recent:
                    self.recent.remove(engine.name)
                self.recent.append(engine.name)
            self.inflight += 1

    async def release(self) -> None:
        """Finish one request and wake requests waiting for the GPU."""
        async with self._cond:
            self.inflight -= 1
            self._cond.notify_all()


@dataclass
class Gateway:
    """Hold engines, the model-to-engine map, and the GPU arbiter.

    Parameters
    ----------
    engines : list[Engine]
        Engines in priority order for listing.
    """

    engines: list[Engine]
    arbiter: Arbiter = field(init=False)
    _owners: dict[str, Engine] = field(default_factory=dict)
    _listed: list[dict[str, Any]] = field(default_factory=list)
    _known: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _refreshed: float = float("-inf")

    def __post_init__(self) -> None:
        """Create the arbiter over this gateway's engines."""
        self.arbiter = Arbiter(self.engines)

    async def refresh(self) -> list[dict[str, Any]]:
        """Rebuild the merged model list from each engine's latest listing.

        An engine that lists no models or cannot be reached keeps its last
        non-empty listing, since Strata lists nothing while it loads a model
        and an engine restart briefly refuses connections. An engine that has
        never listed a model is left out.

        Raises
        ------
        EngineError
            If two engines publish the same model id.
        """
        owners: dict[str, Engine] = {}
        listed: list[dict[str, Any]] = []
        for engine in self.engines:
            try:
                models = await engine.models()
            except EngineError as exc:
                log.warning("%s did not list its models: %s", engine.name, exc)
                models = []

            # Fall back to the engine's last non-empty listing
            if models:
                self._known[engine.name] = models
            models = self._known.get(engine.name, [])
            for model in models:
                if model["id"] in owners:
                    raise EngineError(
                        f"model id {model['id']!r} is served by both "
                        f"{owners[model['id']].name} and {engine.name}"
                    )
                owners[model["id"]] = engine
                if engine.listed:
                    listed.append({**model, "owned_by": engine.name})
        self._owners, self._listed, self._refreshed = owners, listed, time.monotonic()
        return listed

    async def owner(self, model_id: str) -> Engine | None:
        """Return the engine serving ``model_id``, refreshing a stale or missing entry."""
        stale = time.monotonic() - self._refreshed > MODEL_LIST_TTL
        if stale or model_id not in self._owners:
            await self.refresh()
        return self._owners.get(model_id)


async def handle_health(request: web.Request) -> web.Response:
    """Report gateway liveness and each engine's reachability and load state."""
    gateway: Gateway = request.app["gateway"]
    engines: dict[str, str] = {}
    for engine in gateway.engines:
        try:
            engines[engine.name] = "loaded" if await engine.loaded() else "idle"
        except EngineError:
            engines[engine.name] = "offline"
    return web.json_response({"status": "ok", "active": gateway.arbiter.active, "engines": engines})


async def handle_models(request: web.Request) -> web.Response:
    """Serve the merged list of text models."""
    gateway: Gateway = request.app["gateway"]
    try:
        listed = await gateway.refresh()
    except EngineError as exc:
        return api_error(500, str(exc), "server_error")
    return web.json_response({"object": "list", "data": listed})


async def _forward(
    request: web.Request, engine: Engine, body: bytes, path: str | None = None
) -> web.StreamResponse:
    """Stream one request to ``engine`` and its response back to the client.

    A failure before the engine answers becomes a 502; a failure mid-stream
    ends the response early, since its status line is already sent.

    Parameters
    ----------
    request : web.Request
        Client request whose method and headers are forwarded.
    engine : Engine
        Engine that receives the request.
    body : bytes
        Request body, forwarded unchanged.
    path : str or None, optional
        Engine path and query to request; the client's own path when None (the
        default).
    """
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    response: web.StreamResponse | None = None
    try:
        async with engine.session.request(
            request.method,
            engine.url + (path if path is not None else request.rel_url.path_qs),
            headers=headers,
            data=body,
            auto_decompress=False,
        ) as upstream:
            response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
            for key, value in upstream.headers.items():
                if key.lower() not in HOP_BY_HOP:
                    response.headers.add(key, value)
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
    except (ClientError, TimeoutError) as exc:
        if response is None:
            return api_error(502, f"{engine.name} request failed: {exc}", "server_error")
        log.warning("%s stream ended early: %s", engine.name, exc)
        return response
    await response.write_eof()
    return response


def _requested_model(body: bytes, content_type: str) -> object:
    """Return the ``model`` field of a JSON or multipart form request body.

    Image edits arrive as multipart form data with the reference images as file
    parts; every other request is JSON.

    Parameters
    ----------
    body : bytes
        Raw request body, forwarded to the engine unchanged.
    content_type : str
        The request's full Content-Type header, including any boundary.

    Returns
    -------
    object
        The model field's value, or None when the body has no model field.

    Raises
    ------
    ValueError
        If the body is neither JSON nor multipart form data.
    """
    if not content_type.lower().startswith("multipart/form-data"):
        payload = json.loads(body)
        return payload.get("model") if isinstance(payload, dict) else None

    # Parse the form as a MIME message whose header is the request's Content-Type
    message = BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    if not message.is_multipart():
        raise ValueError("multipart body has no parts")
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "model":
            payload = part.get_payload(decode=True)
            return payload.decode() if isinstance(payload, bytes) else None
    return None


async def handle_inference(request: web.Request) -> web.StreamResponse:
    """Route a model request to its engine after claiming the GPU for it."""
    gateway: Gateway = request.app["gateway"]
    body = await request.read()
    try:
        model_id = _requested_model(body, request.headers.get("Content-Type", ""))
    except ValueError:
        return api_error(400, "request body must be JSON or multipart form data")
    if not isinstance(model_id, str):
        return api_error(400, "request body has no model")

    try:
        engine = await gateway.owner(model_id)
    except EngineError as exc:
        return api_error(500, str(exc), "server_error")
    if engine is None:
        return api_error(404, f"model {model_id!r} is not served by any engine")
    if request.path not in engine.paths:
        return api_error(400, f"model {model_id!r} does not serve {request.path}")

    try:
        await gateway.arbiter.acquire(engine)
    except EngineError as exc:
        return api_error(503, f"cannot free the GPU for {model_id}: {exc}", "server_error")
    try:
        return await _forward(request, engine, body)
    finally:
        await gateway.arbiter.release()


async def preload(gateway: Gateway, model_id: str) -> None:
    """Load a text model onto the GPU so the first request skips the load.

    Waits up to ``PRELOAD_TIMEOUT`` seconds for an engine to list the model,
    then sends it a one-token chat request through the arbiter. A request that
    claims the GPU first wins, and the preload is skipped. Failures are logged,
    never raised, since the gateway serves requests either way.

    Parameters
    ----------
    gateway : Gateway
        Gateway whose engines and arbiter serve the model.
    model_id : str
        Id of the text model to load.
    """
    # Wait for the owning engine to start and list the model
    deadline = time.monotonic() + PRELOAD_TIMEOUT
    while True:
        try:
            engine = await gateway.owner(model_id)
        except EngineError as exc:
            log.warning("preload of %s cannot list models: %s", model_id, exc)
            engine = None
        if engine is not None:
            break
        if time.monotonic() > deadline:
            log.warning("preload skipped: no engine listed %s", model_id)
            return
        await asyncio.sleep(PRELOAD_RETRY)

    if engine.paths != TEXT_PATHS:
        log.warning("preload skipped: %s is not a text model", model_id)
        return

    # Leave the GPU to any request that arrived while engines were starting
    arbiter = gateway.arbiter
    if arbiter.active is not None or arbiter.inflight or any(arbiter.waiting.values()):
        log.info("preload of %s skipped: a request claimed the GPU first", model_id)
        return
    try:
        await arbiter.acquire(engine)
    except EngineError as exc:
        log.warning("preload of %s cannot free the GPU: %s", model_id, exc)
        return

    # Generate one token, which loads the model's weights
    body = {"model": model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    log.info("preloading %s on %s", model_id, engine.name)
    started = time.monotonic()
    try:
        async with engine.session.post(engine.url + "/v1/chat/completions", json=body) as response:
            await response.read()
        if response.status == 200:
            log.info("preloaded %s in %.0f s", model_id, time.monotonic() - started)
        else:
            log.warning("preload of %s returned HTTP %d", model_id, response.status)
    except (ClientError, TimeoutError) as exc:
        log.warning("preload of %s failed: %s", model_id, exc)
    finally:
        await arbiter.release()


async def handle_dashboard(request: web.Request) -> web.StreamResponse:
    """Serve Strata's web dashboard, read-only, from the Strata engine that last held the GPU.

    Each Strata engine runs its own dashboard for its own model, so ``/strata/``
    follows the live one, or the most recently used one while an image or llama
    model holds the GPU. Only reads pass: the dashboard's chat would bypass the
    GPU arbiter and its settings form would change the engine. The start page
    opens on the Monitor tab and hides Chat.
    """
    gateway: Gateway = request.app["gateway"]
    engines = {engine.name: engine for engine in gateway.engines if engine.dashboard}
    if not engines:
        return api_error(404, "no Strata engine is configured")

    # Redirect to the trailing slash, which the dashboard's relative paths need
    if request.path == "/strata":
        raise web.HTTPPermanentRedirect("/strata/")
    if request.method not in ("GET", "HEAD"):
        return api_error(405, "the Strata dashboard is read-only through the gateway")

    recent = [name for name in reversed(gateway.arbiter.recent) if name in engines]
    engine = engines[recent[0] if recent else next(iter(engines))]
    path = "/" + request.match_info["tail"]
    if path == "/" and request.method == "GET":
        return await _dashboard_page(engine)
    if request.query_string:
        path += "?" + request.query_string
    return await _forward(request, engine, b"", path)


async def _dashboard_page(engine: Engine) -> web.Response:
    """Return the dashboard's start page, opening on Monitor with the blocked Chat tab hidden."""
    try:
        async with engine.session.get(engine.url + "/") as upstream:
            page = await upstream.read()
            status, content_type = upstream.status, upstream.content_type
    except (ClientError, TimeoutError) as exc:
        return api_error(502, f"{engine.name} dashboard request failed: {exc}", "server_error")
    if status == 200 and content_type == "text/html":
        page = page.replace(b"<head>", DASHBOARD_HEAD, 1)
    return web.Response(body=page, status=status, content_type=content_type, charset="utf-8")


def build_app(engines: list[Engine], preload_model: str = "") -> web.Application:
    """Create the gateway application around ``engines``.

    Parameters
    ----------
    engines : list[Engine]
        Engines in priority order for listing.
    preload_model : str, optional
        Text model to load in the background at startup; empty (the default)
        loads nothing until the first request.
    """
    app = web.Application(client_max_size=64 * 1024**2)
    app["gateway"] = Gateway(engines)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    for path in (*TEXT_PATHS, *IMAGE_PATHS):
        app.router.add_post(path, handle_inference)
    app.router.add_route("*", "/strata", handle_dashboard)
    app.router.add_route("*", "/strata/{tail:.*}", handle_dashboard)

    # Run the preload beside request handling and stop it on shutdown
    if preload_model:

        async def preload_context(app: web.Application) -> AsyncIterator[None]:
            task = asyncio.create_task(preload(app["gateway"], preload_model))
            yield
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        app.cleanup_ctx.append(preload_context)
    return app


ENGINE_TYPES: dict[str, type[Engine]] = {
    "llama": LlamaEngine,
    "strata": StrataEngine,
    "image": ImageEngine,
}


def _pairs(spec: str) -> list[tuple[str, str]]:
    """Split comma-separated ``name=value`` entries, raising ValueError on a malformed one."""
    pairs = []
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        name, sep, value = entry.partition("=")
        if not sep or not value:
            raise ValueError(f"engine entry {entry!r} is not name=value")
        pairs.append((name, value))
    return pairs


def parse_engines(spec: str, session: ClientSession, image_models: str = "") -> list[Engine]:
    """Build engines from ``name=url`` pairs separated by commas.

    The engine type is the name up to its first hyphen, so ``strata`` and
    ``strata-swift`` are both Strata engines with their own containers.

    Parameters
    ----------
    spec : str
        ``LOKI_ENGINES``: each engine's name and base URL.
    session : ClientSession
        Shared client session for engine requests.
    image_models : str, optional
        ``LOKI_IMAGE_MODELS``: each image engine's name and published model id.
        Empty by default.

    Raises
    ------
    ValueError
        If an entry is malformed, repeats a name, names an unknown engine type,
        or an image engine and its model id do not pair up.

    Examples
    --------
    >>> async def demo():
    ...     async with ClientSession() as session:
    ...         return [e.name for e in parse_engines("llama=http://llama:8080", session)]
    >>> asyncio.run(demo())
    ['llama']
    """
    models = dict(_pairs(image_models))
    engines: list[Engine] = []
    for name, url in _pairs(spec):
        kind = name.partition("-")[0]
        if kind not in ENGINE_TYPES:
            expected = ", ".join(ENGINE_TYPES)
            raise ValueError(
                f"unknown engine type {kind!r} in {name!r}; expected one of {expected}"
            )
        if any(engine.name == name for engine in engines):
            raise ValueError(f"engine {name!r} is listed twice")

        # Give an image engine the model id it publishes
        if kind == "image":
            if name not in models:
                raise ValueError(f"image engine {name!r} has no model id in LOKI_IMAGE_MODELS")
            engines.append(ImageEngine(name, url.rstrip("/"), session, models.pop(name)))
        else:
            engines.append(ENGINE_TYPES[kind](name, url.rstrip("/"), session))
    if not engines:
        raise ValueError("no engines configured")
    if models:
        raise ValueError(f"LOKI_IMAGE_MODELS names unknown engines: {', '.join(models)}")
    return engines


def main() -> None:
    """Serve the gateway on port 8080 with engines from ``LOKI_ENGINES``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    spec = os.environ.get("LOKI_ENGINES", "")
    image_models = os.environ.get("LOKI_IMAGE_MODELS", "")
    preload_model = os.environ.get("LOKI_PRELOAD", "")

    async def factory() -> web.Application:
        session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
        app = build_app(parse_engines(spec, session, image_models), preload_model)

        async def close_session(_: web.Application) -> None:
            await session.close()

        app.on_cleanup.append(close_session)
        return app

    web.run_app(factory(), host="0.0.0.0", port=8080)  # noqa: S104 (compose network only)


__all__ = ["Arbiter", "Engine", "Gateway", "build_app", "main", "parse_engines", "preload"]

if __name__ == "__main__":
    main()
