"""Route OpenAI and Anthropic API requests to whichever engine serves the model.

loki runs llama-server and Strata side by side, but one GPU holds only one of
them at a time. The gateway publishes a single API: it merges the engines'
model lists, sends each request to the engine that owns the requested model,
and unloads every other engine's models before the first request to a
different engine. Requests to the engine that already holds the GPU pass
straight through; a request for another engine waits for in-flight requests to
finish, then swaps.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
INFERENCE_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
)
MODEL_LIST_TTL = 30.0
UNLOAD_TIMEOUT = 120.0
POLL_INTERVAL = 0.5


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
        Short engine name shown in logs and the model list (``llama`` or ``strata``).
    url : str
        Base URL of the engine's HTTP API, without a trailing slash.
    session : ClientSession
        Shared client session for engine requests.
    """

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

    async def loaded(self) -> bool:
        """Report whether Strata's engine process is running."""
        body = await self._get_json("/health")
        return bool(isinstance(body, dict) and body.get("loaded"))

    async def request_unload(self) -> None:
        """Ask Strata to unload; a busy engine answers 409 and is retried."""
        try:
            async with self.session.post(self.url + "/unload") as response:
                await response.read()
        except (ClientError, TimeoutError) as exc:
            raise EngineError(f"strata unload failed: {exc}") from exc


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
    _refreshed: float = float("-inf")

    def __post_init__(self) -> None:
        """Create the arbiter over this gateway's engines."""
        self.arbiter = Arbiter(self.engines)

    async def refresh(self) -> list[dict[str, Any]]:
        """Rebuild the merged model list, skipping unreachable engines.

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
                log.warning("skipping %s in the model list: %s", engine.name, exc)
                continue
            for model in models:
                if model["id"] in owners:
                    raise EngineError(
                        f"model id {model['id']!r} is served by both "
                        f"{owners[model['id']].name} and {engine.name}"
                    )
                owners[model["id"]] = engine
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
    """Serve the merged model list."""
    gateway: Gateway = request.app["gateway"]
    try:
        listed = await gateway.refresh()
    except EngineError as exc:
        return api_error(500, str(exc), "server_error")
    return web.json_response({"object": "list", "data": listed})


async def _forward(request: web.Request, engine: Engine, body: bytes) -> web.StreamResponse:
    """Stream one request to ``engine`` and its response back to the client.

    A failure before the engine answers becomes a 502; a failure mid-stream
    ends the response early, since its status line is already sent.
    """
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    response: web.StreamResponse | None = None
    try:
        async with engine.session.request(
            request.method,
            engine.url + request.rel_url.path_qs,
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


async def handle_inference(request: web.Request) -> web.StreamResponse:
    """Route a model request to its engine after claiming the GPU for it."""
    gateway: Gateway = request.app["gateway"]
    body = await request.read()
    try:
        payload = json.loads(body)
    except ValueError:
        return api_error(400, "request body must be JSON")
    model_id = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model_id, str):
        return api_error(400, "request body has no model")

    try:
        engine = await gateway.owner(model_id)
    except EngineError as exc:
        return api_error(500, str(exc), "server_error")
    if engine is None:
        return api_error(404, f"model {model_id!r} is not served by any engine")

    try:
        await gateway.arbiter.acquire(engine)
    except EngineError as exc:
        return api_error(503, f"cannot free the GPU for {model_id}: {exc}", "server_error")
    try:
        return await _forward(request, engine, body)
    finally:
        await gateway.arbiter.release()


def build_app(engines: list[Engine]) -> web.Application:
    """Create the gateway application around ``engines``."""
    app = web.Application(client_max_size=64 * 1024**2)
    app["gateway"] = Gateway(engines)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    for path in INFERENCE_PATHS:
        app.router.add_post(path, handle_inference)
    return app


ENGINE_TYPES: dict[str, Callable[[str, str, ClientSession], Engine]] = {
    "llama": LlamaEngine,
    "strata": StrataEngine,
}


def parse_engines(spec: str, session: ClientSession) -> list[Engine]:
    """Build engines from ``name=url`` pairs separated by commas.

    The engine type is the name up to its first hyphen, so ``strata`` and
    ``strata-swift`` are both Strata engines with their own containers.

    Raises
    ------
    ValueError
        If an entry is malformed, repeats a name, or names an unknown engine type.

    Examples
    --------
    >>> async def demo():
    ...     async with ClientSession() as session:
    ...         return [e.name for e in parse_engines("llama=http://llama:8080", session)]
    >>> asyncio.run(demo())
    ['llama']
    """
    engines = []
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        name, sep, url = entry.partition("=")
        if not sep or not url:
            raise ValueError(f"engine entry {entry!r} is not name=url")
        kind = name.partition("-")[0]
        if kind not in ENGINE_TYPES:
            expected = ", ".join(ENGINE_TYPES)
            raise ValueError(
                f"unknown engine type {kind!r} in {name!r}; expected one of {expected}"
            )
        if any(engine.name == name for engine in engines):
            raise ValueError(f"engine {name!r} is listed twice")
        engines.append(ENGINE_TYPES[kind](name, url.rstrip("/"), session))
    if not engines:
        raise ValueError("no engines configured")
    return engines


def main() -> None:
    """Serve the gateway on port 8080 with engines from ``LOKI_ENGINES``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    spec = os.environ.get("LOKI_ENGINES", "")

    async def factory() -> web.Application:
        session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
        app = build_app(parse_engines(spec, session))

        async def close_session(_: web.Application) -> None:
            await session.close()

        app.on_cleanup.append(close_session)
        return app

    web.run_app(factory(), host="0.0.0.0", port=8080)  # noqa: S104 (compose network only)


__all__ = ["Arbiter", "Engine", "Gateway", "build_app", "main", "parse_engines"]

if __name__ == "__main__":
    main()
