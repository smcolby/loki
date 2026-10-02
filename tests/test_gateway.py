"""Tests for gateway/gateway.py: routing requests and swapping engines on one GPU."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import ClientSession, web

import gateway as gw


@dataclass
class FakeEngine:
    """In-memory stand-in for an engine's HTTP API, recording every call."""

    models: list[str]
    loaded: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    bodies: list[Any] = field(default_factory=list)
    stick: bool = False
    gate: asyncio.Event | None = None

    def app(self) -> web.Application:
        """Build the aiohttp app serving both engines' endpoints."""
        app = web.Application()
        app.router.add_get("/v1/models", self.list_models)
        app.router.add_get("/health", self.health)
        app.router.add_post("/models/unload", self.unload_llama)
        app.router.add_post("/unload", self.unload_strata)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    async def list_models(self, _: web.Request) -> web.Response:
        """List models with llama-server style status objects."""
        data = [
            {"id": m, "status": {"value": "loaded" if m in self.loaded else "unloaded"}}
            for m in self.models
        ]
        return web.json_response({"object": "list", "data": data})

    async def health(self, _: web.Request) -> web.Response:
        """Report Strata-style load state."""
        return web.json_response({"status": "ok", "loaded": bool(self.loaded)})

    async def unload_llama(self, request: web.Request) -> web.Response:
        """Unload one model by name."""
        model = (await request.json())["model"]
        self.calls.append(f"unload:{model}")
        if not self.stick:
            self.loaded.discard(model)
        return web.json_response({"success": True})

    async def unload_strata(self, _: web.Request) -> web.Response:
        """Unload the single model."""
        self.calls.append("unload")
        if not self.stick:
            self.loaded.clear()
        return web.json_response({"status": "unloaded"})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        """Load the requested model and stream two chunks, optionally pausing on a gate."""
        body = await request.json()
        self.bodies.append(body)
        self.calls.append(f"chat:{body['model']}")
        self.loaded.add(body["model"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: one\n\n")
        if self.gate is not None:
            await self.gate.wait()
        await response.write(b"data: two\n\n")
        await response.write_eof()
        return response


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    """Poll unload progress quickly so swaps finish in milliseconds."""
    monkeypatch.setattr(gw, "POLL_INTERVAL", 0.01)


@pytest.fixture
def llama():
    """Fake llama-server router with two models."""
    return FakeEngine(["qwen", "gemma"])


@pytest.fixture
def strata():
    """Fake Strata server with one model."""
    return FakeEngine(["flash"])


@pytest.fixture
async def make_client(aiohttp_client, aiohttp_server):
    """Return a factory that starts a gateway over ``name=FakeEngine-or-url`` pairs."""
    session = ClientSession()

    async def make(**engines: FakeEngine | str):
        parts = []
        for name, engine in engines.items():
            if isinstance(engine, FakeEngine):
                engine = (await aiohttp_server(engine.app())).make_url("")
            parts.append(f"{name}={engine}")
        return await aiohttp_client(gw.build_app(gw.parse_engines(",".join(parts), session)))

    yield make
    await session.close()


@pytest.fixture
async def client(make_client, llama, strata):
    """Gateway client routed to the fake llama and Strata engines."""
    return await make_client(llama=llama, strata=strata)


def _chat(model: str) -> dict:
    """Build a minimal streaming chat request body."""
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True}


async def _complete(client, model: str) -> bytes:
    """Send one chat request through the gateway and read the whole stream."""
    response = await client.post("/v1/chat/completions", json=_chat(model))
    return await response.read()


async def test_models_merges_engines_with_owner(client):
    """/v1/models lists every engine's models tagged with the engine name."""
    body = await (await client.get("/v1/models")).json()

    assert [(m["id"], m["owned_by"]) for m in body["data"]] == [
        ("qwen", "llama"),
        ("gemma", "llama"),
        ("flash", "strata"),
    ]


async def test_chat_streams_from_owning_engine(client, llama, strata):
    """A request reaches only its model's engine with an unchanged body, and streams back."""
    response = await client.post("/v1/chat/completions", json=_chat("qwen"))

    assert response.status == 200
    assert response.headers["Content-Type"] == "text/event-stream"
    assert await response.read() == b"data: one\n\ndata: two\n\n"
    assert llama.bodies == [_chat("qwen")]
    assert strata.bodies == []


async def test_switching_engines_unloads_the_other_first(client, llama, strata):
    """Moving between engines unloads the previous engine before the next one sees the request."""
    for model in ("qwen", "flash", "gemma"):
        await _complete(client, model)

    assert llama.calls == ["chat:qwen", "unload:qwen", "chat:gemma"]
    assert strata.calls == ["chat:flash", "unload"]


async def test_same_engine_requests_do_not_unload(client, llama):
    """Consecutive requests to one engine leave model swaps to that engine's router."""
    for model in ("qwen", "gemma", "qwen"):
        await _complete(client, model)

    assert llama.calls == ["chat:qwen", "chat:gemma", "chat:qwen"]


async def test_other_engine_waits_for_in_flight_stream(client, llama, strata):
    """A request for another engine waits until the current stream ends, then swaps."""
    llama.gate = asyncio.Event()
    first = await client.post("/v1/chat/completions", json=_chat("qwen"))
    assert await first.content.readuntil(b"\n\n") == b"data: one\n\n"

    second = asyncio.create_task(_complete(client, "flash"))
    await asyncio.sleep(0.1)
    assert strata.calls == []
    assert llama.calls == ["chat:qwen"]

    llama.gate.set()
    await first.read()
    await second
    assert llama.calls == ["chat:qwen", "unload:qwen"]
    assert strata.calls == ["chat:flash"]


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        (b"not json", 400, "must be JSON"),
        (b'{"messages": []}', 400, "has no model"),
        (b'{"model": "missing"}', 404, "not served by any engine"),
    ],
    ids=["invalid-json", "no-model", "unknown-model"],
)
async def test_rejects_unroutable_requests(client, body, status, message):
    """Requests without a known model fail with an OpenAI-style error."""
    response = await client.post(
        "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )

    assert response.status == status
    assert message in (await response.json())["error"]["message"]


async def test_unload_timeout_returns_503(client, llama, strata, monkeypatch):
    """An engine that never releases the GPU fails the swap instead of overcommitting VRAM."""
    monkeypatch.setattr(gw, "UNLOAD_TIMEOUT", 0.1)
    llama.loaded.add("qwen")
    llama.stick = True

    response = await client.post("/v1/chat/completions", json=_chat("flash"))

    assert response.status == 503
    assert "did not unload" in (await response.json())["error"]["message"]
    assert strata.calls == []


async def test_offline_engine_is_skipped_and_counts_as_unloaded(make_client, llama):
    """With Strata unreachable, llama models still list and serve."""
    client = await make_client(llama=llama, strata="http://127.0.0.1:9")

    models = await (await client.get("/v1/models")).json()
    chat = await client.post("/v1/chat/completions", json=_chat("qwen"))
    health = await (await client.get("/health")).json()

    assert [m["id"] for m in models["data"]] == ["qwen", "gemma"]
    assert chat.status == 200
    assert health["engines"] == {"llama": "loaded", "strata": "offline"}


async def test_duplicate_model_id_is_an_error(make_client):
    """Two engines publishing one id fail loudly for listing and routing."""
    client = await make_client(llama=FakeEngine(["same"]), strata=FakeEngine(["same"]))

    listing = await client.get("/v1/models")
    chat = await client.post("/v1/chat/completions", json=_chat("same"))

    for response in (listing, chat):
        assert response.status == 500
        assert "served by both" in (await response.json())["error"]["message"]


async def test_engine_dying_before_response_returns_502(make_client, llama):
    """A model listed by an engine that then goes away yields a 502 for the request."""
    client = await make_client(llama=llama, strata=FakeEngine(["flash"]))
    await client.get("/v1/models")
    strata_engine = next(e for e in client.app["gateway"].engines if e.name == "strata")
    strata_engine.url = "http://127.0.0.1:9"

    response = await client.post("/v1/chat/completions", json=_chat("flash"))

    assert response.status == 502


async def test_health_reports_engine_states(client):
    """/health shows which engines hold models and which engine is active."""
    await _complete(client, "qwen")

    body = await (await client.get("/health")).json()

    assert body == {
        "status": "ok",
        "active": "llama",
        "engines": {"llama": "loaded", "strata": "idle"},
    }


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "no engines configured"),
        ("llama", "is not name=url"),
        ("vllm=http://x", "unknown engine"),
    ],
    ids=["empty", "no-url", "unknown-kind"],
)
async def test_parse_engines_rejects_bad_spec(spec, message):
    """Malformed LOKI_ENGINES values raise ValueError naming the problem."""
    async with ClientSession() as session:
        with pytest.raises(ValueError, match=message):
            gw.parse_engines(spec, session)
