"""Tests for gateway/gateway.py: routing requests and swapping engines on one GPU."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import ClientSession, FormData, web

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
        app.router.add_post("/v1/images/generations", self.generate)
        app.router.add_post("/v1/images/edits", self.edit)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/", self.page)
        return app

    async def page(self, _: web.Request) -> web.Response:
        """Serve a dashboard start page naming this engine's first model."""
        body = f"<!doctype html><html><head><title>{self.models[0]}</title></head></html>"
        return web.Response(text=body, content_type="text/html")

    async def metrics(self, request: web.Request) -> web.Response:
        """Answer a dashboard read with this engine's models, recording the query."""
        self.calls.append(f"metrics:{request.query_string}")
        return web.json_response({"models": self.models})

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

    async def unload_strata(self, request: web.Request) -> web.Response:
        """Unload the single model, refusing a request that is not JSON as Strata does."""
        if request.content_type != "application/json":
            return web.json_response({"error": {"message": "send application/json"}}, status=415)
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

    async def generate(self, request: web.Request) -> web.Response:
        """Start the model like the image supervisor and return one sd-server style image."""
        body = await request.json()
        self.bodies.append(body)
        self.calls.append(f"generate:{body['model']}")
        self.loaded.add(body["model"])
        return web.json_response({"created": 0, "data": [{"b64_json": "aW1n"}]})

    async def edit(self, request: web.Request) -> web.Response:
        """Read an sd-server style multipart edit and record its fields and uploaded images."""
        form = await request.post()
        images = [field.file.read() for field in form.getall("image[]", [])]
        self.bodies.append({"model": form["model"], "prompt": form["prompt"], "images": images})
        self.calls.append(f"edit:{form['model']}")
        self.loaded.add(str(form["model"]))
        return web.json_response({"created": 0, "data": [{"b64_json": "ZWRpdA=="}]})


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
    """Return a factory that starts a gateway over ``name=FakeEngine-or-url`` pairs.

    An image engine publishes its fake's first model as its model id.
    """
    session = ClientSession()

    async def make(**engines: FakeEngine | str):
        parts, image_models = [], []
        for name, engine in engines.items():
            if isinstance(engine, FakeEngine):
                if name.startswith("image"):
                    image_models.append(f"{name}={engine.models[0]}")
                engine = (await aiohttp_server(engine.app())).make_url("")
            parts.append(f"{name}={engine}")
        built = gw.parse_engines(",".join(parts), session, ",".join(image_models))
        return await aiohttp_client(gw.build_app(built))

    yield make
    await session.close()


@pytest.fixture
async def client(make_client, llama, strata):
    """Gateway client routed to the fake llama and Strata engines."""
    return await make_client(llama=llama, strata=strata)


def _chat(model: str) -> dict:
    """Build a minimal streaming chat request body."""
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True}


def _generate(model: str) -> dict:
    """Build a minimal image generation request body."""
    return {"model": model, "prompt": "a fox", "size": "512x512"}


def _edit(model: str | None, images: list[bytes]) -> FormData:
    """Build an OpenAI-style multipart image edit request, as Open WebUI sends it."""
    form = FormData()
    if model is not None:
        form.add_field("model", model)
    form.add_field("prompt", "make it night")
    form.add_field("size", "1920x1280")
    for index, image in enumerate(images):
        form.add_field("image[]", image, filename=f"ref{index}.png", content_type="image/png")
    return form


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


async def test_two_strata_engines_swap_like_any_other(make_client, strata):
    """A hyphenated name is a second engine of the prefix's type with its own GPU turn."""
    swift = FakeEngine(["swift"])
    client = await make_client(**{"strata": strata, "strata-swift": swift})

    await _complete(client, "flash")
    await _complete(client, "swift")

    assert strata.calls == ["chat:flash", "unload"]
    assert swift.calls == ["chat:swift"]
    assert (await (await client.get("/health")).json())["active"] == "strata-swift"


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


async def test_image_engine_serves_generations_but_stays_unlisted(make_client, llama):
    """An image model routes generations and is absent from the chat model list."""
    image = FakeEngine(["qwen-image"])
    client = await make_client(llama=llama, **{"image-qwen": image})

    listing = await (await client.get("/v1/models")).json()
    response = await client.post("/v1/images/generations", json=_generate("qwen-image"))

    assert [m["id"] for m in listing["data"]] == ["qwen", "gemma"]
    assert response.status == 200
    assert (await response.json())["data"] == [{"b64_json": "aW1n"}]
    assert image.bodies == [_generate("qwen-image")]


async def test_image_engine_swaps_with_text_engines(make_client, llama):
    """Generating an image takes the GPU from llama, and the next chat takes it back."""
    image = FakeEngine(["qwen-image"])
    client = await make_client(llama=llama, **{"image-qwen": image})

    await _complete(client, "qwen")
    await client.post("/v1/images/generations", json=_generate("qwen-image"))
    await _complete(client, "gemma")

    assert llama.calls == ["chat:qwen", "unload:qwen", "chat:gemma"]
    assert image.calls == ["generate:qwen-image", "unload"]


@pytest.mark.parametrize(
    ("path", "body", "model"),
    [
        ("/v1/chat/completions", _chat("qwen-image"), "qwen-image"),
        ("/v1/images/generations", _generate("qwen"), "qwen"),
    ],
    ids=["chat-to-image", "image-to-chat"],
)
async def test_wrong_path_for_model_returns_400(make_client, llama, path, body, model):
    """A model requested on a path its engine does not serve fails before any swap."""
    image = FakeEngine(["qwen-image"])
    client = await make_client(llama=llama, **{"image-qwen": image})

    response = await client.post(path, json=body)

    assert response.status == 400
    assert f"model {model!r} does not serve {path}" in (await response.json())["error"]["message"]
    assert llama.calls == image.calls == []


async def test_image_edit_upload_reaches_image_engine(make_client, llama):
    """A multipart edit routes by its model field and arrives with every reference image."""
    image = FakeEngine(["qwen-image"])
    client = await make_client(llama=llama, **{"image-qwen": image})

    response = await client.post("/v1/images/edits", data=_edit("qwen-image", [b"one", b"two"]))

    assert response.status == 200
    assert (await response.json())["data"] == [{"b64_json": "ZWRpdA=="}]
    assert image.bodies == [
        {"model": "qwen-image", "prompt": "make it night", "images": [b"one", b"two"]}
    ]


@pytest.mark.parametrize(
    ("model", "status", "message"),
    [
        (None, 400, "has no model"),
        ("qwen", 400, "model 'qwen' does not serve /v1/images/edits"),
    ],
    ids=["no-model", "chat-model"],
)
async def test_image_edit_upload_without_image_model_is_rejected(
    make_client, llama, model, status, message
):
    """An edit upload with no model field, or naming a chat model, fails before any swap."""
    image = FakeEngine(["qwen-image"])
    client = await make_client(llama=llama, **{"image-qwen": image})

    response = await client.post("/v1/images/edits", data=_edit(model, [b"one"]))

    assert response.status == status
    assert message in (await response.json())["error"]["message"]
    assert llama.calls == image.calls == []


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


def _list_nothing(client, engine: FakeEngine) -> None:
    """Make the fake list no models, as Strata does while it loads one."""
    engine.models = []


def _go_offline(client, engine: FakeEngine) -> None:
    """Point the gateway's strata engine at a closed port, as during a container restart."""
    strata_engine = next(e for e in client.app["gateway"].engines if e.name == "strata")
    strata_engine.url = "http://127.0.0.1:9"


@pytest.mark.parametrize("silence", [_list_nothing, _go_offline], ids=["empty", "unreachable"])
async def test_engine_keeps_its_last_listing_while_silent(make_client, llama, strata, silence):
    """A listed engine that stops listing keeps its models listed and routed to it."""
    client = await make_client(llama=llama, strata=strata)
    await client.get("/v1/models")
    silence(client, strata)

    models = await (await client.get("/v1/models")).json()
    owner = await client.app["gateway"].owner("flash")

    assert [m["id"] for m in models["data"]] == ["qwen", "gemma", "flash"]
    assert owner is not None and owner.name == "strata"


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
        ("llama", "is not name=value"),
        ("vllm=http://x", "unknown engine type 'vllm'"),
        ("vllm-strata=http://x", "unknown engine type 'vllm'"),
        ("strata=http://a,strata=http://b", "listed twice"),
    ],
    ids=["empty", "no-url", "unknown-kind", "unknown-prefix", "duplicate"],
)
async def test_parse_engines_rejects_bad_spec(spec, message):
    """Malformed LOKI_ENGINES values raise ValueError naming the problem."""
    async with ClientSession() as session:
        with pytest.raises(ValueError, match=message):
            gw.parse_engines(spec, session)


@pytest.mark.parametrize(
    ("spec", "image_models", "message"),
    [
        ("image-qwen=http://x", "", "has no model id"),
        ("llama=http://x", "image-qwen=q", "names unknown engines: image-qwen"),
        ("image-qwen=http://x", "image-qwen", "is not name=value"),
    ],
    ids=["missing-id", "extra-id", "malformed-id"],
)
async def test_parse_engines_pairs_image_engines_with_ids(spec, image_models, message):
    """Every image engine needs exactly one LOKI_IMAGE_MODELS entry, and vice versa."""
    async with ClientSession() as session:
        with pytest.raises(ValueError, match=message):
            gw.parse_engines(spec, session, image_models)


@pytest.fixture
async def make_gateway(aiohttp_server):
    """Return a factory that builds a Gateway over ``name=FakeEngine`` pairs."""
    session = ClientSession()

    async def make(**engines: FakeEngine) -> gw.Gateway:
        parts = [
            f"{name}={(await aiohttp_server(e.app())).make_url('')}" for name, e in engines.items()
        ]
        return gw.Gateway(gw.parse_engines(",".join(parts), session))

    yield make
    await session.close()


@pytest.fixture
def fast_preload(monkeypatch):
    """Retry the preload's model lookup quickly and give up after half a second."""
    monkeypatch.setattr(gw, "PRELOAD_RETRY", 0.01)
    monkeypatch.setattr(gw, "PRELOAD_TIMEOUT", 0.5)


async def test_preload_unloads_other_engines_and_generates_one_token(
    make_gateway, llama, strata, fast_preload
):
    """The preload takes the GPU like a request and asks the model for a single token."""
    llama.loaded.add("qwen")
    gateway = await make_gateway(llama=llama, strata=strata)

    await gw.preload(gateway, "flash")

    assert llama.calls == ["unload:qwen"]
    assert strata.calls == ["chat:flash"]
    assert strata.bodies[0]["max_tokens"] == 1
    assert gateway.arbiter.active == "strata"
    assert gateway.arbiter.inflight == 0


async def test_preload_waits_for_the_engine_to_list_the_model(make_gateway, fast_preload):
    """An engine that is still starting lists nothing, and the preload retries until it does."""
    starting = FakeEngine([])
    gateway = await make_gateway(strata=starting)

    async def finish_starting() -> None:
        await asyncio.sleep(0.05)
        starting.models.append("flash")

    await asyncio.gather(gw.preload(gateway, "flash"), finish_starting())

    assert starting.calls == ["chat:flash"]


async def test_preload_gives_up_on_a_model_no_engine_serves(make_gateway, llama, fast_preload):
    """An unknown model id is logged and skipped once the wait times out."""
    gateway = await make_gateway(llama=llama)

    await gw.preload(gateway, "missing")

    assert llama.calls == []
    assert gateway.arbiter.active is None


async def test_preload_leaves_the_gpu_to_an_earlier_request(make_gateway, llama, strata):
    """A request that claimed the GPU before the preload keeps its model loaded."""
    gateway = await make_gateway(llama=llama, strata=strata)
    gateway.arbiter.active = "llama"

    await gw.preload(gateway, "flash")

    assert llama.calls == []
    assert strata.calls == []


async def test_preload_skips_image_models(make_gateway, aiohttp_server, fast_preload):
    """Only text models are preloaded, since an image engine needs a prompt to generate."""
    image = FakeEngine(["qwen-image"])
    session = ClientSession()
    url = (await aiohttp_server(image.app())).make_url("")
    gateway = gw.Gateway(gw.parse_engines(f"image-qwen={url}", session, "image-qwen=qwen-image"))

    await gw.preload(gateway, "qwen-image")
    await session.close()

    assert image.calls == []


async def test_app_preloads_its_model_at_startup(
    aiohttp_client, aiohttp_server, strata, fast_preload
):
    """A gateway built with a preload model loads it without any client request."""
    session = ClientSession()
    url = (await aiohttp_server(strata.app())).make_url("")
    app = gw.build_app(gw.parse_engines(f"strata={url}", session), "flash")

    await aiohttp_client(app)
    for _ in range(100):
        if strata.calls:
            break
        await asyncio.sleep(0.01)
    await session.close()

    assert strata.calls == ["chat:flash"]


async def _dashboard_models(client, path: str = "/strata/metrics") -> list[str]:
    """Read the dashboard's metrics through the gateway and return the answering engine's models."""
    response = await client.get(path)
    assert response.status == 200
    return (await response.json())["models"]


async def test_dashboard_follows_the_strata_engine_holding_the_gpu(make_client, llama, strata):
    """/strata/ reaches the live Strata engine, passing the path and query through."""
    swift = FakeEngine(["swift"])
    client = await make_client(llama=llama, strata=strata, **{"strata-swift": swift})

    await _complete(client, "swift")

    assert await _dashboard_models(client, "/strata/metrics?requests=all") == ["swift"]
    assert swift.calls[-1] == "metrics:requests=all"


async def test_dashboard_keeps_the_last_strata_engine_while_another_kind_holds_the_gpu(
    make_client, llama, strata
):
    """An image or llama model on the GPU leaves the dashboard on the last Strata engine used."""
    swift = FakeEngine(["swift"])
    image = FakeEngine(["qwen-image"])
    client = await make_client(
        llama=llama, strata=strata, **{"strata-swift": swift, "image-qwen": image}
    )

    await _complete(client, "swift")
    await _complete(client, "flash")
    await client.post("/v1/images/generations", json=_generate("qwen-image"))
    await _complete(client, "qwen")

    assert await _dashboard_models(client) == ["flash"]


async def test_dashboard_starts_on_the_first_strata_engine(make_client, llama, strata):
    """Before any Strata engine has held the GPU, the dashboard shows the first one."""
    client = await make_client(
        llama=llama, strata=strata, **{"strata-swift": FakeEngine(["swift"])}
    )

    assert await _dashboard_models(client) == ["flash"]


async def test_dashboard_rejects_writes(client, strata):
    """The dashboard's chat and settings writes never reach the engine."""
    response = await client.post("/strata/v1/chat/completions", json=_chat("flash"))

    assert response.status == 405
    assert strata.calls == []


async def test_dashboard_redirects_to_the_trailing_slash(client):
    """/strata redirects to /strata/, which the dashboard's relative paths need."""
    response = await client.get("/strata", allow_redirects=False)

    assert response.status == 308
    assert response.headers["Location"] == "/strata/"


async def test_dashboard_without_strata_engines_is_not_found(make_client, llama):
    """A stack without Strata has no dashboard to serve."""
    client = await make_client(llama=llama)

    response = await client.get("/strata/")

    assert response.status == 404


async def test_dashboard_page_opens_on_the_monitor_tab_without_chat(client):
    """The start page sets #monitor before the dashboard's script runs and hides the Chat tab."""
    response = await client.get("/strata/")
    page = await response.text()

    assert response.status == 200
    assert page.startswith("<!doctype html><html><head><script>")
    assert "'#monitor'" in page
    assert '.st-tab[data-tab="chat"] { display: none; }' in page
    assert "<title>flash</title>" in page
