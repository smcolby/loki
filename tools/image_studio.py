"""Open WebUI pipe function: Image Studio models that send prompts straight to Qwen Image 2.1.

Paste the contents of this file into the Open WebUI function editor
(Admin Panel → Functions → +) with the ID ``image_studio``. It adds three
models, each sending the user's message and images unchanged to the image
engine behind the loki gateway, with no chat model in between:

- **Draft**: 1248x832 at 40 steps, a quick preview of a production image.
- **Production**: 1920x1280 at 40 steps.
- **Refine**: edits the most recent image in the chat at that image's size.

Draft and Production use the images attached to the current message as
references, or none. Refine edits the most recent earlier image in the chat
(a generated one or an upload), adding any images attached to the current
message as extra references. Results are stored as Open WebUI files on the
assistant message, so Refine can pick them up on the next turn.

Open WebUI's modules are imported when a request runs, since they exist only
inside the Open WebUI container.
"""

import base64
import importlib
import io
import json
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

GATEWAY_URL = "http://gateway:8080/v1"
IMAGE_MODEL = "qwen-image-2.1-q8_0"
MAX_REFERENCES = 10
MAX_PIXELS = 1920 * 1280
SIZE_MULTIPLE = 32
REQUEST_TIMEOUT_S = 1800
MAX_SEED = 2**31
EXTRA_ARGS = re.compile(r"<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>", re.DOTALL)


@dataclass(frozen=True)
class Studio:
    """One Image Studio model.

    Attributes
    ----------
    name : str
        Model name shown in Open WebUI.
    steps : int
        Sampling steps per image.
    size : tuple of int or None
        Output width and height, or None to follow the image being refined.
    refine : bool
        Whether the studio edits the most recent image in the chat.
    """

    name: str
    steps: int
    size: tuple[int, int] | None
    refine: bool = False


STUDIOS = {
    "draft": Studio("Image Studio: Draft", 40, (1248, 832)),
    "production": Studio("Image Studio: Production", 40, (1920, 1280)),
    "refine": Studio("Image Studio: Refine", 40, None, refine=True),
}


def refine_size(width: int, height: int) -> tuple[int, int]:
    """Return the output size for refining a ``width`` x ``height`` image.

    The size keeps the image's aspect ratio, shrinks it to ``MAX_PIXELS`` when
    larger (the largest size the engine decodes without tiling), and rounds
    each side to a multiple of ``SIZE_MULTIPLE``.

    Parameters
    ----------
    width, height : int
        Size of the image being refined, in pixels.

    Returns
    -------
    tuple of int
        Output width and height.
    """
    scale = min(1.0, (MAX_PIXELS / (width * height)) ** 0.5)

    def fit(side: int) -> int:
        return max(SIZE_MULTIPLE, round(side * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)

    return fit(width), fit(height)


def message_images(message: dict[str, Any]) -> list[str]:
    """Return the URLs of the images attached to one chat message."""
    urls = []
    for file in message.get("files") or []:
        is_image = file.get("type") == "image" or str(file.get("content_type", "")).startswith(
            "image/"
        )
        if is_image and file.get("url"):
            urls.append(file["url"])
    return urls


def select_images(studio: Studio, messages: list[dict[str, Any]]) -> list[str]:
    """Return the image URLs to send as references, the edited image first.

    Parameters
    ----------
    studio : Studio
        The studio handling the request.
    messages : list of dict
        The chat's messages, oldest first, ending with the current user message.

    Returns
    -------
    list of str
        Image URLs, at most ``MAX_REFERENCES``. Empty for a text-only request.

    Raises
    ------
    ValueError
        If Refine finds no image to edit.
    """
    current = message_images(messages[-1]) if messages else []
    if not studio.refine:
        return current[:MAX_REFERENCES]

    # Refine edits the newest earlier image, or the first attachment without one
    earlier = next(
        (images for images in map(message_images, reversed(messages[:-1])) if images), []
    )
    target = earlier[:1] or current[:1]
    if not target:
        raise ValueError("Refine needs an image to edit: generate one or attach one first.")
    extras = current if earlier else current[1:]
    return (target + extras)[:MAX_REFERENCES]


def thread_end(
    messages_map: dict[str, dict[str, Any]], message_id: str | None, current_id: str | None
) -> str | None:
    """Return the id of the message whose ancestry is the conversation being answered.

    That is the user message the reply ``message_id`` answers: its ``parentId``,
    or the message listing it as a child when the reply was stored without a
    parent. Falls back to the chat's ``current_id``.
    """
    reply = messages_map.get(message_id or "") or {}
    if reply.get("parentId"):
        return reply["parentId"]
    for key, message in messages_map.items():
        if message_id and message_id in (message.get("childrenIds") or []):
            return key
    return current_id


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated with ``override``, merging nested dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_prompt(prompt: str, steps: int, seed: int) -> tuple[str, dict[str, Any]]:
    """Attach sd-server's per-request arguments to ``prompt`` as one tag.

    Any ``<sd_cpp_extra_args>`` tags already in the prompt are removed and
    merged into the studio's step count and seed, with the prompt's values
    taking precedence, so a user can pin a seed or change the steps.

    Parameters
    ----------
    prompt : str
        The user's message.
    steps : int
        The studio's sampling steps.
    seed : int
        The seed to use unless the prompt sets one.

    Returns
    -------
    tuple of (str, dict)
        The prompt with a single merged tag, and the merged arguments.

    Raises
    ------
    ValueError
        If a tag in the prompt does not contain a JSON object.
    """
    user_args: dict[str, Any] = {}
    for match in EXTRA_ARGS.finditer(prompt):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"<sd_cpp_extra_args> is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("<sd_cpp_extra_args> must contain a JSON object")
        user_args = _merge(user_args, parsed)

    args = _merge({"seed": seed, "sample_params": {"sample_steps": steps}}, user_args)
    text = EXTRA_ARGS.sub("", prompt).rstrip()
    return f"{text}\n<sd_cpp_extra_args>{json.dumps(args)}</sd_cpp_extra_args>", args


def message_text(message: dict[str, Any]) -> str:
    """Return a message's text, joining the text parts of multipart content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def chat_title(task_prompt: str) -> str:
    """Return a chat title taken from the user's message in a title task's prompt.

    Open WebUI's title prompt quotes the conversation between ``<chat_history>``
    tags as ``USER:`` and ``ASSISTANT:`` lines. The title is the first sentence
    of the user's first line that is not a Markdown heading, up to six words.
    """
    history = task_prompt.split("<chat_history>")[-1].split("</chat_history>")[0]
    user = history.split("USER:", 1)[-1].split("\nASSISTANT:", 1)[0]
    for line in user.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sentence = line.split(". ", 1)[0].replace("*", "").replace('"', "")
        words = sentence.split()[:6]
        if words:
            return " ".join(words).rstrip(".,:;!?")
    return "Image"


def task_reply(task: str, prompt: str) -> str:
    """Answer an Open WebUI background task without generating an image."""
    if task == "title_generation":
        return json.dumps({"title": chat_title(prompt)})
    if task == "tags_generation":
        return json.dumps({"tags": []})
    if task == "follow_up_generation":
        return json.dumps({"follow_ups": []})
    if task == "query_generation":
        return json.dumps({"queries": []})
    return ""


class Pipe:
    """Open WebUI manifold pipe offering the Image Studio models."""

    def pipes(self) -> list[dict[str, str]]:
        """List the studios as Open WebUI models."""
        return [{"id": key, "name": studio.name} for key, studio in STUDIOS.items()]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Any,
        __metadata__: dict,
        __task__: str | None = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Generate or edit one image and attach it to the assistant message.

        Parameters
        ----------
        body : dict
            The chat completion request; ``model`` ends with the studio key.
        __user__, __request__, __metadata__, __task__, __event_emitter__
            Context Open WebUI passes to pipe functions.

        Returns
        -------
        str
            A one-line summary of the result, or an error message.
        """
        studio = STUDIOS.get(str(body.get("model", "")).rsplit(".", 1)[-1])
        messages = body.get("messages") or []
        prompt = message_text(messages[-1]) if messages else ""
        if __task__:
            return task_reply(__task__, prompt)
        if studio is None:
            return f"Unknown Image Studio model {body.get('model')!r}."

        chats = importlib.import_module("open_webui.models.chats").Chats
        users = importlib.import_module("open_webui.models.users").Users
        user = await users.get_user_by_id(__user__["id"])
        chat_id, message_id = __metadata__.get("chat_id"), __metadata__.get("message_id")
        chat = await chats.get_chat_by_id(chat_id) if chat_id else None
        if chat is None:
            return "Image Studio needs a saved chat; temporary chats are not supported."

        history = chat.chat.get("history", {})
        messages_map = history.get("messages", {})
        thread = importlib.import_module("open_webui.utils.misc").get_message_list(
            messages_map, thread_end(messages_map, message_id, history.get("currentId"))
        )
        # Drop any assistant message after the user's request, so the request is last
        while thread and thread[-1].get("role") != "user":
            thread.pop()
        try:
            urls = select_images(studio, thread)
            request_prompt, args = build_prompt(prompt, studio.steps, secrets.randbelow(MAX_SEED))
        except ValueError as exc:
            return str(exc)

        images = [await self._load_image(url, user) for url in urls]
        size = studio.size or refine_size(*self._image_size(images[0]))
        label = f"{studio.name.split(': ')[1]}, {size[0]}x{size[1]}, {len(images)} reference(s)"
        await self._status(__event_emitter__, f"{label} ...", done=False)

        started = time.monotonic()
        try:
            png = await self._generate(request_prompt, size, images)
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            await self._status(__event_emitter__, "Image generation failed", done=True)
            return f"Image generation failed: {exc}"
        seconds = time.monotonic() - started

        upload_image = importlib.import_module("open_webui.routers.images").upload_image
        metadata = {"chat_id": chat_id, "message_id": message_id}
        _, file = await upload_image(__request__, png, "image/png", metadata, user)
        if __event_emitter__ is not None:
            await __event_emitter__(
                {"type": "files", "data": {"files": [{"type": "image", **file}]}}
            )
        await self._status(__event_emitter__, f"{label}, {seconds:.0f} s", done=True)
        steps = args.get("sample_params", {}).get("sample_steps", studio.steps)
        return f"{label}, {steps} steps, seed {args['seed']}, {seconds:.0f} s"

    async def _generate(
        self, prompt: str, size: tuple[int, int], images: list[tuple[bytes, str]]
    ) -> bytes:
        """Request one image from the gateway and return its PNG bytes."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        size_field = f"{size[0]}x{size[1]}"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if images:
                form = aiohttp.FormData()
                for key, value in (
                    ("model", IMAGE_MODEL),
                    ("prompt", prompt),
                    ("size", size_field),
                    ("response_format", "b64_json"),
                ):
                    form.add_field(key, value)
                for index, (data, content_type) in enumerate(images):
                    form.add_field(
                        "image[]", data, filename=f"ref{index}", content_type=content_type
                    )
                request = session.post(f"{GATEWAY_URL}/images/edits", data=form)
            else:
                payload = {
                    "model": IMAGE_MODEL,
                    "prompt": prompt,
                    "size": size_field,
                    "response_format": "b64_json",
                }
                request = session.post(f"{GATEWAY_URL}/images/generations", json=payload)
            async with request as response:
                body = await response.json(content_type=None)
                if response.status != 200 or not body.get("data"):
                    raise ValueError(f"HTTP {response.status}: {json.dumps(body)[:300]}")
                return base64.b64decode(body["data"][0]["b64_json"])

    async def _load_image(self, url: str, user: Any) -> tuple[bytes, str]:
        """Read an Open WebUI file or data URL into bytes and a content type."""
        if url.startswith("data:"):
            header, encoded = url.split(",", 1)
            return base64.b64decode(encoded), header[5:].split(";")[0] or "image/png"
        file_id = url.split("/api/v1/files/")[-1].split("/content")[0]
        files = importlib.import_module("open_webui.routers.files")
        response = await files.get_file_content_by_id(file_id, user)
        path = getattr(response, "path", None)
        if path is None:
            raise ValueError(f"Image {url} is not available")
        return Path(path).read_bytes(), getattr(response, "media_type", None) or "image/png"

    def _image_size(self, image: tuple[bytes, str]) -> tuple[int, int]:
        """Return the pixel size of an image's bytes."""
        pil = importlib.import_module("PIL.Image")
        with pil.open(io.BytesIO(image[0])) as opened:
            return opened.size

    async def _status(self, emitter: Any, description: str, done: bool) -> None:
        """Show a status line on the assistant message."""
        if emitter is not None:
            await emitter({"type": "status", "data": {"description": description, "done": done}})
