![banner](assets/banner.jpg)

# `loki`: local offline knowledge index

`loki` is a self-hosted AI stack that gives you a private, fully offline knowledge base powered by a local LLM. It combines [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (local inference), [Open WebUI](https://openwebui.com) (chat interface), and [Kiwix](https://kiwix.org) (offline Wikipedia and other knowledge archives), all orchestrated with Docker Compose and managed through a single CLI. An optional second engine, [Strata](https://github.com/Niko1221/Strata), serves Qwen3.8-Flash-Next behind the same API, and an optional [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) engine generates images.

Whether you're working air-gapped, want to keep queries off the cloud, or just want an always-available research assistant, `loki` runs entirely on your own hardware with no external dependencies at runtime.

> **Linux only.** `loki` targets Linux systems with systemd. `loki setup` installs and configures all prerequisites automatically.

> **AMD GPUs only.** `llama-server` is built against ROCm for one AMD GPU architecture. Install the `amdgpu` kernel driver yourself, ideally _before_ installing `loki`; the container bundles the ROCm user-space libraries it needs.

## Prerequisites

- A Linux system with systemd and an AMD GPU (`/dev/kfd` and `/dev/dri` present)
- [`pipx`](https://pipx.pypa.io/stable/installation/)
- `curl`

`loki setup` installs everything else (Docker, aria2, avahi-daemon).

## Installation

Clone the repository and install the CLI:

```bash
git clone https://github.com/smcolby/loki.git
cd loki
pipx install -e .
```

## Configuration

Open `config.yaml` and adjust the settings before running `loki setup`:

```yaml
url: loki.local              # Hostname used to reach the server on your local network.

ports:
  caddy: 80      # Caddy reverse proxy (Open WebUI).
  kiwix: 8080    # Kiwix offline knowledge server.
  api: 8090      # Model API for every engine (OpenAI and Anthropic compatible).

llama:
  models_dir: ~/.llms                               # One subdirectory per model.
  ref: 5d806aa2575e01e126651fd69ab1ab6cefff861d     # llama.cpp commit to build.
  gpu_targets: gfx1100                              # AMD GPU architecture (rocminfo | grep gfx).
  rocm_version: 7.2.4                               # ROCm toolchain and bundled runtime.
  max_loaded: 1                                     # Models kept in VRAM at once.
  defaults:                                         # Options every model gets.
    n-gpu-layers: 99
    flash-attn: "on"

strata:
  engines: {}                                       # Engine name -> config file (see below).
  data_dir: ~/.llms/strata                          # Holds the engine configs and model files.
  ref: 1678de333d0e0711bc414ad992b640e1a37dd814     # Strata commit to build.
  gpu_targets: gfx1100
  rocm_version: 7.10.0a20251120                     # TheRock ROCm wheels.

image:
  engines: {}                                       # Engine name -> model id and sd-server args.
  models_dir: ~/.llms                               # Mounted read-only into every image engine.
  ref: 3f8527a46c54ecf4cb4ed6003da8e8982283c73c     # stable-diffusion.cpp commit to build.
  gpu_targets: gfx1100
  rocm_version: 7.2.4

kiwix_files:
  - name: wikipedia_en_all_nopic
    url: https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2025-12.zim
```

`loki/config.default.yaml` lists the full default `llama.defaults` block. Keys under `defaults` are `llama-server` long option names without the leading dashes. Quote `"on"` and `"off"` so YAML keeps them as strings.

Edit `kiwix_files` (datasets [here](https://download.kiwix.org/zim/)) to match what you want downloaded. `loki` generates the `Caddyfile`, `.env`, `models.ini`, and `compose.engines.yaml`; do not edit those files by hand.

## Adding models

`loki` serves hand-curated GGUF files; it never downloads models. Each subdirectory of `llama.models_dir` that contains a `preset.ini` becomes one or more models. A section header names the model id clients request, and its options are `llama-server` long option names:

```ini
; ~/.llms/gemma4-31b-iq4xs/preset.ini
[gemma4-31b-iq4xs]
model = gemma-4-31B-it-IQ4_XS.gguf
model-draft = mtp-gemma-4-31B-it-Q8_0.gguf
spec-type = draft-mtp
ctx-size = 163840
temp = 1.0
```

Relative `model`, `model-draft`, `mmproj`, and `chat-template-file` paths resolve against the preset's directory, and each file must exist. A preset may declare several sections (for example the same GGUF at two context sizes), but a model id may appear only once across all presets. Global settings belong in `llama.defaults`, so a `[*]` section in a preset is an error.

`loki start` assembles every preset into `models.ini` and lists the model ids it found. Restart the stack (`loki stop && loki start`) after adding or editing a preset.

`llama-server` loads a model on its first request and unloads the least recently used one when more than `max_loaded` would be resident. With `fit: "off"`, a preset whose context does not fit in VRAM fails to load with an error instead of silently shrinking.

### Adding Strata

Strata streams a mixture-of-experts model's experts from host memory, so it runs Qwen3.8-Flash-Next and its fine-tunes on a GPU too small to hold them. Each entry under `strata.engines` maps an engine name to an engine config file in `strata.data_dir`:

```yaml
strata:
  engines:
    qwen: qwen.json
    swift: swift.json
```

Each engine runs in its own container as Compose service `strata-<name>`, defined in the generated `compose.engines.yaml`. Engine names use lowercase letters, digits, `_`, and `-`. A config file looks like this:

```json
{
  "args": ["--pack", "/home/you/.llms/strata/packs/iq2_xs", "--native", "/home/you/.llms/strata/models/model-00001-of-00002.gguf", "--max-context", "131072"],
  "tokenizer": "/home/you/.llms/strata/packs/iq2_xs/tokenizer",
  "model_name": "qwen3.8-flash-next-iq2_xs",
  "env": {"STRATA_RESIDENT_PIN": "0"}
}
```

`args` are Strata engine arguments, `model_name` is the model id clients request, and `env` (optional) sets engine environment variables. Use absolute paths inside `data_dir`, which is mounted read-only at the same path. The image supplies the engine binary, its libraries, and a hipBLASLt tuning table for `gpu_targets`, so an engine config must not set `exe`, `cwd`, `lib_dirs`, `backend`, or `log`. `loki start` stops with an error if a listed engine config is missing. All engines share one image, and an engine that is not loaded holds no GPU memory and little host RAM.

Only one engine holds the GPU at a time. The first request for a model on the other engine waits for in-flight requests to finish, unloads the current engine's models, and then loads the requested one.

### Adding image generation

An image engine runs stable-diffusion.cpp's `sd-server` for one image model. Each entry under `image.engines` names the engine, the model id clients request, and the `sd-server` arguments:

```yaml
image:
  engines:
    qwen:
      model: qwen-image-2.1-q8_0
      args:
        - --diffusion-model
        - /home/you/.llms/qwen-image-2.1/qwen-image-2.1-Q8_0.gguf
        - --llm
        - /home/you/.llms/qwen-image-2.1/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf
        - --llm_vision
        - /home/you/.llms/qwen-image-2.1/Qwen3-VL-8B-Instruct-mmproj-F16.gguf
        - --vae
        - /home/you/.llms/qwen-image-2.1/qwen_image_2.1_vae_bf16.safetensors
        - --params-backend
        - disk
        - --strength
        - "1.0"
        - --width
        - "1920"
        - --height
        - "1280"
```

`--width` and `--height` set the size of a request that names none; both must be multiples of 32. `--llm_vision` and `--strength` matter only for edits (below). Each engine runs as Compose service `image-<name>` (container `loki-image-<name>`) in `compose.engines.yaml`, with `models_dir` mounted read-only at the same path, so use absolute paths inside it. Model ids use letters, digits, `.`, `_`, `:`, `/`, and `-`. All image engines share one image. sd-server keeps about 1.6 GiB of GPU memory after a generation, so a supervisor in the image starts it on the first request and stops it when another engine needs the GPU; the arguments must not set `--listen-ip` or `--listen-port`.

Image models answer `POST /v1/images/generations` and `POST /v1/images/edits` on the model API and stay out of `/v1/models`, which chat clients read as their model list. A chat request for an image model, or an image request for a chat model, fails with HTTP 400. An image request takes the GPU like any other engine swap: it waits for in-flight text requests, unloads the text engines, and then generates. The next chat request stops sd-server and reloads its model.

To generate images from Open WebUI, open **Admin Panel > Settings > Images**, choose the OpenAI engine, set the base URL to `http://gateway:8080/v1`, enter any API key, set the model to the image model id, and set the image size (Open WebUI always sends one, so the engine's default applies only to other clients).

Edits take reference images: `/v1/images/edits` is OpenAI's multipart form, with the images as `image[]` parts, and Qwen Image 2.1 accepts up to 10. Qwen Image reads them through its Qwen3-VL text encoder's vision weights, so the engine needs `--llm_vision` pointing at the matching `mmproj` file (Unsloth's `Qwen3-VL-8B-Instruct-GGUF` ships one); without it sd-server rejects every edit. `--strength 1.0` starts an edit from noise, as Qwen's own pipeline does: sd-server's default of 0.75 starts from the first image, so the result keeps its colors and lighting and largely ignores the instruction. Plain generations have no starting image, so `--strength` does not affect them. To edit from Open WebUI, enable **Image Editing** in the same settings page with the same engine, base URL, key, and model, and set an edit size: without one the output takes the first image's size, which for a phone photo is slow and too large to decode untiled. Attach images to a chat message with image generation on, and Open WebUI sends the images from the last two messages that have any.

## Setup

Run once after configuring:

```bash
loki setup
```

This will:

1. **Display `config.yaml`** and ask you to confirm before proceeding.
2. **Install system packages** (aria2, avahi-daemon, avahi-utils) via `apt-get` or `dnf`.
3. **Install Docker** via the official convenience script and add your user to the `docker` group.
4. **Check for AMD GPU devices** (`/dev/kfd` and `/dev/dri`) and warn if the `amdgpu` driver is missing.
5. **Add `LOKI_ROOT` to your shell profile** so `loki` commands work from any directory.
6. **Generate the `Caddyfile`, `.env`, and `models.ini`** from your configuration and model presets.
7. **Build missing local images**: the API gateway, `llama-server` (10 to 20 minutes), and the Strata and image engines when any is configured.
8. **Download ZIM files** listed in `kiwix_files` using aria2.

Steps that change your system prompt `[Y/n]`. Skipped steps must be completed manually before the stack will function correctly; `loki start` builds any image step 7 skipped.

## Usage

```
loki start    Write generated files, start the Docker Compose stack, and broadcast hostname via mDNS.
loki stop     Stop the Docker Compose stack and terminate the mDNS broadcast.
loki status   Check the health of running services and list model load states.
loki update   Upgrade system packages, pull Docker images, and build missing local images.
loki cleanup  Remove ZIM files and local images no longer matching config.
```

After `loki start`, Open WebUI is available at `http://loki.local` (or whichever `url` you configured).

Local image tags name every build input: `loki-llama:<gpu_targets>-rocm<rocm_version>-<ref>`, `loki-strata:<gpu_targets>-rocm<rocm_version>-<ref>`, `loki-image:<gpu_targets>-rocm<rocm_version>-<ref>`, and `loki-gateway:<hash of the gateway sources>`. `loki update` builds only images whose tag is missing, so an unchanged config rebuilds nothing. To pick up a new llama.cpp release, set `llama.ref` to the new commit and run `loki update`. `loki cleanup` then offers to remove the image built for the previous commit.

## Connecting other clients

A gateway listens on the LAN at `http://loki.local:8090` without an API key and routes each request to the engine serving the requested model. It exposes:

- An OpenAI-compatible API at `/v1` (`/v1/chat/completions`, `/v1/models`). `/v1/models` lists every engine's models, with the engine in `owned_by`. A model stays listed while its engine loads it or restarts, since Strata reports no models during a load. Clients that require a key accept any placeholder value.
- An Anthropic-compatible `/v1/messages` endpoint, so Claude Code can use it by setting `ANTHROPIC_BASE_URL=http://loki.local:8090`.
- `/health`, which reports each engine's state and which one holds the GPU.

Request reasoning depth with the OpenAI `reasoning_effort` field. A chat template may reject effort levels its model does not support; the request then fails with an error rather than running at a different level.

## Connecting Kiwix to Open WebUI

A ready-made Open WebUI tool definition lives at [`tools/kiwix_tool.py`](tools/kiwix_tool.py). It gives the model two functions, `search_article_titles` and `read_articles`, which reach the Kiwix server over the Docker network at `http://kiwix-serve:8080`, so it works regardless of your configured host port. Articles come back as plain text with headings, without citation markers, reference lists, or navigation boxes, and up to 30,000 characters each.

To load it:

1. Open `http://loki.local` and navigate to **Workspace → Tools**, then click **+**.
2. Paste the full contents of `tools/kiwix_tool.py` into the editor and save.
3. Enable the tool for each model: **Workspace → Models** (or **Admin Panel → Settings → Models**), edit the model, and check the tool under **Tools**. A tool that is installed but not enabled on a model is never offered to it.

Open WebUI calls tools natively by default (**Advanced Params → Function Calling**), which every model loki serves supports. To update the tool later, paste the new file over the old one in the same editor.

If web search is also enabled, the model may search the web instead of Kiwix; ask for "the offline Wikipedia" or disable web search in that chat to steer it.

---

## Advanced

### Local hostname resolution

The default `url: loki.local` uses the `.local` TLD, which is broadcast via **mDNS** and resolves automatically on your LAN without any router configuration. `loki setup` installs `avahi-daemon` for this. When `loki start` runs, it spawns `avahi-publish-address` to announce the hostname for as long as the stack is running; `loki stop` terminates the announcement.

If you prefer a non-`.local` hostname (e.g. `loki.home`), set it in `config.yaml` and add a static entry to `/etc/hosts` on each client device. `loki` skips the mDNS broadcast for non-`.local` hostnames.

### `LOKI_ROOT`

By default, loki resolves all paths (`config.yaml`, `Caddyfile`, `.env`, `models.ini`, `data/kiwix/`) relative to the current working directory. Run every `loki` command from the repository root, or set `LOKI_ROOT` to point elsewhere:

```bash
export LOKI_ROOT=/path/to/loki
loki setup
```

`loki setup` offers to add this export to your shell profile automatically (step 5).

### Building for a different GPU

Set `llama.gpu_targets` (and `strata.gpu_targets`) to your GPU's architecture (`rocminfo | grep gfx`, for example `gfx1201`) and run `loki update`. Each image keeps only the ROCm BLAS kernels for that architecture, so it holds a single target.

---

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

Linting mirrors CI exactly; run the same checks locally with:

```bash
ruff format --check .
ruff check .
pyright
```
