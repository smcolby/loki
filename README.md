![banner](assets/banner.jpg)

# LOKI: Local Offline Knowledge Index

LOKI is a self-hosted AI stack that gives you a private, fully offline knowledge base powered by a local LLM. It combines [Ollama](https://ollama.com) (local inference), [Open WebUI](https://openwebui.com) (chat interface), and [Kiwix](https://kiwix.org) (offline Wikipedia and other knowledge archives) — all orchestrated with Docker Compose and managed through a single CLI.

Whether you're working air-gapped, want to keep queries off the cloud, or just want an always-available research assistant, LOKI runs entirely on your own hardware with no external dependencies at runtime.

> **Linux only.** LOKI targets Linux systems with systemd. `loki setup` installs and configures all prerequisites automatically.

## Prerequisites

- A Linux system with systemd
- [`pipx`](https://pipx.pypa.io/stable/installation/)
- `curl`

Everything else — Docker, Ollama, aria2, avahi-daemon — is installed automatically by `loki setup`.

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
  ollama: 11434  # Native Ollama service.

kiwix_files:
  - name: wikipedia_en_all_nopic
    url: https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2025-12.zim

ollama_models:
  - qwen3.5:9b
```

The defaults work out of the box. Edit `kiwix_files` and `ollama_models` to match what you want downloaded. `loki setup` generates the `Caddyfile` and `.env` automatically — do not edit those files by hand.

## Setup

Run once after configuring:

```bash
loki setup
```

This will:

1. **Display `config.yaml`** and ask you to confirm before proceeding.
2. **Install system packages** (aria2, avahi-daemon, avahi-utils) via `apt-get` or `dnf`.
3. **Install Docker** via the official convenience script and add your user to the `docker` group.
4. **Install Ollama** via the official install script.
5. **Configure Ollama network binding** — creates a systemd override so Ollama listens on all interfaces (required for Docker containers to reach it via `host.docker.internal`).
6. **Add `LOKI_ROOT` to your shell profile** so `loki` commands work from any directory.
7. **Generate the `Caddyfile` and `.env`** from your configuration.
8. **Download ZIM files** listed in `kiwix_files` using aria2.

Each step prompts `[Y/n]`. Skipped steps must be completed manually before the stack will function correctly.

## Usage

```
loki start    Pull Ollama models, start the Docker Compose stack, and broadcast hostname via mDNS.
loki stop     Stop the Docker Compose stack and terminate the mDNS broadcast.
loki status   Check the health of running services.
loki cleanup  Remove ZIM files and Ollama models no longer listed in config.
```

After `loki start`, Open WebUI is available at `http://loki.local` (or whichever `url` you configured).

## Connecting Kiwix to Open WebUI

A ready-made Open WebUI tool definition lives at [`tools/kiwix_tool.py`](tools/kiwix_tool.py). It exposes the Kiwix server to the LLM as a callable tool, communicating over the Docker internal network so it works regardless of your configured host port.

To load it:

1. Open `http://loki.local` and navigate to **Admin Panel → Tools**, then click **+**.
2. Paste the full contents of `tools/kiwix_tool.py` into the editor and save.
3. Go to **Admin Panel → Models**, select your model, and enable the Kiwix tool under the **Tools** tab.

### Enabling native tool calling

By default, Open WebUI injects tool definitions into the system prompt, which is unreliable with smaller models. For best results, enable native tool calling:

1. **Admin Panel → Models** → select your model.
2. Under **Advanced Parameters**, set **Tool Calling** to **Native**.
3. Save.

> **Note:** Native tool calling requires a model fine-tuned for function calling (e.g. `qwen3`, `mistral-nemo`, `llama3.1`). If responses degrade after enabling it, the model may not support the feature — revert to the default setting.

---

## Advanced

### Local hostname resolution

The default `url: loki.local` uses the `.local` TLD, which is broadcast via **mDNS** and resolves automatically on your LAN without any router configuration. `loki setup` installs `avahi-daemon` for this. When `loki start` runs, it spawns `avahi-publish-address` to announce the hostname for as long as the stack is running; `loki stop` terminates the announcement.

If you prefer a non-`.local` hostname (e.g. `loki.home`), set it in `config.yaml` and add a static entry to `/etc/hosts` on each client device — the mDNS broadcast is skipped automatically for non-`.local` hostnames.

### LOKI_ROOT

By default, loki resolves all paths (`config.yaml`, `Caddyfile`, `.env`, `data/kiwix/`) relative to the current working directory. Run every `loki` command from the repository root, or set `LOKI_ROOT` to point elsewhere:

```bash
export LOKI_ROOT=/path/to/loki
loki setup
```

`loki setup` offers to add this export to your shell profile automatically (step 6).

### Manual Ollama network binding

If you skipped step 5 during setup, configure it manually:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```
