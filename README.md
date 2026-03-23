# loki
Local Offline Knowledge Index

A CLI tool for managing a local LLM and offline Wikipedia knowledge server orchestrated via Docker.

> **Linux only.** loki targets Linux systems with systemd. `loki setup` installs and configures all prerequisites automatically.

## Prerequisites

- A Linux system with systemd
- `pipx` for isolated Python package installation
- `curl` (for the Docker and Ollama installers)

Everything else — Docker, Ollama, aria2, avahi-daemon — is installed automatically by `loki setup`.

## Installation

Install system-wide with `pipx` to keep the virtual environment isolated:

```bash
pipx install -e .
```

## Configuration

Edit `config.yaml` before running `loki setup`:

```yaml
url: loki.local              # Hostname used to reach the server on your local network.

# Host ports exposed by the Docker Compose services.
# Change these if the defaults conflict with other services on your machine.
ports:
  caddy: 80      # Port for the Caddy reverse proxy (Open WebUI).
  kiwix: 8080    # Port for the Kiwix offline knowledge server.
  ollama: 11434  # Port for the native Ollama service (used for health checks).

kiwix_files:
  - name: wikipedia_en_top_maxi
    url: https://download.kiwix.org/zim/wikipedia/wikipedia_en_top_maxi_2024-10.zim

ollama_models:
  - qwen3:30b
```

`loki setup` generates the `Caddyfile` and `.env` automatically — do not edit them by hand.

### Local hostname resolution

Using the `.local` TLD (the default) means the hostname is broadcast via **mDNS** and resolves on your LAN without any router configuration. `loki setup` installs `avahi-daemon` and `avahi-utils` automatically.

When `loki start` runs, it spawns `avahi-publish-address` in the background to announce the configured hostname. The announcement is active for as long as the stack is running, and is terminated by `loki stop`.

If you prefer a different TLD (e.g. `loki.home`), set `url: loki.home` in `config.yaml` and add a static entry to `/etc/hosts` on each client device. The mDNS broadcast is skipped automatically for non-`.local` hostnames.

### LOKI_ROOT

By default, loki resolves all paths (`config.yaml`, `Caddyfile`, `.env`, `data/kiwix/`) relative to the **current working directory**. Run every `loki` command from the repository root, or set `LOKI_ROOT` to an explicit path:

```bash
export LOKI_ROOT=/path/to/loki
loki setup
```

`loki setup` can add this export to your shell profile automatically.

## Setup

Run `loki setup` once after cloning the repository. It will:

1. **Display `config.yaml`** and ask you to confirm before proceeding.
2. **Install system packages** (aria2, avahi-daemon, avahi-utils) via `apt-get` or `dnf`.
3. **Install Docker** via the official convenience script and add your user to the `docker` group.
4. **Install Ollama** via the official install script.
5. **Configure Ollama network binding** — creates a systemd override so Ollama listens on all interfaces (required for Docker containers to reach it via `host.docker.internal`).
6. **Add `LOKI_ROOT` to your shell profile** so `loki` commands work from any directory.
7. **Generate the `Caddyfile` and `.env`** from your configuration.
8. **Download ZIM files** listed in `kiwix_files` using aria2.

For each step you will be prompted `[Y/n]`. If you decline a step, loki prints a notice and skips it — you will need to complete that step manually before the stack will function correctly.

> **If you prefer to set up manually**, consult the [Docker installation docs](https://docs.docker.com/engine/install/), the [Ollama Linux guide](https://ollama.com/download/linux), and the systemd override instructions below.

### Manual Ollama network binding

If you skipped step 5, create the override manually:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Usage

```
loki setup    Install prerequisites, generate config files, and download ZIM files.
loki start    Pull Ollama models, start the Docker Compose stack, and broadcast hostname via mDNS.
loki stop     Stop the Docker Compose stack and terminate the mDNS broadcast.
loki status   Check the health of running services.
loki cleanup  Remove ZIM files and Ollama models no longer listed in config.
```

## Connecting Kiwix to Open WebUI

A ready-made Open WebUI tool definition lives at [`tools/kiwix_tool.py`](tools/kiwix_tool.py). It exposes the Kiwix server to the LLM as a callable tool, communicating over the Docker internal network so it works regardless of your configured host port.

To load it into Open WebUI:

1. Open the Open WebUI interface (e.g. `http://loki.local`).
2. Navigate to **Admin Panel → Tools** and click **+** to create a new tool.
3. Copy the full contents of `tools/kiwix_tool.py` and paste them into the editor.
4. Save the tool, then open the model's settings under **Admin Panel → Models**, select your model, and enable the Kiwix tool under the **Tools** tab.

### Enabling native tool calling

By default, Open WebUI injects tool definitions into the system prompt, which is unreliable with smaller models. For best results, enable native tool calling so the model uses its built-in function-calling interface instead:

1. Open **Admin Panel → Models** and select your model.
2. Under **Advanced Parameters**, set **Tool Calling** to **Native**.
3. Save. The model will now receive tool definitions as structured function schemas rather than system prompt text.

> **Note:** Native tool calling requires a model fine-tuned for function calling (e.g. `qwen3`, `mistral-nemo`, `llama3.1`). If responses degrade after enabling it, the model may not support the feature — revert to the default setting.

## Development

Run the test suite:

```bash
pip install -e ".[dev]"
pytest
```
