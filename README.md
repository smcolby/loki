# loki
Local Offline Knowledge Index

A CLI tool for managing a local LLM and offline Wikipedia knowledge server orchestrated via Docker.

## Prerequisites

### System dependencies

- [Docker](https://docs.docker.com/get-docker/) with the Compose plugin
- [Ollama](https://ollama.com/) installed and running as a native system service
- [aria2](https://aria2.github.io/) for multi-threaded ZIM file downloads

  ```bash
  # macOS
  brew install aria2

  # Debian/Ubuntu
  sudo apt install aria2
  ```

### Ollama network binding (Linux only)

By default, Ollama on Linux binds to `127.0.0.1`. Docker containers contact the host via `host.docker.internal`, which requires Ollama to listen on all interfaces. Create a systemd override to set `OLLAMA_HOST`:

```bash
sudo systemctl edit ollama
```

Add the following lines, then save and close the editor:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Restart the service:

```bash
sudo systemctl restart ollama
```

## Installation

Install system-wide with `pipx` to keep the virtual environment isolated:

```bash
pipx install -e .
```

## Configuration

Edit `config.yaml` to define the server URL, which ZIM files to download, and which Ollama models to pull:

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

### Local hostname resolution (no router changes required)

Using the `.local` TLD (the default) means the hostname is broadcast via **mDNS** and resolves on your LAN without any router configuration. Most modern operating systems support mDNS natively:

- **macOS** — Bonjour is built in; `.local` names resolve automatically.
- **Windows 10+** — mDNS is supported natively.
- **Linux** — requires `avahi-daemon` (usually pre-installed on desktop distros).
  ```bash
  sudo apt install avahi-daemon       # Debian/Ubuntu
  sudo dnf install avahi              # Fedora
  ```

For `loki.local` to resolve on other devices on your network, the **host machine** needs to announce the name. On Linux, install `avahi-utils` and register the alias:

```bash
sudo apt install avahi-utils
avahi-publish-address -R loki.local $(hostname -I | awk '{print $1}')
```

To make this permanent, create a service file at `/etc/avahi/services/loki.service`:

```xml
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name>loki</name>
  <service>
    <type>_http._tcp</type>
    <port>80</port>
  </service>
</service-group>
```

On macOS, `dns-sd` can register the name:

```bash
dns-sd -P loki _http._tcp local 80 loki.local $(ipconfig getifaddr en0)
```

If you prefer a different TLD (e.g. `loki.home`), simply set `url: loki.home` in `config.yaml` and add a static entry to the `/etc/hosts` file on each client device instead.


## Usage

```
loki setup    Generate the Caddyfile, write port settings, and download missing ZIM files.
loki start    Pull Ollama models and start the Docker Compose stack.
loki stop     Stop the Docker Compose stack.
loki status   Check the health of running services.
loki cleanup  Remove ZIM files and Ollama models no longer listed in config.
```

## Connecting Kiwix to Open WebUI

A ready-made Open WebUI tool definition lives at [`tools/kiwix_tool.py`](tools/kiwix_tool.py). It exposes the Kiwix server to the LLM as a callable tool, communicating over the Docker internal network so it works regardless of your configured host port.

To load it into Open WebUI:

1. Open the Open WebUI interface (e.g. `http://loki.local`).
2. Navigate to **Admin Panel → Tools** and click **+** to create a new tool.
3. Copy the full contents of `tools/kiwix_tool.py` and paste them into the editor.
4. Save the tool and assign it to your model under **Model → Tools**.

## Development

Run the test suite:

```bash
pip install -e ".[dev]"
pytest
```
