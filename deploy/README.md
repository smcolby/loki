# Reference deployment

The configuration of a working loki stack on one RX 7900 XTX (24 GB) with 32 GB of RAM: `config.yaml`, the per-model files that live in `~/.llms`, and the Open WebUI settings. Paths use `/home/you`; replace it with your home directory.

## Contents

| Path | Copy to |
| --- | --- |
| `config.yaml` | `config.yaml` at the repository root |
| `models/<model>/preset.ini` | `~/.llms/<model>/preset.ini` |
| `models/muse-glimmer-30b-q4/chat-template.jinja` | `~/.llms/muse-glimmer-30b-q4/` |
| `models/strata/qwen.json`, `swift.json` | `~/.llms/strata/` |
| `open-webui/config.json` | Open WebUI, **Admin Panel > Settings > Database > Import Config** |
| `open-webui/models.json` | Open WebUI, **Workspace > Models > Import** |

`open-webui/config.json` holds only the image generation, image editing, and context compaction settings, so importing it leaves every other setting alone. `open-webui/models.json` holds each model's parameters (reasoning effort, compaction threshold, native function calling), its Kiwix tool, and the Image Studio models' vision setting.

## Model files

Download each file into the directory shown, under `~/.llms`.

| Directory | File | Source (Hugging Face) |
| --- | --- | --- |
| `gemma4-31b-iq4xs` | `gemma-4-31B-it-IQ4_XS.gguf` | `unsloth/gemma-4-31B-it-GGUF` |
| `gemma4-31b-iq4xs` | `mtp-gemma-4-31B-it-Q8_0.gguf` | `unsloth/gemma-4-31B-it-GGUF`, `MTP/` |
| `muse-glimmer-30b-q4` | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | `unsloth/Muse-Glimmer-30B-GGUF` |
| `muse-glimmer-30b-q4` | `dflash-kquant.gguf` | `unsloth/Muse-Glimmer-30B-GGUF` |
| `qwen3.8-27b-iq4xs` | `Qwen3.8-27B-UD-IQ4_XS.gguf` | `unsloth/Qwen3.8-27B-GGUF` |
| `qwen3.8-27b-iq4xs` | `mmproj-Qwen3.8-27B-Q8_0.gguf` | `unsloth/Qwen3.8-27B-GGUF` publishes `mmproj-F16.gguf`; download it under this name or quantize it to Q8_0 |
| `swift-qwen3.8-27b-iq4xs` | `Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf` | `ukisai/Swift-1.5-Qwen3.8-27B-GGUF` |
| `qwen-image-2.1` | `qwen-image-2.1-Q8_0.gguf` | `unsloth/Qwen-Image-2.1-GGUF` |
| `qwen-image-2.1` | `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` | `unsloth/Qwen3-VL-8B-Instruct-GGUF` |
| `qwen-image-2.1` | `Qwen3-VL-8B-Instruct-mmproj-F16.gguf` | `unsloth/Qwen3-VL-8B-Instruct-GGUF`, saved from `mmproj-F16.gguf` |
| `qwen-image-2.1` | `qwen_image_2.1_vae_bf16.safetensors` | `unsloth/Qwen-Image-2.1-FP8`, `vae/` |
| `strata/models/IQ2_XS` | both `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000N-of-00002.gguf` shards | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ2_XS/` |
| `strata/models/swift-IQ2_XS` | both `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000N-of-00002.gguf` shards | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` |

The Strata engines also read files that Strata's tools build from those shards. Run the tools from the `loki-strata` image after `loki setup` builds it, with `~/.llms/strata` mounted writable:

- `packs/iq2_xs` and `packs/swift-iq2_xs`: `/opt/strata/tools/iq_pack.py --gguf <first shard> --out <pack directory> --experts-bin`, once per model.
- `mtp/rt`: the multi-token prediction drafter, built with `/opt/strata/tools/mtp_fetch.py` and `mtp_pack.py` as described in Strata's `docs/DETAILS.md`.
- `expert-profile.bin`: copy `data/expert-profile.bin` from the Strata image.

## Restore

1. Clone loki and follow the README's installation section.
2. Download the model files above and copy the files from `models/` into `~/.llms`.
3. Build the Strata packs and drafter.
4. Copy `config.yaml` to the repository root, replace `/home/you`, and run `loki setup`.
5. In Open WebUI, create the admin account, then:
   - add `tools/kiwix_tool.py` under **Workspace > Tools** with the ID `local_kiwix_search`;
   - add `tools/image_studio.py` under **Admin Panel > Functions** with the ID `image_studio` and enable it;
   - import `open-webui/config.json` and `open-webui/models.json`;
   - enter a web search provider and its key under **Admin Panel > Settings > Web Search**, if you use one;
   - paste your system prompt under **Settings > General > System Prompt**.
