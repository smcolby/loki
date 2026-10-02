"""Start the Strata server from a user engine config plus the facts this image owns.

The mounted config (``/etc/strata/strata.json``) carries the model choice: engine
``args``, ``tokenizer``, ``model_name``, and optional ``env``. This script fills in
where the engine, its libraries, and its GEMM tuning table live inside the image,
then replaces itself with ``serve/server.py`` listening on all interfaces with the
model unloaded until the first request.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

STRATA_ROOT = Path("/opt/strata")
USER_CONFIG = Path("/etc/strata/strata.json")
MERGED_CONFIG = Path("/tmp/strata.json")  # noqa: S108 (container-private path)
TUNING_TABLE = STRATA_ROOT / "hipblaslt-tuning.txt"
IMAGE_KEYS = ("exe", "cwd", "lib_dirs", "backend", "log")


def merge_config(user: dict[str, Any], root: Path, tuning: Path | None) -> dict[str, Any]:
    """Combine a user engine config with the image's engine paths.

    Parameters
    ----------
    user : dict[str, Any]
        Engine config with at least ``args``, ``tokenizer``, and ``model_name``.
    root : Path
        Directory holding the engine, libraries, and server sources.
    tuning : Path or None
        hipBLASLt tuning table built for this image, if one exists.

    Returns
    -------
    dict[str, Any]
        Config ready for ``serve/server.py --config``.

    Raises
    ------
    ValueError
        If a required key is missing or the user config sets a key the image owns.
    """
    missing = [key for key in ("args", "tokenizer", "model_name") if key not in user]
    if missing:
        raise ValueError(f"strata.json is missing {', '.join(missing)}")
    owned = [key for key in IMAGE_KEYS if key in user]
    if owned:
        raise ValueError(f"strata.json sets {', '.join(owned)}; the image provides these")

    merged = dict(user)
    merged.update(
        exe=str(root / "engine" / "strata"),
        cwd=str(root),
        lib_dirs=[str(root / "lib")],
        backend="hip",
        log="/dev/stderr",
    )

    # Default to the image's tuning table unless the config names its own
    env = dict(user.get("env") or {})
    if tuning is not None:
        env.setdefault("STRATA_HIPBLASLT_TUNING", str(tuning))
    merged["env"] = env
    return merged


def main() -> None:
    """Write the merged config and exec the Strata server."""
    try:
        user = json.loads(USER_CONFIG.read_text())
        merged = merge_config(user, STRATA_ROOT, TUNING_TABLE if TUNING_TABLE.is_file() else None)
    except (OSError, ValueError) as exc:
        sys.exit(f"Cannot start Strata: {exc}")
    MERGED_CONFIG.write_text(json.dumps(merged, indent=1))

    server = STRATA_ROOT / "serve" / "server.py"
    argv = [
        sys.executable,
        str(server),
        "--engine",
        "strata",
        "--config",
        str(MERGED_CONFIG),
        "--host",
        "0.0.0.0",  # noqa: S104 (published only through the compose network)
        "--port",
        "8080",
        "--lazy",
        *sys.argv[1:],
    ]
    os.execv(sys.executable, argv)  # noqa: S606 (fixed interpreter and script)


if __name__ == "__main__":
    main()
