from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except Exception as exc:  # pragma: no cover - runtime installer will handle
    print("Missing dependency: huggingface_hub. Install it first.", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def read_hf_token(env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if not line or line.strip().startswith("#"):
            continue
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def main() -> int:
    token = read_hf_token(ENV_PATH)
    if not token:
        print("HF_TOKEN not found in .env", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        who = api.whoami(token=token)
    except Exception as exc:
        print("Authentication failed:", exc, file=sys.stderr)
        return 3

    # Print a small, safe summary about the authenticated user
    username = who.get("name") or who.get("id") or who.get("email") or who.get("username")
    print("Authenticated user:", username or who)

    try:
        datasets_gen = api.list_datasets(author=username, token=token, sort="lastModified")
    except Exception as exc:
        print("Failed to list datasets for the user:", exc, file=sys.stderr)
        return 4

    # The API may return a generator — iterate safely and show up to 50 items.
    from itertools import islice

    first_batch = list(islice(datasets_gen, 50))
    if not first_batch:
        print("No datasets found for user.")
        return 0

    print("Datasets (showing up to 50):")
    for item in first_batch:
        ident = getattr(item, "id", None) or getattr(item, "datasetId", None) or str(item)
        print("-", ident)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
