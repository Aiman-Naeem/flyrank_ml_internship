from __future__ import annotations

import sys
from pathlib import Path

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

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print("Missing dependency: huggingface_hub. Install it first.", file=sys.stderr)
        return 3

    api = HfApi()
    dataset_id = "FlyRank/internship-warehouse"

    try:
        info = api.dataset_info(dataset_id, token=token)
    except Exception as exc:
        print(f"Failed to fetch dataset info for {dataset_id}:", exc, file=sys.stderr)
        return 4

    print(f"Dataset found: {dataset_id}")
    # Print some safe details
    try:
        print("- Description:", getattr(info, "description", None) or info.get("description") if isinstance(info, dict) else None)
    except Exception:
        pass

    try:
        files = api.list_repo_files(dataset_id, token=token, repo_type="dataset")
        print(f"- Files ({len(files)}):")
        for path in files[:50]:
            print("  -", path)
    except Exception as exc:
        print("- Could not list files:", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
