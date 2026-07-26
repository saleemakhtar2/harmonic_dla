"""Create an immutable manifest for a planned validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation-manifest.json"))
    args = parser.parse_args()
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "configs": [{"path": str(path), "sha256": sha256(path)} for path in sorted(args.configs)],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
