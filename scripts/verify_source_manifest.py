#!/usr/bin/env python3
"""Verify that every copied source file is identified and byte-bound."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    document = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    if document.get("format") != "router-v42-selected-source-manifest-v1":
        raise ValueError("Unknown source manifest format")
    files = document["files"]
    listed = set()
    for row in files:
        name = row["file"]
        path = ROOT / name
        if (name in listed or Path(name).is_absolute() or ".." in Path(name).parts
                or not path.is_file() or path.is_symlink()):
            raise ValueError("Missing, duplicate, or unsafe source path")
        listed.add(name)
        if (not row["origin"].startswith(("VeOmni/", "opencompass/", "modelings/", "eval/"))
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])):
            raise ValueError(f"Invalid source origin/hash: {name}")
        actual = sha256(path)
        if (actual != row.get("release_sha256")
                or row.get("modified") is not (actual != row["source_sha256"])):
            raise ValueError(f"Copied source changed after review: {name}")
    expected = {
        path.relative_to(ROOT).as_posix()
        for tree in (ROOT / "src", ROOT / "configs/datasets/ruler")
        for path in tree.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    }
    if listed != expected:
        raise ValueError(f"Unlisted or missing copied source: {len(expected - listed)} extra, {len(listed - expected)} missing")
    print(json.dumps({"status": "passed", "source_files": len(listed),
                      "portable_modifications": sum(row["modified"] for row in files)}))


if __name__ == "__main__":
    main()
