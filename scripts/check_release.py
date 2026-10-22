#!/usr/bin/env python3
"""Check publishable files for local paths, unlisted binaries, and credentials.

The scan reports filenames, line numbers and rule names only. It never echoes a
matched line, since matched text could contain a secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".example", ".gitignore"}
BLOCKED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "wandb", "runs", "outputs", "checkpoints", ".venv"}
BLOCKED_SUFFIXES = {".pt", ".pth", ".safetensors", ".parquet", ".arrow", ".jsonl", ".log", ".tar", ".zip", ".7z", ".pem", ".key", ".p12", ".pfx"}
# Published figure/table images, visually reviewed; provenance is in assets/README.md.
REVIEWED_ASSETS = {
    "assets/oda-main-overview.png": "068c44f2127776c0e31a071ecb1444123b61dfc5d6d18d75c1e74e3d7683a8a4",
    "assets/oda-ruler-table1.png": "c5268466b828f38e5e61309a8fc669a2e7d75d5ff69ea34cdf0cef5b858fe031",
}
RULES = {
    "private_path": re.compile(r"(?:/home/(?!user\b)|/Users/|/[A-Za-z]{1,16}Data[0-9]+/|[A-Za-z]:\\Users\\|/root/\.ssh)", re.I),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "credential": re.compile(r"(?:sk-proj-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"),
    "assigned_secret": re.compile(r"(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[=:]\s*['\"][^'\"]{8,}['\"]", re.I),
    "internal_address": re.compile(r"(?:https?://|ssh://)[a-z0-9.-]*(?:intranet|internal|localhost|127\.0\.0\.1|10\.[0-9]+\.[0-9]+\.[0-9]+)[^\s]*", re.I),
    "server_label": re.compile(r"\bgpu[0-9]+\b", re.I),
    "institutional_identifier": re.compile(r"\b[0-9]{6,}-[a-z0-9-]*(?:university|institution|organization)[a-z0-9-]*\b", re.I),
}
def scan(root: Path = ROOT) -> dict:
    findings = []
    manifest = {}
    if (root / ".git").exists():
        result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                cwd=root, check=True, capture_output=True, text=True)
        paths = [root / name for name in set(result.stdout.split("\0")) if name]
    else:
        paths = root.rglob("*")
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"file": relative, "rule": "symlink"})
            continue
        if path.is_dir():
            continue
        if any(part in BLOCKED_PARTS for part in path.relative_to(root).parts):
            findings.append({"file": relative, "rule": "blocked_directory"})
            continue
        if path.suffix in BLOCKED_SUFFIXES:
            findings.append({"file": relative, "rule": "binary_or_asset"})
            continue
        if relative in REVIEWED_ASSETS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != REVIEWED_ASSETS[relative]:
                findings.append({"file": relative, "rule": "asset_hash_mismatch"})
            else:
                manifest[relative] = digest
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", "NOTICE", ".gitignore"}:
            findings.append({"file": relative, "rule": "unlisted_type"})
            continue
        data = path.read_bytes()
        if b"\0" in data:
            findings.append({"file": relative, "rule": "binary_content"})
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"file": relative, "rule": "non_utf8"})
            continue
        manifest[relative] = hashlib.sha256(data).hexdigest()
        for number, line in enumerate(content.splitlines(), 1):
            # A scanner rule may contain its own example pattern text.
            if relative == "scripts/check_release.py" and line.lstrip().startswith(tuple(f'"{name}"' for name in RULES)):
                continue
            for rule, expression in RULES.items():
                if rule == "email" and (relative == "LICENSE" or relative.startswith("licenses/")):
                    continue  # Preserve upstream legal notices verbatim.
                if expression.search(line):
                    findings.append({"file": relative, "line": number, "rule": rule})
    return {"status": "passed" if not findings else "failed", "files": len(manifest), "findings": findings, "sha256": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print complete hashes and finding locations")
    args = parser.parse_args()
    result = scan()
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        print(json.dumps({key: result[key] for key in ("status", "files", "findings")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
