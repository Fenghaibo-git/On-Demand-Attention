#!/usr/bin/env python3
"""Create a deterministic, privacy-scanned source-only ICLR attachment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
import zipfile
from pathlib import Path

from check_release import scan
from verify_source_manifest import main as verify_source_manifest


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT.parent / "router_v42_iclr_submission.zip")
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve the existing release archive")
    if (output == ROOT or ROOT in output.parents) or not output.parent.is_dir():
        raise ValueError("Archive output must be outside the source package")
    if not (output.name.endswith(".zip") or output.name.endswith(".tar.gz")):
        raise ValueError("Use a .zip or .tar.gz output filename")
    verify_source_manifest()
    review = scan(ROOT)
    if review["status"] != "passed":
        raise ValueError(json.dumps({"status": "failed", "findings": review["findings"]}))
    temporary = output.with_name("." + output.name + ".tmp")
    try:
        if output.name.endswith(".zip"):
            with zipfile.ZipFile(temporary, "w") as archive:
                for name, expected in sorted(review["sha256"].items()):
                    path = ROOT / name
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError("Source changed during packaging")
                    info = zipfile.ZipInfo("router_v42_iclr/" + name,
                                           date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | (0o755 if os.access(path, os.X_OK) else 0o644)) << 16
                    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            with zipfile.ZipFile(temporary) as archive:
                found = {}
                for entry in archive.infolist():
                    if entry.is_dir() or not entry.filename.startswith("router_v42_iclr/"):
                        raise ValueError("Archive contains an unexpected entry")
                    name = entry.filename[len("router_v42_iclr/"):]
                    found[name] = hashlib.sha256(archive.read(entry)).hexdigest()
        else:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w") as archive:
                        for name, expected in sorted(review["sha256"].items()):
                            path = ROOT / name
                            data = path.read_bytes()
                            if hashlib.sha256(data).hexdigest() != expected:
                                raise ValueError("Source changed during packaging")
                            info = tarfile.TarInfo(name="router_v42_iclr/" + name)
                            info.size = len(data)
                            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                            info.mtime = info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            archive.addfile(info, io.BytesIO(data))
            found = {}
            with tarfile.open(temporary, "r:gz") as archive:
                for entry in archive:
                    if not entry.isfile() or not entry.name.startswith("router_v42_iclr/"):
                        raise ValueError("Archive contains an unexpected entry")
                    name = entry.name[len("router_v42_iclr/"):]
                    found[name] = hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
        if found != review["sha256"]:
            raise ValueError("Archive content differs from the reviewed source")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256(output)
    digest_path = output.with_name(output.name + ".sha256")
    digest_path.write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"status": "passed", "archive": str(output),
                      "files": len(review["sha256"]), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
