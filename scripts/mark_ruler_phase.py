#!/usr/bin/env python3
"""Bind each RULER inference/scoring stage to its actual OpenCompass outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "score"))
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    case = args.case_dir.resolve(strict=True)
    kind = "predictions" if args.phase == "infer" else "results"
    roots = [p for p in (case / "opencompass").rglob(kind) if p.is_dir()]
    if len(roots) != 1:
        raise ValueError("Expected exactly one OpenCompass timestamp directory for this case")
    root = roots[0]
    files = sorted(path for path in root.rglob("*.json") if path.is_file())
    if len(files) < 13:
        raise ValueError("RULER stage produced fewer than the expected 13 task artifacts")
    log = case / f"logs/{args.phase}.log"
    if not log.is_file() or "OpenCompass - ERROR -" in log.read_text(errors="replace"):
        raise ValueError("OpenCompass reported a stage error")
    runtime = case / "hf_runtime"
    sources = {
        "runtime_config_sha256": sha256(runtime / "config.json"),
        "runtime_audit_sha256": sha256(runtime / "runtime_audit.json"),
        "evaluation_config_sha256": sha256(Path(os.environ.get("ROUTER_RULER_CONFIG", Path(__file__).resolve().parents[1] / "configs/eval_ruler16k.py"))),
        "wrapper_sha256": sha256(Path(__file__).resolve().parents[1] / "src/oda_opencompass.py"),
    }
    payload = {"status": "passed", "phase": args.phase, "sources": sources,
               "task_artifacts": [{"path": path.relative_to(case).as_posix(), "sha256": sha256(path)} for path in files]}
    if args.phase == "score":
        infer = json.loads((case / "markers/infer.json").read_text())
        if infer.get("status") != "passed" or infer.get("sources") != sources:
            raise ValueError("Scoring does not match the completed inference stage")
        for artifact in infer["task_artifacts"]:
            if sha256(case / artifact["path"]) != artifact["sha256"]:
                raise ValueError("Inference outputs changed before scoring")
        payload["inference_marker_sha256"] = sha256(case / "markers/infer.json")
    marker = case / f"markers/{args.phase}.json"
    if marker.exists():
        raise FileExistsError("Phase marker already exists")
    temporary = marker.with_name("." + marker.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    print(json.dumps({"status": "passed", "phase": args.phase, "task_artifacts": len(files)}))


if __name__ == "__main__":
    main()
