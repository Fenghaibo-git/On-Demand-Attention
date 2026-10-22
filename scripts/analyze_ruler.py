#!/usr/bin/env python3
"""Audit all 13 RULER16K scores and decode-only Router decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


TASKS = {
    "ruler_cwe_16k", "ruler_fwe_16k", "ruler_vt_16k",
    "ruler_niah_single_1_16k", "ruler_niah_single_2_16k", "ruler_niah_single_3_16k",
    "ruler_niah_multikey_1_16k", "ruler_niah_multikey_2_16k", "ruler_niah_multikey_3_16k",
    "ruler_niah_multivalue_16k", "ruler_niah_multiquery_16k",
    "ruler_qa_squad_16k", "ruler_qa_hotpotqa_16k",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    case = args.case_dir.resolve(strict=True)
    root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if case.parent.parent.parent != root:
        raise ValueError("Analysis target is outside the canonical run root")
    marker_path = case / "markers/score.json"
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "passed" or marker.get("phase") != "score":
        raise ValueError("Scoring has not passed")
    runtime = case / "hf_runtime"
    if (marker["sources"]["runtime_config_sha256"] != sha256(runtime / "config.json")
            or marker["sources"]["runtime_audit_sha256"] != sha256(runtime / "runtime_audit.json")):
        raise ValueError("Scoring and runtime sources differ")
    config = json.loads((runtime / "config.json").read_text())
    audit = json.loads((runtime / "runtime_audit.json").read_text())
    results = []
    for artifact in marker["task_artifacts"]:
        path = case / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError("Scored output changed after its success marker")
        results.append(path)
    if len(results) != len(TASKS) or {path.stem for path in results} != TASKS:
        raise ValueError("Expected exactly one result for each of 13 RULER16K tasks")
    scores = {}
    for path in results:
        payload = json.loads(path.read_text())
        score = payload.get("score")
        details = payload.get("details")
        if (not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 100 or not isinstance(details, dict)
                or set(details) - {"type"} != {str(index) for index in range(100)}):
            raise ValueError(f"Incomplete 100-example result: {path.name}")
        scores[path.stem] = float(score)
    telemetry_paths = sorted((case / "telemetry").glob("router-*.json"))
    if not telemetry_paths:
        raise ValueError("Selected-cache Router telemetry is missing")
    routed = full = local = nonfinite = 0
    for path in telemetry_paths:
        payload = json.loads(path.read_text())
        if (payload.get("format") != "qwen3.5-selected-hybrid-cache-router-aggregate-v1"
                or payload.get("runtime_history_mode") != "selected_hybrid_cache"
                or payload.get("router_mode") != "routed"
                or payload.get("router_action_threshold") != 0
                or payload.get("router_action_comparison") != ">"
                or payload.get("selected_hybrid_cache_invariants_passed") is not True
                or payload.get("router_score_space") != "signed_log1p_robust_decision_margin"
                or payload.get("checkpoint", {}).get("file_sha256") != audit["export_sha256"]
                or payload.get("checkpoint", {}).get("objective_sha256") != config["router_expected_objective_sha256"]):
            raise ValueError(f"Router telemetry contract changed: {path.name}")
        count = int(payload["routed_row_tokens"])
        f = int(payload["full_row_tokens"])
        l = int(payload["local_row_tokens"])
        if count < 0 or f < 0 or l < 0 or f + l != count:
            raise ValueError("Invalid selected-cache route counts")
        routed += count
        full += f
        local += l
        nonfinite += int(payload["nonfinite_score_row_tokens"])
    if routed <= 0:
        raise ValueError("No routed decode token was observed")
    output = case / "ruler16k_analysis.json"
    success = case / "markers/analysis.json"
    if output.exists() or success.exists():
        raise FileExistsError("Preserve the previous analysis")
    report = {
        "format": "router-v42-ruler16k-analysis-v1", "status": "passed",
        "case": case.name, "tasks": 13, "examples_per_task": 100,
        "score": sum(scores.values()) / 13, "task_scores": dict(sorted(scores.items())),
        "routed_decode_tokens": routed, "full_decode_tokens": full,
        "local_decode_tokens": local, "nonfinite_score_tokens": nonfinite,
        "decode_full_rate": full / routed,
        "score_marker_sha256": sha256(marker_path),
        "result_sha256": {path.stem: sha256(path) for path in sorted(results)},
        "telemetry_sha256": {path.name: sha256(path) for path in telemetry_paths},
    }
    temporary = output.with_name(".ruler16k_analysis.json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    success_tmp = success.with_name(".analysis.json.tmp")
    success_tmp.write_text(json.dumps({"status": "passed", "analysis_sha256": sha256(output)},
                                      indent=2, sort_keys=True) + "\n")
    os.replace(success_tmp, success)
    print(json.dumps({"status": "passed", "score": report["score"],
                      "decode_full_rate": report["decode_full_rate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
