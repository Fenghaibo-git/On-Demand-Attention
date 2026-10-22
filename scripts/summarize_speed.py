#!/usr/bin/env python3
"""Compare matched measured decode throughput; keep quality separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    case = args.case_dir.resolve(strict=True)
    run_root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if case.parent.parent.parent != run_root or case.name not in {
            "step_1024", "step_2048", "step_3072", "merge_1024_2048_3072"}:
        raise ValueError("Speed comparison must belong to a canonical v4.2 case")
    output = case / "acceleration/summary.json"
    if output.exists():
        raise FileExistsError("Preserve the previous speed summary")
    modes = {}
    for mode in ("router", "full", "local", "native"):
        path = case / "acceleration" / mode / "result.json"
        if not path.is_file():
            if mode in ("router", "full"):
                raise ValueError("Both Router and Full timing are required")
            continue
        row = json.loads(path.read_text())
        if (row.get("status") != "observation_complete" or row.get("mode") != mode
                or row.get("requests_sha256") != sha256(path.with_name("requests.jsonl"))):
            raise ValueError(f"Incomplete or changed timing for {mode}")
        modes[mode] = row
    reference = modes["router"]
    comparables = ("method_case", "runtime_config_sha256", "base_config_sha256",
                   "vllm_version", "prompt_lengths", "decode_steps",
                   "measured_repeats_per_length", "warmups_per_length",
                   "fixed_continuation_token_id", "first_token_excluded")
    for mode, row in modes.items():
        if any(row.get(key) != reference.get(key) for key in comparables):
            raise ValueError(f"Timing protocol differs for {mode}")
        if row.get("source_sha256") != reference.get("source_sha256"):
            raise ValueError(f"Runtime source revision differs for {mode}")
    comparison = {}
    for length in reference["prompt_lengths"]:
        key = str(length)
        rates = {mode: row["per_length"][key]["pooled_decode_tps"] for mode, row in modes.items()}
        comparison[key] = {
            "pooled_decode_tps": rates,
            "router_over_full": rates["router"] / rates["full"],
            "router_over_native": rates["router"] / rates["native"] if "native" in rates else None,
        }
    telemetry = reference["telemetry"]
    if not isinstance(telemetry, list) or len(telemetry) != 1:
        raise ValueError("Expected one serial Router worker telemetry record")
    counts = telemetry[0]["counts"]
    routed = counts["routed_tokens"]
    if not routed or not 0 <= counts["logical_full_tokens"] <= routed:
        raise ValueError("Invalid Router decision counts")
    payload = {
        "format": "router-v42-serial-speed-comparison-v1",
        "status": "observation_complete", "method_case": case.name,
        "vllm_version": reference["vllm_version"],
        "prompt_lengths": reference["prompt_lengths"],
        "decode_steps": reference["decode_steps"],
        "measured_repeats_per_length": reference["measured_repeats_per_length"],
        "first_token_excluded": True,
        "full_rate": counts["logical_full_tokens"] / routed,
        "full_rate_denominator": "routed decode tokens; prefill excluded",
        "per_length": comparison,
        "result_sha256": {mode: sha256(case / "acceleration" / mode / "result.json")
                          for mode in modes},
        "quality_qualified": False,
    }
    temporary = output.with_name(".summary.json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": "observation_complete", "per_length": comparison}, sort_keys=True))


if __name__ == "__main__":
    main()
