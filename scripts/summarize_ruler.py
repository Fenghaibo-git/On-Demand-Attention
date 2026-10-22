#!/usr/bin/env python3
"""Create one hash-bound v4.2 RULER16K summary from four audited cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CASES = ("step_1024", "step_2048", "step_3072", "merge_1024_2048_3072")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if run.parent != root:
        raise ValueError("Summary target is outside the canonical run root")
    training_marker = run / "training_passed.json"
    if json.loads(training_marker.read_text()).get("audit_sha256") != sha256(run / "training_audit.json"):
        raise ValueError("Training audit changed")
    output = run / "evaluations/ruler16k_summary.json"
    success = run / "ruler16k_passed.json"
    if output.exists() or success.exists():
        raise FileExistsError("Preserve the previous RULER summary")
    records = {}
    for name in CASES:
        case = run / "evaluations" / name
        path = case / "ruler16k_analysis.json"
        marker = json.loads((case / "markers/analysis.json").read_text())
        report = json.loads(path.read_text())
        if (marker.get("status") != "passed" or marker.get("analysis_sha256") != sha256(path)
                or report.get("status") != "passed" or report.get("case") != name
                or report.get("tasks") != 13 or report.get("examples_per_task") != 100):
            raise ValueError(f"RULER case is incomplete or changed: {name}")
        records[name] = {"score": report["score"],
                         "decode_full_rate": report["decode_full_rate"],
                         "analysis_sha256": sha256(path)}
    payload = {"format": "router-v42-ruler16k-summary-v1", "status": "passed",
               "method": "v4.2", "protocol": "RULER16K 13x100 threshold0",
               "training_audit_sha256": sha256(run / "training_audit.json"),
               "cases": records}
    temporary = output.with_name(".ruler16k_summary.json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    success_tmp = success.with_name(".ruler16k_passed.json.tmp")
    success_tmp.write_text(json.dumps({"status": "passed", "summary_sha256": sha256(output)},
                                      indent=2, sort_keys=True) + "\n")
    os.replace(success_tmp, success)
    print(json.dumps({"status": "passed", "cases": records}, sort_keys=True))


if __name__ == "__main__":
    main()
