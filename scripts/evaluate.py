#!/usr/bin/env python3
"""Evaluate prompt/answers JSONL with greedy decoding at batch size 1.

The metric is case-insensitive, whitespace-normalized exact match against any
reference. Use the dedicated RULER pipeline for the RULER benchmark protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from generate import add_runtime_args, generate_results, validate_runtime_args
from router_checkpoint import ROOT, sha256


def read_examples(path):
    examples = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Line {line_number} needs a nonempty prompt")
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers or not all(isinstance(x, str) for x in answers):
                raise ValueError(f"Line {line_number} needs a nonempty list of answer strings")
            examples.append(row)
    if not examples:
        raise ValueError("Empty evaluation input")
    return examples


def exact_match(prediction, answers):
    normalize = lambda text: " ".join(text.casefold().split())
    return int(any(normalize(prediction) == normalize(answer) for answer in answers))


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "infer", "score"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Keep evaluation outputs outside the source checkout")
    prediction_path = output / "predictions.jsonl"
    marker = output / "inference_passed.json"
    if args.stage != "score":
        validate_runtime_args(parser, args)
        if args.input is None:
            parser.error("Inference requires --input")
        examples = read_examples(args.input)
        if output.exists():
            parser.error("Output already exists; preserve it and choose a fresh directory, or use --stage score")
        output.mkdir(parents=True)
        (output / "README.md").write_text("# On Demand Attention JSONL evaluation\n\n"
            "Greedy batch-1 predictions and normalized exact-match scores.\n")
        contract = {"backend": args.backend, "batch_size": 1, "raw_prompt": args.raw,
                    "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
                    "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
                    "model": str(args.model.resolve()) if args.model else None,
                    "router_sha256": sha256(args.router) if args.router else None,
                    "admission_sha256": sha256(args.admission) if args.admission else None,
                    "entry_sha256": sha256(__file__), "generation_sha256": sha256(ROOT / "scripts/generate.py")}
        atomic_json(output / "protocol.json", contract)
        count = 0
        with prediction_path.open("x") as stream:
            for row, result in zip(examples, generate_results(args, (x["prompt"] for x in examples))):
                stream.write(json.dumps({"id": row.get("id", count), "prompt": row["prompt"],
                    "answers": row["answers"], **result}, ensure_ascii=False) + "\n")
                stream.flush()
                count += 1
        if count != len(examples):
            raise RuntimeError("Incomplete inference; no success marker written")
        atomic_json(marker, {"status": "passed", "count": count,
            "prediction_sha256": sha256(prediction_path), "protocol_sha256": sha256(output / "protocol.json")})
    if args.stage != "infer":
        evidence = json.loads(marker.read_text())
        if (evidence.get("status") != "passed" or evidence["prediction_sha256"] != sha256(prediction_path)
                or evidence["protocol_sha256"] != sha256(output / "protocol.json")):
            raise ValueError("Inference evidence changed")
        rows = read_examples(prediction_path)
        if len(rows) != evidence["count"]:
            raise ValueError("Prediction count mismatch")
        scores = [exact_match(row["prediction"], row["answers"]) for row in rows]
        report = {"metric": "normalized_exact_match", "score": 100 * sum(scores) / len(scores),
                  "correct": sum(scores), "count": len(scores),
                  "inference_marker_sha256": sha256(marker)}
        atomic_json(output / "scores.json", report)
        atomic_json(output / "scoring_passed.json", {"status": "passed", "score_sha256": sha256(output / "scores.json")})
        print(json.dumps(report))


if __name__ == "__main__":
    main()
