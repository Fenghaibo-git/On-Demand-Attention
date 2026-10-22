#!/usr/bin/env python3
"""Serial Qwen3.5-2B decode benchmark with one fixed continuation.

Run each mode in a fresh process using the pinned vLLM 0.18 environment.
Quality evaluation is separate. The first output token and warmup request are
excluded from decode tokens per second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODES = {"router": "Qwen35FastRouter2B", "full": "Qwen35FastFull2B",
         "local": "Qwen35FastLocal2B"}
CASES = {"step_1024", "step_2048", "step_3072", "merge_1024_2048_3072"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_report(worker):
    return worker.model_runner.model.runtime_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path, help="Prepared Qwen3.5-2B hf_runtime")
    parser.add_argument("--mode", required=True, choices=(*MODES, "native"))
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.decode_steps < 2 or args.repeats < 1:
        parser.error("Require at least two decode steps and one measured repeat")
    lengths = [int(x) for x in args.lengths.split(",")]
    if not lengths or len(lengths) != len(set(lengths)) or min(lengths) < 16:
        parser.error("Prompt lengths must be distinct and at least 16 tokens")
    runtime = args.runtime.resolve(strict=True)
    if runtime.name != "hf_runtime" or runtime.parent.name not in CASES:
        raise ValueError("Select one prepared v4.2 evaluation case")
    run = runtime.parents[2]
    run_root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if run.parent != run_root:
        raise ValueError("Benchmark output must belong to the canonical run root")
    config = json.loads((runtime / "config.json").read_text())
    marker = json.loads((runtime / "RUNTIME_PREPARED.json").read_text())
    if (marker.get("status") != "passed"
            or marker.get("config_sha256") != sha256(runtime / "config.json")
            or marker.get("runtime_audit_sha256") != sha256(runtime / "runtime_audit.json")):
        raise ValueError("Prepared runtime changed")
    base = Path(config["router_base_model_path"]).resolve(strict=True)
    if base.name != "Qwen3.5-2B" or sha256(base / "config.json") != config["router_base_config_sha256"]:
        raise ValueError("Base model configuration changed")
    admission = runtime.parent / "vllm_head_admission.json"
    if args.mode == "router":
        document = json.loads(admission.read_text())
        if (document.get("status") != "passed"
                or document.get("checkpoint") != config["router_checkpoint"]
                or document.get("base_model_path") != str(base)):
            raise ValueError("Router head admission does not bind this runtime")
        for name, expected in document["bindings"].items():
            if sha256(Path(name)) != expected:
                raise ValueError("Router admission binding changed")
    cache_root = Path(os.environ["ROUTER_DATA_ROOT"]).resolve(strict=True) / "cache"
    if not cache_root.is_dir():
        raise ValueError("Create the canonical dataset cache root before timing")
    catalog = cache_root / "README.md"
    if not catalog.is_file():
        raise ValueError("Read and maintain the dataset cache README before adding assets")
    catalog.read_text()
    cache = cache_root / "router_v42_vllm"
    fresh_cache = not cache.exists()
    cache.mkdir(exist_ok=True)
    cache_readme = cache / "README.md"
    if not cache_readme.exists():
        cache_readme.write_text("# Router v4.2 vLLM cache\n\nGenerated kernel and model-loading cache; safe to regenerate.\n")
    if fresh_cache:
        with catalog.open("a") as handle:
            handle.write("\n- `router_v42_vllm/`: regenerable Router v4.2 native-vLLM caches.\n")
    for name, folder in (("TRITON_CACHE_DIR", "triton"),
                         ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                         ("VLLM_CACHE_ROOT", "vllm"), ("HF_HOME", "huggingface")):
        target = cache / folder
        target.mkdir(exist_ok=True)
        os.environ[name] = str(target)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", Q35_DUAL_TP_SIZE="1",
                      Q35_DUAL_POLICY={"router": "router", "full": "always_full",
                                       "local": "always_local", "native": "router"}[args.mode],
                      Q35_DUAL_FULL_INPUT="embedding", Q35_DUAL_THRESHOLD="0",
                      Q35_STREAM_GRAPH="1", Q35_STREAM_ORACLE="0",
                      ROUTER_MODEL_ROOT=str(base.parent), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if args.mode == "router":
        os.environ["Q35_DUAL_ROUTER_ADMISSION"] = str(admission)
    else:
        os.environ.pop("Q35_DUAL_ROUTER_ADMISSION", None)
    output = runtime.parent / "acceleration" / args.mode
    if output.exists():
        raise FileExistsError("Preserve the previous timing attempt")
    output.mkdir(parents=True)
    from tokenizers import Tokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from qwen3_5_router_runtime_experiments.decode_speed_timing import timed_request
    from qwen3_5_router_runtime_experiments.qwen35_state_contract import native_qwen35_state_options

    maximum = max(lengths) + args.decode_steps + 16
    if maximum > int(config["max_position_embeddings"]):
        raise ValueError("Requested timing length exceeds the native model context")
    options = native_qwen35_state_options()
    if args.mode != "native":
        import qwen35_2b_absolute_vllm  # registers the local architecture
        options["hf_overrides"] = {"architectures": [MODES[args.mode]]}
    started = time.perf_counter()
    llm = LLM(model=str(base), skip_tokenizer_init=True, language_model_only=True,
              tensor_parallel_size=1, pipeline_parallel_size=1, dtype="bfloat16",
              seed=2048, max_model_len=maximum, max_num_seqs=1,
              max_num_batched_tokens=maximum, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              async_scheduling=False, gpu_memory_utilization=0.90,
              attention_config={"backend": "FLASH_ATTN"},
              compilation_config={"mode": 0}, disable_custom_all_reduce=True,
              distributed_executor_backend="uni", **options)
    loaded_seconds = time.perf_counter() - started
    if args.mode != "native":
        from qwen35_2b_absolute_vllm.serial_position_source import install_cpu_position_source
        llm.collective_rpc(install_cpu_position_source)
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    unit = tokenizer.encode("The archive stores numbered records. Each record has a key and a value.\n",
                            add_special_tokens=False).ids
    if not unit:
        raise ValueError("Synthetic timing sentence produced no tokens")
    parameters = SamplingParams(temperature=0, max_tokens=args.decode_steps + 1,
                                min_tokens=args.decode_steps + 1, ignore_eos=True,
                                allowed_token_ids=[unit[0]], detokenize=False,
                                output_kind=RequestOutputKind.DELTA)
    rows = []
    for length in lengths:
        prompt = (unit * ((length + len(unit) - 1) // len(unit)))[:length]
        for repeat in range(args.repeats + 1):
            observation = timed_request(llm.llm_engine, f"{length}-{repeat}", prompt, parameters)
            if observation["decode_tokens"] != args.decode_steps:
                raise ValueError("Fixed continuation length changed")
            observation.update(prompt_tokens=length, repeat=repeat, warmup=(repeat == 0))
            rows.append(observation)
            with (output / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(observation, sort_keys=True) + "\n")
    grouped = {}
    for length in lengths:
        measured = [row for row in rows if row["prompt_tokens"] == length and not row["warmup"]]
        seconds = sum(row["decode_seconds"] for row in measured)
        grouped[str(length)] = {
            "requests": len(measured), "decode_tokens": sum(row["decode_tokens"] for row in measured),
            "decode_seconds": seconds,
            "pooled_decode_tps": sum(row["decode_tokens"] for row in measured) / seconds,
            "median_request_decode_tps": statistics.median(row["decode_tps"] for row in measured),
        }
    telemetry = llm.collective_rpc(runtime_report) if args.mode != "native" else None
    report = {
        "format": "router-v42-serial-decode-v1", "status": "observation_complete",
        "mode": args.mode, "method_case": runtime.parent.name,
        "runtime_config_sha256": sha256(runtime / "config.json"),
        "base_config_sha256": sha256(base / "config.json"),
        "head_admission_sha256": sha256(admission) if args.mode == "router" else None,
        "vllm_version": __import__("vllm").__version__, "load_seconds": loaded_seconds,
        "prompt_lengths": lengths, "decode_steps": args.decode_steps,
        "measured_repeats_per_length": args.repeats, "warmups_per_length": 1,
        "fixed_continuation_token_id": unit[0], "first_token_excluded": True,
        "per_length": grouped, "telemetry": telemetry,
        "requests_sha256": sha256(output / "requests.jsonl"),
        "source_sha256": {
            path: sha256(ROOT / path) for path in (
                "scripts/benchmark_vllm.py",
                "src/qwen35_2b_absolute_vllm/runtime.py",
                "src/qwen35_2b_absolute_vllm/serial_position_source.py",
                "src/qwen3_5_router_runtime_experiments/vllm_runtime.py",
                "src/qwen3_5_router_runtime_experiments/portable_head.py",
                "src/qwen3_5_router_runtime_experiments/decode_speed_timing.py",
                "src/qwen3_5_router_runtime_experiments/qwen35_state_contract.py",
            )
        },
        "quality_qualified": False,
    }
    temporary = output / ".result.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "result.json")
    print(json.dumps({"status": "observation_complete", "mode": args.mode,
                      "per_length": grouped}, sort_keys=True))


if __name__ == "__main__":
    main()
