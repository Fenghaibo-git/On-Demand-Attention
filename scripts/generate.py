#!/usr/bin/env python3
"""Generate with a v4.2 Router, one prompt at a time (batch size 1)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from router_checkpoint import load_hf, sha256


def vllm_results(args, prompts):
    if args.admission is None:
        raise ValueError("vLLM requires --admission from scripts/prepare_inference.py")
    doc = json.loads(args.admission.read_text())
    if doc.get("status") != "passed":
        raise ValueError("Head admission is not passed")
    for filename, digest in doc["bindings"].items():
        if sha256(filename) != digest:
            raise ValueError("A head-admission input changed")
    base = Path(doc["base_model_path"])
    if args.model and args.model.resolve() != base.resolve():
        raise ValueError("--model differs from the admitted model")
    os.environ.update(ROUTER_MODEL_ROOT=str(base.parent), VLLM_ENABLE_V1_MULTIPROCESSING="0",
        Q35_DUAL_ROUTER_ADMISSION=str(args.admission.resolve()), Q35_DUAL_TP_SIZE="1",
        Q35_DUAL_POLICY="router", Q35_DUAL_FULL_INPUT="embedding", Q35_DUAL_THRESHOLD="0",
        Q35_STREAM_GRAPH="1", Q35_STREAM_ORACLE="0", Q3_STREAM_GRAPH="1", Q3_STREAM_ORACLE="0")
    from vllm import LLM, SamplingParams
    options = {}
    if doc["family"] == "qwen3" and base.name == "Qwen3-8B":
        from qwen3_absolute_vllm import register
        register()
        architecture = "Qwen3FastRouter"
        from qwen3_5_router_runtime_experiments.serial_position_source import install_cpu_position_source
    elif doc["family"] == "qwen35" and base.name in {"Qwen3.5-2B", "Qwen3.5-27B"}:
        from qwen3_5_router_runtime_experiments.qwen35_state_contract import native_qwen35_state_options
        options.update(native_qwen35_state_options())
        if base.name == "Qwen3.5-2B":
            import qwen35_2b_absolute_vllm
            from qwen35_2b_absolute_vllm.serial_position_source import install_cpu_position_source
            architecture = "Qwen35FastRouter2B"
        else:
            import qwen35_absolute_tp_vllm
            from qwen35_absolute_tp_vllm.serial_position_source import install_cpu_position_source
            architecture = "Qwen35FastRouter27B"
    else:
        raise ValueError("No native vLLM Router is included for this model")
    options["hf_overrides"] = {"architectures": [architecture]}
    llm = LLM(model=str(base), language_model_only=True, tensor_parallel_size=1,
        pipeline_parallel_size=1, dtype="bfloat16", max_model_len=args.max_model_len,
        max_num_seqs=1, max_num_batched_tokens=args.max_model_len,
        enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False,
        async_scheduling=False, attention_config={"backend": "FLASH_ATTN"},
        compilation_config={"mode": 0}, disable_custom_all_reduce=True,
        distributed_executor_backend="uni", **options)
    llm.collective_rpc(install_cpu_position_source)
    tokenizer = llm.get_tokenizer()
    parameters = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    for prompt in prompts:
        text = prompt if args.raw else tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer.encode(text, add_special_tokens=False)) + args.max_new_tokens > args.max_model_len:
            raise ValueError("Prompt and generation exceed --max-model-len")
        result = llm.generate([text], parameters, use_tqdm=False)
        yield {"prediction": result[0].outputs[0].text,
               "input_tokens": len(result[0].prompt_token_ids),
               "output_tokens": len(result[0].outputs[0].token_ids)}


def add_runtime_args(parser):
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--router", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw", action="store_true", help="Skip the model's chat template")


def validate_runtime_args(parser, args):
    if args.max_new_tokens < 1 or args.max_model_len < 1:
        parser.error("Token limits must be positive")
    if args.backend == "vllm":
        if args.router:
            parser.error("Use --admission for vLLM; --router is the HF argument")
        return
    if not args.model or not args.router:
        parser.error("Transformers inference needs --model and --router")
    # Reject experimental environment overrides instead of silently changing v4.2.
    for key in os.environ:
        if key.startswith(("QWEN3_SELECTED_CACHE_", "QWEN35_SELECTED_CACHE_")):
            parser.error(f"Unset {key}; this entry fixes the v4.2 runtime contract")


def hf_results(args, prompts):
    model, tokenizer, report = load_hf(args.model, args.router, args.device)
    import torch
    for prompt in prompts:
        text = prompt if args.raw else tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(args.device)
        inputs.pop("token_type_ids", None)
        if inputs.input_ids.shape[-1] + args.max_new_tokens > args.max_model_len:
            raise ValueError("Prompt and generation exceed --max-model-len")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, num_beams=1,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id))
        yield {"prediction": tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True),
               "input_tokens": inputs.input_ids.shape[-1],
               "output_tokens": output.shape[-1] - inputs.input_ids.shape[-1]}


def generate_results(args, prompts):
    yield from (vllm_results(args, prompts) if args.backend == "vllm" else hf_results(args, prompts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--prompt", action="append", required=True)
    args = parser.parse_args()
    validate_runtime_args(parser, args)
    for result in generate_results(args, args.prompt):
        print(result["prediction"])


if __name__ == "__main__":
    main()
