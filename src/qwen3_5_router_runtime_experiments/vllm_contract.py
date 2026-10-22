"""Import-free fail-closed checks for the pinned native-vLLM experiment."""

import math
import os


def validate_text_config(reference, active):
    """Reject architectural overrides; fixed tolerances for JSON/FP32 scalars."""
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "rms_norm_eps",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "attention_bias",
        "tie_word_embeddings",
        "vocab_size",
    )
    for key in keys:
        if key not in reference:
            continue
        expected, value = reference[key], active.get(key)
        if isinstance(expected, float):
            valid = (
                isinstance(value, (float, int))
                and math.isfinite(value)
                and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-12)
            )
        else:
            valid = type(value) is type(expected) and value == expected
        if not valid:
            raise ValueError(f"Base architecture override is not admitted: {key}")


def validate_vllm_contract(version, config):
    if version != "0.18.0":
        raise ValueError("Native Qwen3.5 dual cache is pinned to vLLM 0.18.0")
    tp_size = int(os.environ.get("Q35_DUAL_TP_SIZE", "2" if os.environ.get("Q35_DUAL_TP2_128K") == "1" else "1"))
    if tp_size not in (1, 2, 4, 8) or (tp_size > 2 and os.environ.get("Q35_DUAL_SPEEDCURVE") != "1"):
        raise ValueError("Tensor parallel size is outside the explicitly gated speed experiment")
    required = {
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": 1,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "speculative": False,
        "quantized": False,
        "dtype": "torch.bfloat16",
        "backend": "FLASH_ATTN",
        "sink_size": 4,
        "window_size": 2048,
    }
    missing_or_changed = {
        k: (config.get(k), v) for k, v in required.items() if config.get(k) != v
    }
    if missing_or_changed:
        raise ValueError(
            f"Unsupported native-vLLM experiment contract: {missing_or_changed}"
        )
    if config.get("kv_cache_dtype") not in {"auto", "bfloat16"}:
        raise ValueError("Only unquantized BF16 KV is admitted")
    if config.get("policy") not in {"router", "always_local", "always_full"}:
        raise ValueError("Unknown route policy")
    if config.get("full_input_mode") not in {"embedding", "local_final_hidden"}:
        raise ValueError("Unknown Full input")
    if not math.isfinite(float(config.get("threshold", float("nan")))):
        raise ValueError("Threshold must be finite")


class SerialRequestClock:
    """Reject chunked/interleaved/resumed requests not covered by this runtime."""

    def __init__(self):
        self.length = 0
        self.failed = False

    def classify(self, positions):
        if not positions:
            raise ValueError("Empty real forward")
        if positions[0] == 0:
            if positions != list(range(len(positions))):
                raise ValueError(
                    "Prefill positions must be contiguous and start at zero"
                )
            return "prefill"
        if self.failed:
            raise RuntimeError(
                "Failed request cannot be resumed without fresh Full prefill"
            )
        if positions != [self.length] or self.length == 0:
            raise ValueError(
                "Require one causal decode token for the current serial request"
            )
        return "decode"

    def commit(self, phase, count):
        if phase == "prefill":
            self.length, self.failed = count, False
        elif phase == "decode" and count == 1:
            self.length += 1
        else:
            raise ValueError("Invalid request-clock commit")
