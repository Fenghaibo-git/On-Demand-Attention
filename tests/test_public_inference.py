"""Exercise the public loader with tiny, locally constructed model weights."""
from copy import deepcopy

import pytest
import torch

from router_checkpoint import load_hf, validate_export
from prepare_run import configure_model
from veomni.utils.tensor_state import tensor_state_sha256


def build_export(tmp_path, family, monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    if family == "qwen3":
        from test_qwen_large_robust_margin import tiny_config
        from veomni.models.transformers.qwen3_large_robust_margin_router.modeling_qwen3_large_robust_margin_router import ModelClass
        variant = "qwen3_8b"
    elif family == "qwen35":
        from test_qwen3_5_robust_decision_margin_router import _tiny_model_config as tiny_config
        from veomni.models.transformers.qwen3_5_three_stream_nll_selected_robust_decision_margin_router.modeling_qwen3_5_three_stream_nll_selected_robust_decision_margin_router import ModelClass
        variant = "qwen35_2b_mb1"
    else:
        pytest.importorskip("transformers.models.gemma4_unified")
        from test_gemma4_unified_router import tiny_config
        from veomni.models.transformers.gemma4_unified_three_stream_router.modeling_gemma4_unified_three_stream_router import ModelClass
        variant = "gemma4_12b"
    if family == "qwen35":
        from test_selected_hybrid_cache_decode import force_cpu_fallbacks
        force_cpu_fallbacks(monkeypatch)
    cfg = tiny_config()
    cfg.architectures = [ModelClass.__name__]
    cfg.router_sink_size, cfg.router_window_size = 4, 2048
    cfg.router_optimizer_step_reduction = "global_eligible_token_mean_v1"
    cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = 0, 1, 2
    trained = ModelClass(cfg).eval()
    native_values = deepcopy(cfg.to_dict())
    native_values.pop("architectures", None)
    native = AutoModelForCausalLM.from_config(
        AutoConfig.for_model(native_values.pop("model_type"), **native_values))
    base = tmp_path / "base"
    if family == "gemma4":
        from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
        composite = Gemma4UnifiedConfig(text_config=native.config.to_dict(), vision_config=None, audio_config=None)
        native = Gemma4UnifiedForConditionalGeneration(composite)
    native.save_pretrained(base)
    vocab = {f"v{i}": i for i in range(cfg.vocab_size)}
    inner = Tokenizer(WordLevel(vocab, unk_token="v3"))
    inner.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, pad_token="v0", bos_token="v1", eos_token="v2", unk_token="v3",
        chat_template="{{ messages[0]['content'] }}")
    tokenizer.save_pretrained(base)
    values, _ = configure_model(trained.config.to_dict(), variant, base, "test-only-base-manifest")
    state = {"router_head." + key: value.detach().float().clone() for key, value in trained.router_head.state_dict().items()}
    payload = {"format_version": 1, "model_config": values, "state_dict": state,
               "tensor_state_sha256": tensor_state_sha256(state)}
    checkpoint = tmp_path / "router.pt"
    torch.save(payload, checkpoint)
    return base, checkpoint, payload


@pytest.mark.parametrize("family", ["qwen3", "qwen35", "gemma4"])
def test_export_load_and_greedy_decode(tmp_path, family, monkeypatch):
    base, checkpoint, payload = build_export(tmp_path, family, monkeypatch)
    model, tokenizer, report = load_hf(base, checkpoint, "cpu")
    assert report["family"] == family
    assert report["training_completion_verified"] is False
    inputs = tokenizer("v4 v5 v6", return_tensors="pt")
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
    assert output.shape == (1, 6)
    for name, value in model.router_head.state_dict().items():
        torch.testing.assert_close(value, payload["state_dict"]["router_head." + name], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["threshold", "tensor", "base"])
def test_export_rejects_changed_contract(tmp_path, monkeypatch, mutation):
    base, checkpoint, payload = build_export(tmp_path, "qwen3", monkeypatch)
    if mutation == "threshold":
        payload["model_config"]["router_history_nll_threshold"] = .001
        torch.save(payload, checkpoint)
    elif mutation == "tensor":
        next(iter(payload["state_dict"].values())).flatten()[0] += 1
        torch.save(payload, checkpoint)
    else:
        path = base / "config.json"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        validate_export(checkpoint, base)


def test_qwen3_rotary_matches_native_at_long_positions():
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as Native
    from veomni.models.transformers.qwen3_router.modeling_qwen3_router import Qwen3RotaryEmbedding
    config = Qwen3Config(hidden_size=32, num_attention_heads=4, head_dim=8, rope_theta=1000000.)
    position = torch.tensor([[0, 16, 2053, 16383, 32767]])
    hidden = torch.zeros(1, 5, 32)
    for actual, expected in zip(Qwen3RotaryEmbedding(config)(hidden, position), Native(config)(hidden, position)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_public_evaluation_and_separate_scoring(tmp_path, monkeypatch):
    import json
    import sys
    from evaluate import main
    base, checkpoint, _ = build_export(tmp_path, "qwen3", monkeypatch)
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps({"prompt": "v4 v5", "answers": ["v6"]}) + "\n")
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", str(base), "--router", str(checkpoint),
        "--input", str(examples), "--output", str(output), "--device", "cpu", "--max-new-tokens", "3"])
    main()
    assert json.loads((output / "inference_passed.json").read_text())["count"] == 1
    first = (output / "predictions.jsonl").read_bytes()
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--stage", "score", "--output", str(output)])
    main()
    assert (output / "predictions.jsonl").read_bytes() == first
    assert json.loads((output / "scores.json").read_text())["count"] == 1
    (output / "predictions.jsonl").write_bytes(first + b"\n")
    with pytest.raises(ValueError, match="Inference evidence changed"):
        main()
