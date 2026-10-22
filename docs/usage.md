# Usage guide

Run the commands below from the repository root. For a short overview, see the [English README](../README.md) or [Chinese README](../README_zh.md).

Train a lightweight Router to choose between Full and Local attention while keeping the language model frozen. This repository provides Router training, Transformers reference inference, vLLM acceleration, and evaluation scripts.

**Release status:** code and documentation. Trained Router weights will be released separately on **Hugging Face**; links will be added here when available. For now, inference requires an export from your own training run.

## Supported implementations

| Backbone | Training | Transformers | vLLM |
|---|---|---|---|
| Qwen3-8B | Yes | Yes | Experimental |
| Qwen3.5-2B | Yes | Yes | Experimental |
| Qwen3.5-27B | Yes | Yes | Experimental |
| Gemma-4-12B-it | Yes | Yes | Not included |

The vLLM implementations use serial requests, greedy decoding, and tensor parallel size 1. They are research runtimes with pinned dependencies, not general-purpose serving backends. Gemma uses its native sliding/global attention structure and a separate Transformers environment.

**Defaults:** inference/evaluation batch size **1**, training microbatch **1 per GPU**. The formal training recipe retains global batch 64 through eight GPUs and eight accumulation rounds. Published paper results and their evaluation settings are summarized in [RESULTS.md](../RESULTS.md).

## Setup

```bash
git clone https://github.com/Fenghaibo-git/On-Demand-Attention.git
cd On-Demand-Attention
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

Follow [ENVIRONMENTS.md](../ENVIRONMENTS.md) to install the Transformers or vLLM environment. Copy `.env.example` to `.env`, set your model/data/run locations, then load it:

```bash
set -a
source .env
set +a
```

Base models, datasets, checkpoints, and generated outputs live outside this checkout. Obtain base models and datasets under their respective licenses. Each model needs tokenized data produced with its own tokenizer.

## Train

For Qwen3.5-2B, prepare the tokenized data described in the [reproduction guide](reproduction.md), then:

```bash
python scripts/prepare_run.py --variant qwen35_2b_mb1 \
  --model-path "$ROUTER_MODEL_ROOT/Qwen3.5-2B" \
  --train-path "$ROUTER_DATA_ROOT/processed/oda-qwen35/train/data" \
  --dataset-manifest "$ROUTER_DATA_ROOT/processed/oda-qwen35/manifest.json" \
  --run-name oda-qwen35-2b
bash scripts/smoke.sh "$ROUTER_RUN_ROOT/oda-qwen35-2b"
bash scripts/train.sh "$ROUTER_RUN_ROOT/oda-qwen35-2b"
```

Other variants are `qwen3_8b`, `qwen35_27b`, and `gemma4_12b`; use the corresponding base model and dataset. Formal training uses online W&B and recoverable DCP checkpoints. A single-GPU diagnostic is available through `bash scripts/quick_smoke.sh RUN_DIR`. See the [reproduction guide](reproduction.md) for data preparation, resume, checkpoint auditing, and the three-checkpoint merge.

## Inference

Transformers, with one prompt at a time:

```bash
python scripts/generate.py --backend hf \
  --model "$ROUTER_MODEL_ROOT/Qwen3.5-2B" \
  --router "$ROUTER_RUN_ROOT/oda-qwen35-2b/trainable_exports/global_step_3072/trainable_state.pt" \
  --prompt "Explain how attention works." --max-new-tokens 256
```

The same entry accepts Qwen3 and Gemma exports in their respective environments. Prompts use the model's chat template with thinking disabled; `--raw` accepts an already formatted prompt.

For vLLM, first validate the export in the **Transformers environment**:

```bash
python scripts/prepare_inference.py \
  --model "$ROUTER_MODEL_ROOT/Qwen3.5-2B" \
  --router "$ROUTER_RUN_ROOT/oda-qwen35-2b/trainable_exports/global_step_3072/trainable_state.pt" \
  --output "$ROUTER_RUN_ROOT/oda-qwen35-2b/vllm-admission"
```

Switch to the **vLLM environment**, keep the same checkout and asset paths, and run:

```bash
python scripts/generate.py --backend vllm \
  --admission "$ROUTER_RUN_ROOT/oda-qwen35-2b/vllm-admission/admission.json" \
  --prompt "Explain how attention works." --max-new-tokens 256
```

Admission checks export compatibility and binds its source files; it does not certify benchmark quality. The native runtime also checks the base-model configuration revision. Use base directory names `Qwen3-8B`, `Qwen3.5-2B`, or `Qwen3.5-27B` for vLLM.

## Evaluation

For a small custom evaluation, provide JSONL with one `prompt` and a nonempty `answers` list per line:

```json
{"prompt": "What is 2 + 2? Answer with one number.", "answers": ["4"]}
```

```bash
python scripts/evaluate.py --backend hf \
  --model "$ROUTER_MODEL_ROOT/Qwen3.5-2B" \
  --router "$ROUTER_RUN_ROOT/oda-qwen35-2b/trainable_exports/global_step_3072/trainable_state.pt" \
  --input "$ROUTER_DATA_ROOT/processed/my-eval/examples.jsonl" \
  --output "$ROUTER_RUN_ROOT/oda-qwen35-2b/eval-custom"
```

This reports case-insensitive, whitespace-normalized exact match. All listed Transformers models are supported; vLLM uses `--backend vllm --admission ...` instead of `--model/--router`. Inference and scoring have separate completion records. Use `--stage score --output ...` to repeat scoring without rerunning inference.

The dedicated [RULER16K and decode-speed workflow](reproduction.md#merge-and-ruler16k) currently covers Qwen3.5-2B. Generic JSONL exact match is a separate metric and must not be reported as a RULER score.

## Method

The backbone stays frozen while the Router learns to predict the benefit of Full attention. Define the gain as:

$$
g = \mathrm{NLL}_{\mathrm{Local}} - \mathrm{NLL}_{\mathrm{Full}}
$$

The Router uses Huber loss with delta 1 and the target:

$$
y = \mathrm{sign}(g - 0.001)\log(1 + |g - 0.001|)
$$

Eligible tokens are equally weighted across the entire optimizer step. History selects Local when the gain is negative; inference selects Full when the Router score is strictly positive. Full prefill is followed by selected-cache decode, with sink 4 and local window 2048.

## Validation and license

CPU tests cover the objective, global token normalization, cache behavior, and tiny-model export/loading/generation. They do not establish full-model GPU quality or throughput. Run `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests` in the relevant Transformers environment; Gemma tests require its separate environment.

The repository is licensed under [GPL-3.0](../LICENSE), retaining the target repository's original license. Bundled upstream components retain their copyright notices and [Apache-2.0 terms](../licenses/Apache-2.0.txt). See [NOTICE](../NOTICE) and [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) for attribution and modifications.
