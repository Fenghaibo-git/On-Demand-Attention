# On-Demand Attention

**Language Models Know When to Recall**

[**arXiv:2609.20734**](https://arxiv.org/abs/2609.20734) | [English](README.md) | [简体中文](README_zh.md)

On-Demand Attention (ODA) runs Local attention first, then uses a lightweight recall head to decide whether to recompute the step with Full attention. Only the recall head is trained; the language model stays frozen, and the complete historical KV cache remains available for future recall.

[![Paper Figure 1: ODA method overview, decoding speed, and decode FLOPs](assets/oda-main-overview.png)](https://arxiv.org/html/2609.20734v2#S1.F1)

*Figure 1. ODA decoding and execution efficiency. Efficiency panels use fixed continuations and prescribed Full-call rates.*

## Results

[![Paper Table 1: complete RULER results at 4K, 8K, 16K, 32K, and 64K](assets/oda-ruler-table1.png)](https://arxiv.org/html/2609.20734v2#S3.T1)

*Table 1. RULER results from the paper. Percentages below ODA scores are Full-call rates; Avg averages scores and rates separately over the five lengths.*

[Experiment settings and reproduction status](RESULTS.md).

## Quick start

This release includes **Router v4.2 training, Transformers inference, evaluation, and experimental vLLM backends** for Qwen3/Qwen3.5. Gemma uses Transformers; see the [supported models and runtime limits](docs/usage.md#supported-implementations). Inference/evaluation batch size and training microbatch per GPU default to **1**.

Trained Router weights will be released separately on **Hugging Face**. Until then, use an export from your own training run.

```bash
git clone https://github.com/Fenghaibo-git/On-Demand-Attention.git
cd On-Demand-Attention
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

Install the [required environment](ENVIRONMENTS.md), then run Transformers inference with a local base model and a trained Router export:

```bash
python scripts/generate.py --backend hf \
  --model /path/to/models/Qwen3.5-2B \
  --router /path/to/trainable_state.pt \
  --prompt "Explain how attention works." --max-new-tokens 256
```

[Training](docs/reproduction.md#train-and-audit) · [vLLM inference](docs/usage.md#inference) · [Evaluation](docs/usage.md#evaluation) · [RULER16K workflow](docs/reproduction.md#merge-and-ruler16k)

## License

[GPL-3.0](LICENSE). Upstream copyright and license notices are retained; see [NOTICE](NOTICE).
