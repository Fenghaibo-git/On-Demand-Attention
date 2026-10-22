# Paper results

All results below come from [On-Demand Attention, arXiv:2609.20734v2](https://arxiv.org/abs/2609.20734v2), dated September 26, 2026. The README displays the paper's original [Figure 1](https://arxiv.org/html/2609.20734v2#S1.F1) and [Table 1](https://arxiv.org/html/2609.20734v2#S3.T1); their sources are recorded in [assets/README.md](assets/README.md).

## RULER

Average columns from [Table 1](https://arxiv.org/html/2609.20734v2#S3.T1). Scores and Full-call rates are averaged equally across 4K, 8K, 16K, 32K, and 64K. Within each length, Full-call rates pool routed decoding positions. Qwen3-1.7B uses a different recall head at 16K; see the paper's [evaluation settings](https://arxiv.org/html/2609.20734v2#A2.SS1).

| Model | Full attention | StreamingLLM | ODA | ODA Full-call rate |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 79.98 | 27.21 | 79.29 | 58.43% |
| Qwen3-8B | 91.45 | 32.87 | 90.72 | 51.75% |
| Qwen3.5-2B | 94.11 | 34.04 | 93.52 | 42.33% |
| Qwen3.5-27B | 96.05 | 37.98 | 94.90 | 31.89% |
| Gemma-4-12B-it | 96.48 | 35.65 | 95.95 | 66.88% |

## LongBench

Average columns from [Table 2](https://arxiv.org/html/2609.20734v2#S3.T2), covering the paper's **ten LongBench v1 tasks**. Scores and Full-call rates are task means.

| Model | Full attention | StreamingLLM | ODA | ODA Full-call rate |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 39.50 | 27.06 | 39.20 | 80.12% |
| Qwen3-8B | 45.48 | 32.47 | 45.40 | 88.35% |
| Qwen3.5-2B | 41.57 | 29.35 | 41.55 | 82.23% |
| Qwen3.5-27B | 51.71 | 45.14 | 51.55 | 53.07% |
| Gemma-4-12B-it | 51.47 | 43.40 | 50.97 | 58.65% |

## Controlled decoding speed

[Figure 1 and Appendix B.7](https://arxiv.org/html/2609.20734v2#A2.SS7): Qwen3-1.7B, vLLM 0.18, A100 GPUs, batch 1, fixed continuations, and prescribed Full-call schedules. Throughput excludes prefill and is the median of three requests after warm-up. At 516,788 input tokens, TP4, and a 12.5% Full-call rate, ODA achieves **205.71 tokens/s**, versus **77.64 tokens/s** for native Full attention: **2.65× throughput**. This timing experiment is independent of the learned-policy quality evaluations above.

## Reproduction status

These are published paper results. The current release provides code and configuration templates; its default microbatch-1 training recipe has not been evaluated to establish these numbers. Paper experiments use model- and benchmark-specific heads and settings, detailed in [Appendix B.1](https://arxiv.org/html/2609.20734v2#A2.SS1). New runs must report their own scores, Full-call rates, and timing settings.
