# Result boundary

The existing Qwen3.5-2B historical reference used Router v4.2's robust decision-margin method with global batch 64, microbatch 8 per GPU, eight data-parallel ranks, one accumulation round, and checkpoints every 1024 updates. The following RULER16K values used threshold zero and one training seed. They are historical observations, not results of the new microbatch-1 default or of the native-vLLM speed extension.

| Router export | RULER16K score | Decode Full rate |
|---|---:|---:|
| Step 1024 | 92.15 | 21.77% |
| Step 2048 | 94.08 | 44.33% |
| Step 3072 | 93.48 | 34.56% |
| Equal-weight mean of 1024/2048/3072 | 93.47 | 30.23% |

This repository has no checkpoint or prediction artifact from that run. Reproduction with a newly prepared run must report its own scores and Full rates. In particular, the default microbatch-1 recipe changes gradient grouping and has no completed result in this package. The Qwen3-8B, Qwen3.5-27B, and Gemma-4-12B configurations are backbone migrations and must not inherit these Qwen3.5-2B values.

Decode speed must report the benchmark's `pooled_decode_tps`, prompt length, GPU, runtime version, warmup/repeat count, and quality qualification. No decode-speed value is asserted here because the portable package has not been run in the pinned native GPU environment.
