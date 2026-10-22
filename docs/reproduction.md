# Router v4.2 reproduction code

This package contains the v4.2 Router training implementation, Qwen3.5-2B RULER16K evaluation, and a separate native-vLLM decode-speed experiment. It contains source code and configuration templates only. Model weights, datasets, checkpoints, generated predictions, W&B credentials, and machine-specific paths are deliberately absent.

The v4.2 method trains a frozen-backbone Router on the eligible-token mean Huber loss with target $\operatorname{sign}(g-0.001)\log(1+|g-0.001|)$, where $g = \mathrm{NLL}_{\mathrm{Local}} - \mathrm{NLL}_{\mathrm{Full}}$. The history input selects Local only when $g<0$; deployment selects Full only when the finite Router score is `> 0`. The default new run has global batch 64, eight data-parallel ranks, microbatch one per rank, eight accumulation rounds, 3072 updates, a 102-step warmup, cosine decay from `3e-4` to `3e-5`, and DCP plus Router export every 512 updates. The main checkpoints are steps 1024, 2048, and 3072 and their equal-weight parameter merge.

`qwen35_2b_paper_reference_mb8` reproduces the historical v4.2 **method** with microbatch eight, one accumulation round, and a 1024-step save interval. Its historical scores belong to that historical configuration; no scores are claimed for the new MB1 configuration. The Qwen3-8B, Qwen3.5-27B, and Gemma-4-12B templates are separately identified backbone migrations, not Qwen3.5-2B results. See [RESULTS.md](../RESULTS.md) and [ENVIRONMENTS.md](../ENVIRONMENTS.md).

For Qwen3.5-2B, the seeded A0 head has 22 tensors and 28,325,889 parameters. `prepare_run.py` checks its tensor-state SHA-256 against the verified shared initialization before creating the run configuration. Different backbones generate their own shape-compatible initialization and keep that identity in their scientific contract.

## Layout and external assets

`src/veomni` and `src/tasks/train_torch.py` implement training, global eligible-token aggregation, local metric logging, W&B comparison metrics, and DCP. `src/tools` contains the Qwen3.5 retokenization and long-run validator. `configs/` freezes model, training, metric, and RULER settings. `src/hf_qwen35` is the selected-cache HF evaluator. `src/qwen35_2b_absolute_vllm` and `src/qwen3_5_router_runtime_experiments` implement the native speed extension. `scripts/` contains the runnable stages and audits.

Set the variables in `.env.example` for your own machine without committing the resulting `.env`. `ROUTER_DATA_ROOT` must have `raw/`, `processed/`, and `cache/`; `ROUTER_MODEL_ROOT` holds separately obtained base models; `ROUTER_RUN_ROOT` holds generated runs and checkpoints. Review each asset directory's README before creating or modifying assets. The preparation scripts enforce these roots, and generated dataset/run directories receive their own README. Licenses for datasets and base models are separate from this source release.

Use the environment versions in [ENVIRONMENTS.md](../ENVIRONMENTS.md). `PYTHONPATH="$PWD/src"` makes the selected VeOmni source importable. Training uses eight local GPUs and online W&B. The new run requires real model weights, the complete tokenized train split of 196,608 rows, and that dataset's `manifest.json`. For Qwen3.5-2B the builder is:

```bash
python3 src/tools/prepare_qwen3_5_step35_router_dataset.py \
  --source-root "$ROUTER_DATA_ROOT/raw/step35/json/general" \
  --tokenizer-path "$ROUTER_MODEL_ROOT/Qwen3.5-2B" \
  --output-root "$ROUTER_DATA_ROOT/processed/router-v42-qwen35"
```

The raw Step-3.5 manifest and source records must be acquired separately. The builder verifies the upstream revision, deduplicates before the split, and writes its own manifest/README. It never substitutes Qwen3 token IDs for Qwen3.5 token IDs. For another backbone, supply its own compatible tokenized split and manifest; do not reuse this split across tokenizers.

## Train and audit

These commands create a **new** run. The two-update smoke executes real weights, saves DCP, restores step 1, and records an estimate for the configured save interval. It is a prerequisite for formal training.

If only one suitable GPU is free, `scripts/quick_smoke.sh RUN_DIR` performs one real-weight v4.2 forward/backward optimizer update, a DCP save, and a Router export at the 16K data boundary. It uses DP1 and batch 1 and records those differences in the run's `qualification/quick_smoke/diagnostic_contract.json`; it does not replace the eight-GPU smoke required before formal training. Run it in the pinned training environment after `prepare_run.py`, with exactly one GPU visible.

```bash
export PYTHONPATH="$PWD/src"
python3 scripts/prepare_run.py --variant qwen35_2b_mb1 \
  --model-path "$ROUTER_MODEL_ROOT/Qwen3.5-2B" \
  --train-path "$ROUTER_DATA_ROOT/processed/router-v42-qwen35/train/data" \
  --dataset-manifest "$ROUTER_DATA_ROOT/processed/router-v42-qwen35/manifest.json" \
  --run-name router-v42-qwen35-2b
scripts/smoke.sh "$ROUTER_RUN_ROOT/router-v42-qwen35-2b"
scripts/train.sh "$ROUTER_RUN_ROOT/router-v42-qwen35-2b"
```

Formal training checks the comparison schema, W&B account, exact step aggregation, recoverable DCP, and source/config contract. It writes a local metric row before the corresponding W&B row. `scripts/audit_run.py` checks steps 1–3072, all required save points, DCP metadata and referenced shard lengths, FP32 Router exports, and the W&B run identity. `training_passed.json` is written only after that audit. A DCP payload needs a real restore to prove its bytes are usable; the smoke tests that path before training.

On an interrupted formal run, use `scripts/resume_train.sh RUN_DIR [CHECKPOINT_STEP]`. It selects a structurally and semantically verified DCP, archives the full old ledger under the run's `recovery/attempt_*` directory, retains only the verified prefix for the trainer, and resumes the same W&B run with `resume=must`. It refuses to resume a completed run or to infer state from metrics alone. Each failed attempt remains under `recovery/`.

## Merge and RULER16K

The equal-weight merge uses only steps 1024, 2048, and 3072:

```bash
python3 scripts/merge_router.py "$ROUTER_RUN_ROOT/router-v42-qwen35-2b"
python3 scripts/prepare_hf_runtime.py "$ROUTER_RUN_ROOT/router-v42-qwen35-2b" --merge
scripts/evaluate_ruler.sh infer "$ROUTER_RUN_ROOT/router-v42-qwen35-2b" merge_1024_2048_3072
scripts/evaluate_ruler.sh score "$ROUTER_RUN_ROOT/router-v42-qwen35-2b" merge_1024_2048_3072
```

For individual checkpoints, replace `--merge` with `--step 1024`, `2048`, or `3072`, then use the matching `step_*` case. `RULER_DATA_ROOT` must contain separately obtained RULER source files; `OPENCOMPASS_ROOT` must point to a compatible OpenCompass checkout. The protocol is 13 tasks × 100 examples, 16K context, maximum 512 output tokens, greedy, batch 1, seed 42, thinking disabled, SDPA, Full prefill, selected-cache decode, and Router threshold zero. Inference and scoring receive distinct success markers. Existing incomplete case directories are preserved for recovery; move them to the run's `recovery/attempt_*` directory before retrying rather than overwriting them.

After scoring, run `python3 scripts/analyze_ruler.py CASE_DIR` for each of the four cases, then `python3 scripts/summarize_ruler.py RUN_DIR`. The analysis verifies exactly 100 scored examples for each of 13 tasks and sums actual selected-cache decode decisions to derive the Full rate. The final summary is written only when all four hash-bound case analyses pass.

## Native decode speed

The native vLLM path is an **acceleration experiment** over the same audited Qwen3.5-2B Router head. It is separate from the HF RULER quality numbers. Prepare head admission in the Transformers 5 environment, then switch to the pinned vLLM 0.18 environment and run each mode in a fresh process:

```bash
python3 scripts/prepare_vllm_admission.py \
  "$ROUTER_RUN_ROOT/router-v42-qwen35-2b/evaluations/merge_1024_2048_3072/hf_runtime"
python3 scripts/benchmark_vllm.py \
  "$ROUTER_RUN_ROOT/router-v42-qwen35-2b/evaluations/merge_1024_2048_3072/hf_runtime" --mode router
python3 scripts/benchmark_vllm.py \
  "$ROUTER_RUN_ROOT/router-v42-qwen35-2b/evaluations/merge_1024_2048_3072/hf_runtime" --mode full
python3 scripts/summarize_speed.py \
  "$ROUTER_RUN_ROOT/router-v42-qwen35-2b/evaluations/merge_1024_2048_3072"
```

Optional `--mode local` and `--mode native` measure additional baselines. The default prompts are synthetic 2048/4096/8192-token contexts; each mode uses one warmup and three measured requests per length, a fixed forced continuation, 128 measured decode steps, and excludes the first output token. `pooled_decode_tps` is total measured decode tokens divided by total measured decode seconds. `summarize_speed.py` checks that all modes use the same protocol/source hashes before calculating ratios. These scripts do not establish quality equivalence; evaluate that separately before making a quality-preserving speed claim.

## Release verification

Run `python3 -B scripts/check_release.py` and `python3 -B scripts/verify_source_manifest.py` before attaching or publishing. They fail on machine-specific paths, email addresses, credentials, internal addresses, symlinks, model/data/checkpoint binaries, caches, unlisted file types, or changed copied source; the privacy scan prints finding locations only. `SOURCE_MANIFEST.json` lists the origin and original/release hashes of selected source files. `NOTICE` describes adaptations. The release has no embedded experimental outputs or private W&B identity. `python3 -B scripts/package_release.py` produces a deterministic source-only ZIP archive and an adjacent SHA-256 checksum outside this directory after both checks pass.

Historical validation of the earlier reproduction package used the pinned training environment, a real Qwen3.5-2B base model and the complete tokenized train split. The optional single-GPU quick smoke completed one optimizer update on a 9,763-token example with 7,710 eligible tokens, finite loss, a DCP save and a Router export; its step took 62.3 seconds on an 80 GB A100. The generated run and its logs are outside this source package. This check does not establish the formal eight-GPU trajectory, DCP restore, RULER inference or native-vLLM GPU throughput. Package-level checks also covered syntax, all five templates, imports in the pinned training and native-vLLM containers, the 22-tensor Qwen3.5 A0 head, strict loading of an existing audited head, RULER configuration parsing, source hashes, and privacy rules. Published paper results are summarized separately in [RESULTS.md](../RESULTS.md).

The GitHub release adds the common inference and JSONL evaluation entries in the root README. Release validation uses CPU tests and import checks; full-model GPU training, benchmark quality, and speed have not been rerun for this release. Set `ROUTER_PAPER_BATCH=1` for both RULER stages only when explicitly reproducing the historical batch-2 evaluation protocol.
