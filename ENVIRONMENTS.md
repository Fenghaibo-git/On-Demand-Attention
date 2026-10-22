# Environments

Use separate Python 3.10 environments for Transformers and vLLM. Commands below pin the core versions used by these implementations. CUDA training and native acceleration require an NVIDIA GPU and a compatible driver. A local CUDA toolkit is also needed when building the training kernels.

## Transformers: Qwen3 and Qwen3.5

```bash
python3.10 -m venv /path/to/envs/oda-hf
source /path/to/envs/oda-hf/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/transformers.txt
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

For Qwen3.5 GPU training, also install the fused kernels after PyTorch:

```bash
python -m pip install ninja packaging
python -m pip install --no-build-isolation flash-linear-attention==0.5.2 causal-conv1d==1.6.2.post1
```

Training uses PyTorch 2.7.1, Transformers 5.9.0, W&B 0.28.1, and the selected VeOmni source bundled here. CPU tests and Transformers reference inference can use the unfused fallback. Installing the generic upstream VeOmni package over this checkout can select incompatible dependencies.

## Transformers: Gemma

Create another environment using the same Python and PyTorch versions. Install `requirements/gemma.txt` in place of `requirements/transformers.txt`, then run:

```bash
python scripts/patch_gemma_environment.py
```

Gemma requires official Transformers revision `dd211984bd751c0c7a2a0b9d74e01000359a77dd` (5.10.0). The patch makes its optional Hugging Face DCP writer import conditional because that writer is absent from the qualified PyTorch 2.7.1 build. VeOmni's own recoverable DCP path remains enabled; unsupported HF-format distributed export still raises an error. The patch checks the source before editing the active environment.

## vLLM

```bash
python3.10 -m venv /path/to/envs/oda-vllm
source /path/to/envs/oda-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/vllm.txt
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

This uses vLLM 0.18.0, PyTorch 2.10.0, and Transformers 4.57.6. It supports BF16, FlashAttention, one request at a time, greedy decode, and TP1/PP1. Prefix caching, chunked prefill, speculative decoding, and asynchronous scheduling are disabled by the entry. Keep the same source and asset paths when switching environments because admission records bind those files.

The position-source adapters check specific vLLM source hashes and reject incompatible installations. Relevant contracts are in `src/qwen3_5_router_runtime_experiments/vllm_contract.py` and each backend's `serial_position_source.py`. CPU import checks do not validate GPU kernels or speed.

## RULER evaluation

Install OpenCompass in the Transformers environment. The scripts target OpenCompass source revision `ade472440cdf0fb8064b4d0ee57102d8e0c06523`:

```bash
git clone https://github.com/open-compass/opencompass.git /path/to/opencompass
git -C /path/to/opencompass checkout ade472440cdf0fb8064b4d0ee57102d8e0c06523
python -m pip install -e /path/to/opencompass
```

Preserve the pinned Transformers/PyTorch versions when resolving optional OpenCompass dependencies. The included `oda_opencompass` adapter retains the chat-template and truncation behavior used by the reference evaluation. Set `OPENCOMPASS_ROOT` and `RULER_DATA_ROOT` in `.env`; obtain RULER's source datasets separately. RULER defaults to batch 1. `ROUTER_PAPER_BATCH=1` explicitly selects the historical batch-2 evaluation configuration for both inference and scoring.

Record `pip freeze`, GPU/driver, base-model revision, data manifest, and source commit with each new experiment. Neither model weights nor container images are redistributed here.
