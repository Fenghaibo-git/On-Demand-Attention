"""Qwen3.5-2B v4.2 RULER16K: 13 tasks, 100 examples each."""

import os
from mmengine.config import read_base
from oda_opencompass import ODAChatModel

with read_base():
    from .datasets.ruler.ruler_16k_gen import ruler_datasets


runtime = os.environ["ROUTER_HF_RUNTIME"]
models = [
    dict(
        type=ODAChatModel,
        abbr="qwen35-2b-router-v42-ruler16k",
        path=runtime,
        tokenizer_path=runtime,
        max_seq_len=16384,
        max_out_len=512,
        batch_size=1,
        tokenizer_kwargs=dict(padding_side="left", truncation_side="left", trust_remote_code=True),
        chat_template_kwargs=dict(enable_thinking=False),
        model_kwargs=dict(torch_dtype="auto", device_map="auto", trust_remote_code=True,
                          attn_implementation="sdpa"),
        generation_kwargs=dict(do_sample=False, num_beams=1, num_return_sequences=1),
        run_cfg=dict(num_gpus=1),
    )
]
datasets = ruler_datasets
