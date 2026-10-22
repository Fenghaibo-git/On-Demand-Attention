# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import torch


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .chat_template import ChatTemplate


def split_into_chunks(sequence: Sequence[int], chunk_size: int) -> List[List[int]]:
    """
    Splits a long sequence into chunks.
    """
    total_len = len(sequence)
    chunks = []
    for i in range(0, total_len, chunk_size):
        chunks.append(sequence[i : i + chunk_size])

    return chunks


def process_pretrain_example(
    example: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "content_split",
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokens = tokenizer.encode(text_example, add_special_tokens=False) + [tokenizer.eos_token_id]
    for input_ids in split_into_chunks(tokens, max_seq_len):
        examples.append(
            {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor([1] * len(input_ids)),
                "labels": torch.tensor(input_ids),
            }
        )

    return examples


def process_sft_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokenized_example = chat_template.encode_messages(text_example, max_seq_len=max_seq_len)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    return [tokenized_example]


def process_tokenized_example(
    example: Dict[str, Any],
    max_seq_len: int,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    del source_name
    if "input_ids" not in example:
        raise KeyError("A tokenized example must contain `input_ids`.")

    input_ids = torch.as_tensor(example["input_ids"], dtype=torch.long)[:max_seq_len]
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("Tokenized `input_ids` must be a non-empty one-dimensional sequence.")

    if "labels" in example and example["labels"] is not None:
        labels = torch.as_tensor(example["labels"], dtype=torch.long)[:max_seq_len]
    elif "loss_mask" in example and example["loss_mask"] is not None:
        loss_mask = torch.as_tensor(example["loss_mask"], dtype=torch.bool)[:max_seq_len]
        if loss_mask.shape != input_ids.shape:
            raise ValueError("Tokenized `loss_mask` must have the same length as `input_ids`.")
        labels = input_ids.clone().masked_fill(~loss_mask, -100)
    else:
        labels = input_ids.clone()

    if labels.shape != input_ids.shape:
        raise ValueError("Tokenized `labels` or `loss_mask` must have the same length as `input_ids`.")

    if "attention_mask" in example and example["attention_mask"] is not None:
        attention_mask = torch.as_tensor(example["attention_mask"], dtype=torch.long)[:max_seq_len]
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Tokenized `attention_mask` must have the same length as `input_ids`.")
    else:
        attention_mask = torch.ones_like(input_ids)

    return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]
