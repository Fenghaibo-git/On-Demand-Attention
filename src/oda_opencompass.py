# Adapted for On Demand Attention; see NOTICE.
# flake8: noqa
# yapf: disable
import atexit
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from mmengine.device import is_npu_available

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


_HF_BENCHMARK_SUMMARY_FORMAT = (
    'qwen3-hf-opencompass-active-generate-process-v1')


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2,
                   sort_keys=True) + '\n',
        encoding='utf-8')
    os.replace(temporary, path)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_cuda_device_indices(model: Any) -> List[int]:
    if not torch.cuda.is_available():
        return []
    device_values = list(getattr(model, 'hf_device_map', {}).values())
    if not device_values:
        device_values = [getattr(model, 'device', None)]
    indices = set()
    concrete_device_seen = False
    for value in device_values:
        if isinstance(value, int):
            concrete_device_seen = True
            indices.add(value)
            continue
        if value is None:
            continue
        concrete_device_seen = True
        if str(value) in {'cpu', 'disk', 'meta'}:
            continue
        try:
            device = torch.device(value)
        except (RuntimeError, TypeError):
            continue
        if device.type == 'cuda':
            indices.add(
                device.index
                if device.index is not None else torch.cuda.current_device())
    if not indices and not concrete_device_seen:
        indices.add(torch.cuda.current_device())
    return sorted(indices)


def _synchronize_model_accelerators(model: Any) -> None:
    """Make timing include work on the CUDA devices that hold the model."""
    if torch.cuda.is_available():
        for index in _model_cuda_device_indices(model):
            torch.cuda.synchronize(index)
    elif is_npu_available() and hasattr(torch, 'npu'):
        torch.npu.synchronize()


def _terminal_aware_output_counts(
        outputs: torch.Tensor,
        eos_token_ids: List[int]) -> Dict[str, int]:
    """Count useful generated IDs and actual padded batch decode slots.

    ``model.generate`` returns a rectangular tensor. Rows that encounter EOS
    early may contain trailing pad IDs while the rest of the batch continues.
    ``output_tokens`` ends each row at its first EOS (inclusive), whereas
    ``generated_sequence_slots`` retains every rectangular position and is the
    closest count of the row-token work performed by batched HF decoding.
    """
    if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
        raise TypeError('HF benchmark accounting requires a rank-two tensor')
    eos = {int(value) for value in eos_token_ids}
    output_tokens = 0
    rows_with_eos = 0
    rows_without_eos = 0
    for row in outputs.detach().to(device='cpu', dtype=torch.long).tolist():
        count = len(row)
        first_eos = next((index for index, value in enumerate(row)
                          if int(value) in eos), None)
        if first_eos is not None:
            count = first_eos + 1
            rows_with_eos += 1
        else:
            rows_without_eos += 1
        output_tokens += count
    return {
        'output_tokens': output_tokens,
        'generated_sequence_slots': int(outputs.numel()),
        'rows_with_eos': rows_with_eos,
        'rows_without_eos': rows_without_eos,
    }


def _middle_truncate_token_fields(tokens: Dict[str, torch.Tensor],
                                  max_length: int) -> Dict[str, torch.Tensor]:
    """Keep the beginning and end of overlong token fields.

    LongBench places task instructions/questions near a prompt boundary.  Its
    reference 16K protocol therefore removes tokens from the middle when the
    prompt does not fit.  The tokenizer must not truncate first; otherwise the
    beginning has already been lost before this function runs.

    The returned fields retain exactly ``max_length`` sequence positions when
    truncation is required, including when ``max_length`` is odd.
    """
    if max_length <= 0:
        raise ValueError('max_length must be positive for middle truncation')

    head_length = max_length // 2
    tail_length = max_length - head_length
    truncated = {}
    for key, value in tokens.items():
        if value.ndim >= 2 and value.shape[1] > max_length:
            truncated[key] = torch.cat(
                (value[:, :head_length], value[:, -tail_length:]), dim=1)
        else:
            truncated[key] = value
    return truncated


def _validate_strict_no_truncation_budget(
        tokens: Dict[str, torch.Tensor],
        max_prompt_length: int) -> List[int]:
    """Validate every unpadded prompt against a hard token budget.

    This is deliberately separate from tokenizer truncation.  Formal
    no-truncation evaluations must fail before model execution when even one
    rendered prompt cannot reserve its task-specific generation allowance.
    """
    if max_prompt_length <= 0:
        raise ValueError(
            'max_prompt_length must be positive for strict no-truncation')
    if 'input_ids' not in tokens:
        raise ValueError('Strict no-truncation validation requires input_ids')

    input_ids = tokens['input_ids']
    if input_ids.ndim != 2:
        raise ValueError(
            'Strict no-truncation validation expects batched input_ids')
    attention_mask = tokens.get('attention_mask')
    if attention_mask is None:
        lengths = [int(input_ids.shape[1])] * int(input_ids.shape[0])
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                'attention_mask shape differs from input_ids in strict mode')
        lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]

    overflow = [
        {'batch_index': index, 'prompt_tokens': length}
        for index, length in enumerate(lengths)
        if length > max_prompt_length
    ]
    if overflow:
        raise ValueError(
            'Strict no-truncation token budget exceeded: '
            f'max_prompt_tokens={max_prompt_length}, overflow={overflow}')
    return lengths


def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(self.max_sequence_id_len, len(sequence_ids))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(s in lookback_tokens_batch[i] for s in self.stop_words)
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])

def _get_possible_max_seq_len(max_seq_len, path):
    if max_seq_len is not None:
        return max_seq_len

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    possible_keys = [
        'max_position_embeddings',
        'seq_length',
        'model_max_length',
    ]
    for k in possible_keys:
        if hasattr(config, k):
            return getattr(config, k)
    raise ValueError('max_seq_len is not provided and cannot be inferred from the model config.')


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def _format_with_fast_chat_template(inputs: List[str], name: str='vicuna'):
    try:
        from fastchat.model import get_conversation_template
    except ImportError:
        raise ModuleNotFoundError('fastchat not found. Please install with\npip install "fschat[model_worker,webui]"')

    outputs = []
    for _input in inputs:
        template = get_conversation_template(name)
        for item in _input:
            if item['role'] == 'user':
                template.append_message(template.roles[0], item['content'])
            elif item['role'] == 'assistant':
                template.append_message(template.roles[1], item['content'])
            elif item['role'] == 'system':
                continue
            else:
                raise ValueError(f"Unknown role {item['role']}")
        template.append_message(template.roles[1], None)
        outputs.append(template.get_prompt())
    return outputs


def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]
    )
    return APITemplateParser(meta_template or default_meta_template)


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs


@MODELS.register_module()
class ODAChatModel(BaseModel):
    """Model wrapper for HuggingFace models designed for chat.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 pad_token_id: Optional[int] = None,
                 fastchat_template: Optional[str] = None,
                 chat_template_kwargs: Optional[Dict] = None,
                 stop_words: Optional[str] = [],
                 mode: str = 'none',
                 **other_kwargs):

        self._hf_benchmark_started = time.perf_counter()
        self._hf_benchmark_wrapper_seconds = 0.0
        self._hf_benchmark_model_seconds = 0.0
        self._hf_benchmark_generate_attempts = 0
        self._hf_benchmark_successful_generate_calls = 0
        self._hf_benchmark_request_count = 0
        self._hf_benchmark_prompt_tokens = 0
        self._hf_benchmark_padded_prompt_slots = 0
        self._hf_benchmark_output_tokens = 0
        self._hf_benchmark_generated_sequence_slots = 0
        self._hf_benchmark_rows_with_eos = 0
        self._hf_benchmark_rows_without_eos = 0
        self._hf_benchmark_errors = 0
        self._hf_benchmark_batch_histogram: Dict[str, int] = {}
        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        assert mode in ['none', 'mid', 'strict']
        self.mode = mode
        self.logger.info(f'using stop words: {self.stop_words}')

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

        self._hf_benchmark_model_ready = time.perf_counter()
        self._hf_benchmark_enabled = bool(
            os.environ.get('QWEN3_HF_BENCHMARK_SUMMARY_PATH', '').strip()
        ) and not tokenizer_only
        if self._hf_benchmark_enabled:
            atexit.register(self.flush_hf_benchmark_summary)
            self.flush_hf_benchmark_summary()

    def _hf_eos_token_ids(self, generation_kwargs: Dict[str, Any]) -> List[int]:
        value = generation_kwargs.get('eos_token_id')
        if value is None and hasattr(self.model, 'generation_config'):
            value = self.model.generation_config.eos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    def hf_benchmark_summary(self) -> Dict[str, Any]:
        now = time.perf_counter()
        model_seconds = self._hf_benchmark_model_seconds
        wrapper_seconds = self._hf_benchmark_wrapper_seconds
        output_tokens = self._hf_benchmark_output_tokens
        generated_slots = self._hf_benchmark_generated_sequence_slots
        if self._hf_benchmark_errors:
            status = 'failed'
        elif self._hf_benchmark_successful_generate_calls:
            status = 'passed'
        else:
            status = 'ready'
        cuda_indices = _model_cuda_device_indices(self.model)
        cuda = {
            'available': torch.cuda.is_available(),
            'visible_device_count': (
                torch.cuda.device_count() if torch.cuda.is_available() else 0),
            'model_local_device_indices': cuda_indices,
            'max_memory_allocated_bytes': (
                max((torch.cuda.max_memory_allocated(index)
                     for index in cuda_indices), default=0)
                if torch.cuda.is_available() else 0),
            'max_memory_reserved_bytes': (
                max((torch.cuda.max_memory_reserved(index)
                     for index in cuda_indices), default=0)
                if torch.cuda.is_available() else 0),
        }
        return {
            'format': _HF_BENCHMARK_SUMMARY_FORMAT,
            'status': status,
            'case': os.environ.get('QWEN3_THREE_STREAM_CASE_NAME'),
            'pid': os.getpid(),
            'hostname': platform.node(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'model_path': str(Path(self.path).expanduser().resolve()),
            'versions': {
                'python': platform.python_version(),
                'torch': torch.__version__,
                'transformers': _package_version('transformers'),
                'opencompass': _package_version('opencompass'),
            },
            'timing_seconds': {
                'model_and_tokenizer_load': (
                    self._hf_benchmark_model_ready
                    - self._hf_benchmark_started),
                'wrapper_generate_calls': wrapper_seconds,
                'model_generate_calls': model_seconds,
                'process_elapsed': now - self._hf_benchmark_started,
            },
            'timing_contract': {
                'model_generate_calls': (
                    'sum of wall-clock intervals around model.generate; all '
                    'model CUDA accelerators synchronized immediately before and '
                    'after each call'),
                'wrapper_generate_calls': (
                    'sum of complete ODAChatModel.generate '
                    'calls, including chat rendering, tokenization, model '
                    'generation, decoding, and stop-string trimming'),
                'model_load_excluded_from_both': True,
            },
            'generate_call_attempts': self._hf_benchmark_generate_attempts,
            'successful_generate_calls': (
                self._hf_benchmark_successful_generate_calls),
            'request_count': self._hf_benchmark_request_count,
            'prompt_tokens_unpadded': self._hf_benchmark_prompt_tokens,
            'prompt_sequence_slots_padded': (
                self._hf_benchmark_padded_prompt_slots),
            'output_tokens': output_tokens,
            'generated_sequence_slots': generated_slots,
            'output_token_contract': (
                'sum of generated token IDs through the first EOS per row, '
                'inclusive; a row without EOS contributes its complete '
                'generated suffix'),
            'generated_sequence_slot_contract': (
                'all positions in the rectangular generated suffix; this '
                'captures padded row-token work for batched HF decoding'),
            'rows_with_eos': self._hf_benchmark_rows_with_eos,
            'rows_without_eos': self._hf_benchmark_rows_without_eos,
            'error_count': self._hf_benchmark_errors,
            'batch_size_histogram': dict(sorted(
                self._hf_benchmark_batch_histogram.items(),
                key=lambda item: int(item[0]))),
            'derived': {
                'active_output_tokens_per_second': (
                    output_tokens / model_seconds if model_seconds > 0 else None),
                'active_generated_sequence_slots_per_second': (
                    generated_slots / model_seconds
                    if model_seconds > 0 else None),
                'wrapper_output_tokens_per_second': (
                    output_tokens / wrapper_seconds
                    if wrapper_seconds > 0 else None),
                'active_model_milliseconds_per_output_token': (
                    1000.0 * model_seconds / output_tokens
                    if output_tokens > 0 else None),
            },
            'cuda': cuda,
        }

    def flush_hf_benchmark_summary(self) -> None:
        if not getattr(self, '_hf_benchmark_enabled', False):
            return
        value = os.environ.get(
            'QWEN3_HF_BENCHMARK_SUMMARY_PATH', '').strip()
        if not value:
            return
        target = Path(value.replace('{pid}', str(os.getpid()))).expanduser()
        try:
            _atomic_json(target, self.hf_benchmark_summary())
        except Exception:
            if not getattr(__import__('sys'), 'is_finalizing')():
                raise

    def _load_tokenizer(self, path: Optional[str], kwargs: dict, pad_token_id: Optional[int] = None):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if pad_token_id is not None:
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != pad_token_id:
                self.logger.warning(f'pad_token_id is not consistent. Using {pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = pad_token_id
            return
        if self.tokenizer.pad_token_id is not None:
            return
        self.logger.warning('pad_token_id is not set for the tokenizer.')
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False


    def get_ppl_tokenwise(self, inputs: List[str], label: List[List[int]], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get inference-ppl per token given a list of inputs and label.

        Args:
            inputs (List[str]): A list of strings.
            label (List[List[int]]): A list of list of label, each label is a tuple of (start, end, 1)
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'

        tokens = self.tokenizer(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            import numpy as np
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        loss = loss.cpu().numpy()

        decode_messages = [[self.tokenizer.decode([input_id]) for input_id in token] for token in tokens['input_ids']]
        char_messages = [[ch for ch in message] for message in messages]

        # shifted to align label and loss
        for i in range(len(decode_messages)):
            decode_messages[i] = decode_messages[i][1:]

        aggregated_label_list = [[] for _ in range(len(decode_messages))]

        tag_list = [[] for _ in range(len(decode_messages))]

        for tmp_index, label_list in enumerate(label):
            for single_label in label_list:
                left = single_label[0]
                right = single_label[1]
                for i in range(left, right):
                    aggregated_label_list[tmp_index].append(i)


        def align_sequences(seq1, seq2, sep_len):
            """
            seq1: decoded sequence from token, one token may contain multiple characters
            seq2: original separate character sequence
            """
            i, j = 0, 0
            matched_pairs = []
            while i < len(seq1) and j < len(seq2):
                word = seq1[i]
                if len(word) == 0:
                    matched_pairs.append((word, []))
                    i += 1
                    continue

                if '\ufffd' in word:
                    for _ in range(sep_len):
                        matched_pairs.append((word, [j]))
                        i += 1
                    j += 1
                    continue

                char_sequence = ''
                while j < len(seq2) and (char_sequence != word):
                    char_sequence += seq2[j]
                    if char_sequence == word:
                        matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                        j += 1
                        break
                    elif len(char_sequence) > len(word):
                        if word == char_sequence[-len(word):]:
                            matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                            j += 1
                            break
                        else:
                            j += 1
                    else:
                        j += 1
                i += 1

            return matched_pairs



        if 'qwen' in self.path or 'Qwen' in self.path:
            sep_len = 2
        elif 'Llama-3' in self.path:
            sep_len = 2
        elif 'Yi' in self.path:
            sep_len = 3
        elif 'Llama-2' in self.path:
            sep_len = 3
        elif 'deepseek' in self.path:
            sep_len = 2
        else:
            sep_len = 3


        matched_pairs_list = [align_sequences(decode_messages[i], char_messages[i], sep_len) for i in range(len(decode_messages))]
        for match_index, matched_pairs in enumerate(matched_pairs_list):
            for i, (word, indices) in enumerate(matched_pairs):
                for j in indices:
                    if j in aggregated_label_list[match_index]:
                        tag_list[match_index].append(i)
                        break

        inference_loss_list = []
        token_len_list = []
        for i in range(len(loss)):
            inference_loss = 0
            token_len = 0
            for j in range(len(loss[i])):
                if j in tag_list[i]:

                    inference_loss += loss[i][j]
                    print(loss[i][j])
                    token_len += 1
            inference_loss_list.append(inference_loss)
            token_len_list.append(token_len)

        return inference_loss_list, token_len_list

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        wrapper_started = time.perf_counter()
        benchmark_enabled = getattr(self, '_hf_benchmark_enabled', False)
        if benchmark_enabled:
            self._hf_benchmark_generate_attempts += 1
        try:
            messages = _convert_chat_messages(inputs)
            batch_size = len(messages)

            tokenize_kwargs = dict(
                return_tensors='pt',
                padding=True,
                # ``mid`` mode performs its own exact boundary-preserving
                # truncation below. Applying tokenizer truncation first would
                # irreversibly discard the beginning of an overlong LongBench
                # prompt because these tokenizers use left truncation.
                truncation=self.mode not in {'mid', 'strict'},
                add_special_tokens=True,
                max_length=self.max_seq_len
            )
            if self.fastchat_template:
                messages = _format_with_fast_chat_template(
                    messages, self.fastchat_template)
                tokens = self.tokenizer(messages, **tokenize_kwargs)
            else:
                messages = [
                    self.tokenizer.apply_chat_template(
                        m,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.chat_template_kwargs,
                    )
                    for m in messages
                ]
                tokenize_kwargs['add_special_tokens'] = False
                tokens = self.tokenizer(messages, **tokenize_kwargs)

            if self.mode == 'mid':
                # Reserve space for the tokens to be generated in the future.
                max_prompt_len = self.max_seq_len - max_out_len
                tokens = _middle_truncate_token_fields(tokens, max_prompt_len)
            elif self.mode == 'strict':
                # Reserve the complete task-specific output allowance and
                # reject overlong prompts. No input token is ever discarded.
                max_prompt_len = self.max_seq_len - max_out_len
                _validate_strict_no_truncation_budget(tokens, max_prompt_len)

            if benchmark_enabled:
                attention_mask = tokens.get('attention_mask')
                prompt_tokens = (
                    int(attention_mask.sum().item())
                    if attention_mask is not None
                    else int(tokens['input_ids'].numel()))
                self._hf_benchmark_prompt_tokens += prompt_tokens
                self._hf_benchmark_padded_prompt_slots += int(
                    tokens['input_ids'].numel())

            tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

            generation_kwargs = self.generation_kwargs.copy()
            generation_kwargs.update(kwargs)
            stopping_criteria = list(set(stopping_criteria + self.stop_words))
            if stopping_criteria:
                generation_kwargs['stopping_criteria'] = _get_stopping_criteria(
                    stopping_criteria, self.tokenizer, batch_size)
            if max_out_len is not None:
                generation_kwargs['max_new_tokens'] = max_out_len
            if min_out_len is not None:
                generation_kwargs['min_new_tokens'] = min_out_len
            generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
            self.logger.info('Generation Args of Huggingface: ')
            self.logger.info(generation_kwargs)

            # step-2: conduct model forward to generate output
            if benchmark_enabled:
                _synchronize_model_accelerators(self.model)
                model_started = time.perf_counter()
            try:
                outputs = self.model.generate(**tokens, **generation_kwargs)
            finally:
                if benchmark_enabled:
                    _synchronize_model_accelerators(self.model)
                    self._hf_benchmark_model_seconds += (
                        time.perf_counter() - model_started)
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

            if benchmark_enabled:
                counts = _terminal_aware_output_counts(
                    outputs,
                    self._hf_eos_token_ids(generation_kwargs))
                self._hf_benchmark_successful_generate_calls += 1
                self._hf_benchmark_request_count += batch_size
                self._hf_benchmark_output_tokens += counts['output_tokens']
                self._hf_benchmark_generated_sequence_slots += counts[
                    'generated_sequence_slots']
                self._hf_benchmark_rows_with_eos += counts['rows_with_eos']
                self._hf_benchmark_rows_without_eos += counts[
                    'rows_without_eos']
                batch_key = str(batch_size)
                self._hf_benchmark_batch_histogram[batch_key] = (
                    self._hf_benchmark_batch_histogram.get(batch_key, 0) + 1)

            # step-3: decode the output
            decodeds = self.tokenizer.batch_decode(outputs)
            for stop in stopping_criteria:
                decodeds = [t.split(stop)[0] for t in decodeds]

            return decodeds
        except Exception:
            if benchmark_enabled:
                self._hf_benchmark_errors += 1
            raise
        finally:
            if benchmark_enabled:
                self._hf_benchmark_wrapper_seconds += (
                    time.perf_counter() - wrapper_started)
                self.flush_hf_benchmark_summary()

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(
            m,
            add_generation_prompt=True,
            return_dict=True,
            **self.chat_template_kwargs,
        )
        return len(t['input_ids'])

def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs


class ODABaseModel(ODAChatModel):

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 stop_words: Optional[str] = [],
                 drop_middle: bool = False,
                 **other_kwargs):

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = LMTemplateParser()
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self.drop_middle = drop_middle
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.stop_words = stop_words

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        messages = _convert_base_messages(inputs)
        batch_size = len(messages)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )

        if self.drop_middle:
            assert len(inputs) == 1
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.max_seq_len:
                input_ids = torch.cat([input_ids[:, : self.max_seq_len // 2], input_ids[:, - self.max_seq_len // 2:]], dim=-1)
            tokens = {'input_ids': input_ids, }
        else:
            tokens = self.tokenizer(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(stopping_criteria, self.tokenizer, batch_size)
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens, **generation_kwargs)
        outputs = outputs[:, tokens['input_ids'].shape[1]:]

        # step-3: decode the output
        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for stop in stopping_criteria:
            decodeds = [token.split(stop)[0] for token in decodeds]

        return decodeds

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )

        if self.drop_middle:
            assert len(inputs) == 1
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.max_seq_len:
                input_ids = torch.cat([input_ids[:, : self.max_seq_len // 2], input_ids[:, - self.max_seq_len // 2:]], dim=-1)
            tokens = {'input_ids': input_ids, }
        else:
            tokens = self.tokenizer(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            import numpy as np
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(self, inputs: List[str], conts:  List[str]) -> List[float]:
        mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
        return - self.get_ppl(inputs, mask_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool=True) -> int:
        m = _convert_base_messages([prompt])[0]
        t = self.tokenizer(m, add_special_tokens=add_special_tokens)
        return len(t['input_ids'])
