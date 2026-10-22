# Modified for On Demand Attention; see NOTICE.
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    replace_return_docstrings,
)

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import slice_position_embedding
from ....ops.loss import causallm_loss_function
from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available, is_torch_npu_available
from ...module_utils import GradientCheckpointingLayer


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    from transformers.integrations.flex_attention import make_flex_block_causal_mask

    raw_flex_attention = flex_attention
else:
    raw_flex_attention = None

compiled_flex_attention = None


if is_liger_kernel_available():
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "Qwen3Config"


def rms_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return weight * hidden_states.to(input_dtype)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if (
            config.sliding_window and config._attn_implementation != "flash_attention_2"
        ):  # diff with Llama is this warning
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # Transformers 5 stores the native RoPE settings in rope_parameters.
        if getattr(config, "rope_parameters", None) is not None:
            self.rope_type = config.rope_parameters["rope_type"]
        elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default" and "default" not in ROPE_INIT_FUNCTIONS:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as NativeRotaryEmbedding
            self.rope_init_fn = NativeRotaryEmbedding.compute_default_rope_parameters
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


QWEN3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # --- slice position embedding if using sp ---
        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
        position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
        # --- slice position embedding if using sp ---

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            if config.get_text_config().sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - config.get_text_config().sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs): ...


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_function = causallm_loss_function

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        loss = None
        logits = None
        loss, logits = self.loss_function(hidden_states, self.lm_head.weight, labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


_ROUTER_CONFIG_DEFAULTS = {
    "router_sink_size": 4,
    "router_window_size": 2048,
    "router_hidden_size": 256, # Hidden width of the router MLP.
    # Keep the original single-hidden-layer MLP as the checkpoint-compatible
    # default.  Capacity experiments opt into the interaction/residual head.
    "router_head_architecture": "shallow_mlp",
    "router_num_blocks": 2,
    "router_intermediate_size": 512,
    # Stabilized three-tower ablations are opt-in.  Keeping these defaults
    # disabled preserves the exact state schema and forward function of the
    # existing schema-v2 checkpoints.
    "router_group_rms_enabled": False,
    "router_group_rms_eps": 1.0e-6,
    "router_learned_fusion_scales": False,
    "router_fusion_scale_init": 0.5,
    "router_layer_scale_enabled": False,
    "router_layer_scale_init": 0.1,
    "router_kl_chunk_size": 16, # Chunk size for router KL divergence.
    "router_distill_temperature": 1.0, # Router distillation temperature.
    "router_kl_threshold": 0.05, # Router KL divergence threshold.
    "router_target_temperature": 0.02, # Router target temperature.
    "router_use_soft_targets": False, # Whether to use soft targets.
    "router_auto_positive_weight": True, # Whether to compute positive-class weight automatically.
    "router_positive_weight": 1.0, # Positive-class weight for the router.
    "router_negative_weight": 1.0, # Negative-class weight for the router.
    "router_loss_weight": 1.0, # Router loss weight.
    "router_target_only": False, # Whether to use target tokens only.
    "router_return_hidden_states": False, # Whether to return router hidden states.
    "router_check_finite": False, # Whether to check router outputs for finite values.
    # Iteration-2 options are opt-in so format-v1 checkpoints keep the exact
    # BCE population and interpretation they were trained with.
    "router_eligible_only": False,
    "router_objective": "weighted_bce",
    "router_log_kl_epsilon": 1.0e-5,
    "router_huber_delta": 1.0,
    "router_tail_fraction": 0.20,
    "router_tail_weight": 5.0,
    "router_listwise_temperature": 1.0,
    "router_hybrid_alpha": 0.5,
    "router_return_token_nll": False,
    # The schema is deliberately explicit so later feature ablations can be
    # expressed in config instead of creating another router implementation.
    # `full_history` is the previous token's Always-Full final hidden state.
    "router_feature_schema": ["full_history"],
}

_ROUTER_FEATURE_NAMES = {
    "full_history", # Historical hidden state from the Full branch.
    "current_embedding", # Embedding of the current token.
    "history_state", # generic fixed-trunk/selected history state
    "current_swa_hidden", # current token final hidden after the SWA candidate
    "normalized_log_context", # Normalized log context length.
    "local_coverage", # Fraction covered by Local attention.
}

# Return the compiled FlexAttention function, compiling it on first use.
def _get_compiled_flex_attention() -> Callable:
    global compiled_flex_attention
    if raw_flex_attention is None:
        raise RuntimeError("Qwen3 Router requires PyTorch FlexAttention.")
    if compiled_flex_attention is None:
        compiled_flex_attention = torch.compile(raw_flex_attention)  # Compile once for all layers.
    return compiled_flex_attention

# Fill in missing router defaults and validate parameter ranges and feature names.
def _set_router_config_defaults(config: Qwen3Config) -> None:
    for name, value in _ROUTER_CONFIG_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, value)

    if config.router_sink_size < 0:
        raise ValueError("`router_sink_size` must be non-negative.")
    if config.router_window_size < 1:
        raise ValueError("`router_window_size` includes the current token and must be at least 1.")
    if config.router_hidden_size < 1:
        raise ValueError("`router_hidden_size` must be positive.")
    valid_head_architectures = {
        "shallow_mlp",
        "interaction_swiglu",
        "three_tower_interaction_swiglu",
        "three_tower_interaction_swiglu_stabilized",
    }
    if config.router_head_architecture not in valid_head_architectures:
        raise ValueError(
            f"Unknown `router_head_architecture` {config.router_head_architecture!r}; "
            f"expected one of {sorted(valid_head_architectures)}."
        )
    if int(config.router_num_blocks) < 1:
        raise ValueError("`router_num_blocks` must be positive.")
    if int(config.router_intermediate_size) < 1:
        raise ValueError("`router_intermediate_size` must be positive.")
    if float(config.router_group_rms_eps) <= 0:
        raise ValueError("`router_group_rms_eps` must be positive.")
    if not math.isfinite(float(config.router_fusion_scale_init)):
        raise ValueError("`router_fusion_scale_init` must be finite.")
    if not math.isfinite(float(config.router_layer_scale_init)):
        raise ValueError("`router_layer_scale_init` must be finite.")
    if config.router_kl_chunk_size < 1:
        raise ValueError("`router_kl_chunk_size` must be positive.")
    if config.router_distill_temperature <= 0:
        raise ValueError("`router_distill_temperature` must be positive.")
    if config.router_target_temperature <= 0:
        raise ValueError("`router_target_temperature` must be positive.")
    if config.router_positive_weight < 0 or config.router_negative_weight < 0:
        raise ValueError("Router BCE weights must be non-negative.")
    valid_objectives = {
        "weighted_bce",
        "tail_log_kl_huber",
        "kl_mass_listwise",
        "hybrid_tail_listwise",
    }
    if config.router_objective not in valid_objectives:
        raise ValueError(
            f"Unknown `router_objective` {config.router_objective!r}; "
            f"expected one of {sorted(valid_objectives)}."
        )
    if config.router_log_kl_epsilon <= 0:
        raise ValueError("`router_log_kl_epsilon` must be positive.")
    if config.router_huber_delta <= 0:
        raise ValueError("`router_huber_delta` must be positive.")
    if not 0 < config.router_tail_fraction <= 1:
        raise ValueError("`router_tail_fraction` must be in (0, 1].")
    if config.router_tail_weight <= 0:
        raise ValueError("`router_tail_weight` must be positive.")
    if config.router_listwise_temperature <= 0:
        raise ValueError("`router_listwise_temperature` must be positive.")
    if not 0 <= config.router_hybrid_alpha <= 1:
        raise ValueError("`router_hybrid_alpha` must be in [0, 1].")
    if config.attention_dropout != 0:
        raise ValueError("Qwen3 Router oracle collection requires `attention_dropout=0`.")

    feature_schema = config.router_feature_schema
    if isinstance(feature_schema, str):
        feature_schema = [name.strip() for name in feature_schema.split(",") if name.strip()]
    else:
        feature_schema = list(feature_schema)
    if not feature_schema:
        raise ValueError("`router_feature_schema` must contain at least one feature.")
    unknown_features = set(feature_schema).difference(_ROUTER_FEATURE_NAMES)
    if unknown_features:
        raise ValueError(f"Unknown Qwen3 Router feature(s): {sorted(unknown_features)}.")
    if len(feature_schema) != len(set(feature_schema)):
        raise ValueError("`router_feature_schema` cannot contain duplicate features.")
    if config.router_head_architecture == "interaction_swiglu":
        required_features = {"full_history", "current_embedding"}
        missing_features = required_features.difference(feature_schema)
        if missing_features:
            raise ValueError(
                "`interaction_swiglu` requires causal history/current interaction features; "
                f"missing {sorted(missing_features)}."
            )
    if config.router_head_architecture in {
        "three_tower_interaction_swiglu",
        "three_tower_interaction_swiglu_stabilized",
    }:
        expected_features = ["history_state", "current_embedding", "current_swa_hidden"]
        if feature_schema != expected_features:
            raise ValueError(
                "The three-tower Router requires the exact ordered feature schema "
                f"{expected_features}, got {feature_schema}."
            )
    config.router_feature_schema = feature_schema


def build_router_eligible_mask(
    valid_token_mask: torch.BoolTensor,
    labels: torch.LongTensor,
    position_ids: torch.LongTensor,
    sink_size: int,
    window_size: int,
) -> torch.BoolTensor:
    """Return the Iteration-2 population aligned with prediction of ``labels[i + 1]``.

    The next token must be valid, labelled, and in the same packed segment.
    ``position_ids`` are zero based, hence the first eligible context length is
    ``sink_size + window_size + 1`` (2053 for sink-4/window-2048).
    """

    if labels.shape != valid_token_mask.shape or position_ids.shape != valid_token_mask.shape:
        raise ValueError("Router eligibility inputs must have identical [batch, sequence] shapes.")
    if sink_size < 0 or window_size < 1:
        raise ValueError("Router eligibility requires sink_size >= 0 and window_size >= 1.")

    has_aligned_target = torch.zeros_like(valid_token_mask)
    if labels.shape[1] > 1:
        has_aligned_target[:, :-1] = (
            valid_token_mask[:, 1:]
            & labels[:, 1:].ne(-100)
            & position_ids[:, 1:].eq(position_ids[:, :-1] + 1)
        )
    structurally_eligible = position_ids.add(1).gt(int(sink_size) + int(window_size))
    return valid_token_mask & has_aligned_target & structurally_eligible

# Return feature widths in the order specified by router_feature_schema.
def _router_feature_dimensions(config: Qwen3Config) -> Tuple[int, ...]:
    dimensions = {
        "full_history": config.hidden_size,
        "current_embedding": config.hidden_size,
        "history_state": config.hidden_size,
        "current_swa_hidden": config.hidden_size,
        "normalized_log_context": 1,
        "local_coverage": 1,
    }
    return tuple(dimensions[name] for name in config.router_feature_schema)

# Compute the duplicate-free union coverage of sink and recent tokens.
def _exact_local_coverage(
    context_length: torch.Tensor,
    sink_size: int,
    window_size: int,
) -> torch.Tensor:
    """Fraction covered by the duplicate-free union of sink and recent tokens."""

    context_length = context_length.float().clamp_min(1.0)
    recent = context_length.clamp_max(float(window_size))
    sink_outside_recent = (context_length - float(window_size)).clamp(
        min=0.0, max=float(sink_size)
    )
    return (recent + sink_outside_recent).clamp_max(context_length) / context_length

# Check whether a position has past context available to the router.
def _router_has_past(
    position_ids: torch.LongTensor,
    valid_token_mask: torch.BoolTensor,
    sink_size: int,
    window_size: int,
) -> torch.BoolTensor:
    can_read_any_past = sink_size > 0 or window_size > 1
    return valid_token_mask & position_ids.gt(0) & can_read_any_past


def _check_router_tensor_is_finite(name: str, tensor: torch.Tensor) -> None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return

    finite_values = tensor[finite_mask]
    finite_abs_max = finite_values.abs().max().item() if finite_values.numel() else float("nan")
    raise FloatingPointError(
        f"Non-finite Qwen3 Router tensor `{name}`: "
        f"nan={torch.isnan(tensor).sum().item()}, "
        f"posinf={torch.isposinf(tensor).sum().item()}, "
        f"neginf={torch.isneginf(tensor).sum().item()}, "
        f"finite_abs_max={finite_abs_max}."
    )


def create_router_past_block_mask(
    position_ids: torch.LongTensor,
    valid_token_mask: torch.BoolTensor,
    sink_size: int,
    window_size: int,
) -> "BlockMask":
    """Build the past-only sink-plus-window mask used by every router layer."""
    if raw_flex_attention is None:
        raise RuntimeError("Qwen3 Router requires PyTorch FlexAttention.")
    if position_ids.ndim != 2 or valid_token_mask.ndim != 2:
        raise ValueError("Router masks require 2-D position IDs and validity masks.")
    if position_ids.shape != valid_token_mask.shape:
        raise ValueError("Position IDs and validity masks must have identical shapes.")

    batch_size, sequence_length = position_ids.shape
    # Every position-id reset starts a new packed sequence. Invalid left-padding
    # zeros may create extra segments, but they are excluded by valid_token_mask.
    segment_ids = position_ids.eq(0).to(torch.int64).cumsum(dim=-1)
    can_read_any_past = sink_size > 0 or window_size > 1

    def past_only_mask(b, h, q_idx, kv_idx):
        del h
        q_position = position_ids[b, q_idx]
        kv_position = position_ids[b, kv_idx]
        q_valid = valid_token_mask[b, q_idx]
        kv_valid = valid_token_mask[b, kv_idx]
        same_segment = segment_ids[b, q_idx] == segment_ids[b, kv_idx]

        is_past = kv_position < q_position
        is_sink = kv_position < sink_size
        is_recent = kv_position >= q_position - window_size + 1
        allowed_past = q_valid & kv_valid & same_segment & is_past & (is_sink | is_recent)

        # FlexAttention must never receive an entirely unsafe row. The dummy
        # diagonal is used only when the mathematical past set is empty and is
        # removed before the online-softmax self merge.
        has_past = q_valid & (q_position > 0) & can_read_any_past
        safe_dummy = (~has_past) & (q_idx == kv_idx)
        return allowed_past | safe_dummy

    return create_block_mask(
        past_only_mask,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=position_ids.device,
    )

# Merge past attention and current-token Local attention with online softmax.
def online_softmax_merge_local_self(
    past_output: torch.Tensor,
    past_lse: torch.Tensor,
    self_score: torch.Tensor,
    self_value: torch.Tensor,
    empty_past_mask: torch.BoolTensor,
    valid_token_mask: torch.BoolTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge past-only attention with the local current-token KV in FP32."""
    output_dtype = past_output.dtype
    empty = empty_past_mask[:, None, :]

    past_output_fp32 = past_output.float().masked_fill(empty[..., None], 0.0)
    past_lse_fp32 = past_lse.float().masked_fill(empty, float("-inf"))
    self_score_fp32 = self_score.float()
    self_value_fp32 = self_value.float()

    merged_lse = torch.logaddexp(past_lse_fp32, self_score_fp32)
    past_weight = torch.exp(past_lse_fp32 - merged_lse)
    self_weight = torch.exp(self_score_fp32 - merged_lse)
    merged_output = past_weight[..., None] * past_output_fp32 + self_weight[..., None] * self_value_fp32
    merged_output = merged_output.masked_fill(~valid_token_mask[:, None, :, None], 0.0)
    return merged_output.to(output_dtype), merged_lse

# Router training output omits full-vocabulary logits to reduce memory use.
@dataclass
class Qwen3RouterCausalLMOutput(ModelOutput):
    """Output of router-oracle training; no full-vocabulary logits are retained."""

    loss: Optional[torch.FloatTensor] = None
    router_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[torch.FloatTensor] = None
    router_targets: Optional[torch.FloatTensor] = None
    router_divergence: Optional[torch.FloatTensor] = None
    full_token_nll: Optional[torch.FloatTensor] = None
    local_token_nll: Optional[torch.FloatTensor] = None
    router_target_rate: Optional[torch.FloatTensor] = None
    router_prediction_rate_at_0_5: Optional[torch.FloatTensor] = None
    router_mean_divergence: Optional[torch.FloatTensor] = None
    router_effective_positive_weight: Optional[torch.FloatTensor] = None
    router_effective_hard_divergence: Optional[torch.FloatTensor] = None
    router_effective_soft_divergence: Optional[torch.FloatTensor] = None
    router_local_mean_divergence: Optional[torch.FloatTensor] = None
    router_accuracy: Optional[torch.FloatTensor] = None
    router_precision: Optional[torch.FloatTensor] = None
    router_recall: Optional[torch.FloatTensor] = None
    router_f1: Optional[torch.FloatTensor] = None
    training_metrics: Optional[dict] = None
    training_probe_tensors: Optional[dict] = None
    full_hidden_states: Optional[torch.FloatTensor] = None
    local_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3FullLocalCausalLMOutput(ModelOutput):
    """Base-model Full/Local counterfactuals without Router predictions."""

    full_to_local_kl: Optional[torch.FloatTensor] = None
    local_to_full_kl: Optional[torch.FloatTensor] = None
    js_divergence: Optional[torch.FloatTensor] = None
    full_token_nll: Optional[torch.FloatTensor] = None
    local_token_nll: Optional[torch.FloatTensor] = None
    all_full_token_nll: Optional[torch.FloatTensor] = None
    all_local_token_nll: Optional[torch.FloatTensor] = None
    full_top1_token_id: Optional[torch.LongTensor] = None
    local_top1_token_id: Optional[torch.LongTensor] = None
    full_top1_probability: Optional[torch.FloatTensor] = None
    local_top1_probability: Optional[torch.FloatTensor] = None
    full_top1_margin: Optional[torch.FloatTensor] = None
    local_top1_margin: Optional[torch.FloatTensor] = None
    full_entropy: Optional[torch.FloatTensor] = None
    local_entropy: Optional[torch.FloatTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    aligned_target_mask: Optional[torch.BoolTensor] = None
    structural_aligned_target_mask: Optional[torch.BoolTensor] = None
    eligible_token_mask: Optional[torch.BoolTensor] = None
    structural_eligible_token_mask: Optional[torch.BoolTensor] = None

# Full attention matches the original Qwen3 operations; Local uses Full history KV and current Local KV.
class Qwen3RouterAttention(Qwen3Attention):
    """Qwen3 attention exposing Full KV and the Local shadow computation."""

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states

    def full_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attention_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        batch_size, _, sequence_length, _ = query_states.shape
        attention_output = attention_output.reshape(batch_size, sequence_length, -1).contiguous()
        return self.o_proj(attention_output)

    def local_shadow_attention(
        self,
        local_query_states: torch.Tensor,
        full_key_states: torch.Tensor,
        full_value_states: torch.Tensor,
        local_key_states: torch.Tensor,
        local_value_states: torch.Tensor,
        past_block_mask: "BlockMask",
        position_ids: torch.LongTensor,
        valid_token_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        run_flex_attention = _get_compiled_flex_attention()
        past_output, past_lse = run_flex_attention(
            local_query_states,
            full_key_states,
            full_value_states,
            block_mask=past_block_mask,
            scale=self.scaling,
            enable_gqa=self.num_key_value_groups > 1,
            return_lse=True,
            kernel_options={"ROWS_GUARANTEED_SAFE": False},
        )
        if self.config.router_check_finite:
            _check_router_tensor_is_finite(f"layer_{self.layer_idx}.flex_past_output", past_output)
            _check_router_tensor_is_finite(f"layer_{self.layer_idx}.flex_past_lse", past_lse)

        local_key_by_query_head = repeat_kv(local_key_states, self.num_key_value_groups)
        local_value_by_query_head = repeat_kv(local_value_states, self.num_key_value_groups)
        self_score = (local_query_states.float() * local_key_by_query_head.float()).sum(dim=-1) * self.scaling
        has_past = _router_has_past(
            position_ids,
            valid_token_mask,
            self.config.router_sink_size,
            self.config.router_window_size,
        )
        merged_output, merged_lse = online_softmax_merge_local_self(
            past_output,
            past_lse,
            self_score,
            local_value_by_query_head,
            empty_past_mask=~has_past,
            valid_token_mask=valid_token_mask,
        )
        if self.config.router_check_finite:
            _check_router_tensor_is_finite(f"layer_{self.layer_idx}.local_self_score", self_score)
            _check_router_tensor_is_finite(f"layer_{self.layer_idx}.merged_output", merged_output)
            _check_router_tensor_is_finite(f"layer_{self.layer_idx}.merged_lse", merged_lse)
        batch_size, _, sequence_length, _ = merged_output.shape
        merged_output = merged_output.transpose(1, 2).reshape(batch_size, sequence_length, -1).contiguous()
        return self.o_proj(merged_output), merged_lse

# Dual-branch decoder layer shares weights while advancing Full and Local residual/MLP states separately.
class Qwen3RouterDecoderLayer(nn.Module):
    """Advance the Always-Full trunk and all one-step Local shadows together."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3RouterAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        local_hidden_states: torch.Tensor,
        full_attention_mask: Optional[torch.Tensor],
        past_block_mask: "BlockMask",
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        valid_token_mask: torch.BoolTensor,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full_residual = full_hidden_states
        local_residual = local_hidden_states
        full_normalized = self.input_layernorm(full_hidden_states)
        local_normalized = self.input_layernorm(local_hidden_states)

        full_query, full_key, full_value = self.self_attn.project_qkv(full_normalized, position_embeddings)
        local_query, local_key, local_value = self.self_attn.project_qkv(local_normalized, position_embeddings)
        full_attention_output = self.self_attn.full_attention(
            full_query,
            full_key,
            full_value,
            full_attention_mask,
            **kwargs,
        )
        local_attention_output, _ = self.self_attn.local_shadow_attention(
            local_query,
            full_key,
            full_value,
            local_key,
            local_value,
            past_block_mask,
            position_ids,
            valid_token_mask,
        )
        full_hidden_states = full_residual + full_attention_output
        local_hidden_states = local_residual + local_attention_output

        full_residual = full_hidden_states
        local_residual = local_hidden_states
        full_hidden_states = full_residual + self.mlp(self.post_attention_layernorm(full_hidden_states))
        local_hidden_states = local_residual + self.mlp(self.post_attention_layernorm(local_hidden_states))
        return full_hidden_states, local_hidden_states

# The router backbone requires cache, gradient checkpointing, and Ulysses to be disabled.
class Qwen3RouterPreTrainedModel(Qwen3PreTrainedModel):
    _no_split_modules = ["Qwen3RouterDecoderLayer"]
    supports_gradient_checkpointing = False
    _supports_cache_class = False
    _supports_static_cache = False

# The router backbone requires cache, gradient checkpointing, and Ulysses to be disabled.
class Qwen3RouterModel(Qwen3RouterPreTrainedModel):
    """A physical Qwen3 copy with an interleaved Full/Local shadow stream."""

    _update_causal_mask = Qwen3Model._update_causal_mask
    _prepare_4d_causal_attention_mask_with_cache_position = staticmethod(
        Qwen3Model._prepare_4d_causal_attention_mask_with_cache_position
    )

    def __init__(self, config: Qwen3Config):
        _set_router_config_defaults(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3RouterDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        if raw_flex_attention is None:
            raise RuntimeError("Qwen3 Router requires PyTorch FlexAttention.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` and `inputs_embeds`.")
        if past_key_values is not None or use_cache:
            raise ValueError("Qwen3 Router oracle training does not support KV-cache decoding.")
        if output_attentions or output_hidden_states:
            raise ValueError("Qwen3 Router oracle training does not retain attention or per-layer hidden states.")
        if get_parallel_state().sp_enabled:
            raise ValueError("Qwen3 Router currently requires `ulysses_parallel_size=1`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, sequence_length, _ = inputs_embeds.shape

        if attention_mask is None:
            valid_token_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=inputs_embeds.device)
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("Qwen3 Router requires a dense 2-D attention mask.")
            valid_token_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

        if cache_position is None:
            cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)
        if position_ids.shape != (batch_size, sequence_length):
            raise ValueError("Qwen3 Router position IDs must match the dense token matrix.")
        position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)

        full_attention_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values=None,
            output_attentions=False,
        )
        past_block_mask = create_router_past_block_mask(
            position_ids,
            valid_token_mask,
            sink_size=self.config.router_sink_size,
            window_size=self.config.router_window_size,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        full_hidden_states = inputs_embeds
        local_hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            full_hidden_states, local_hidden_states = decoder_layer(
                full_hidden_states,
                local_hidden_states,
                full_attention_mask=full_attention_mask,
                past_block_mask=past_block_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                valid_token_mask=valid_token_mask,
                **flash_attn_kwargs,
            )
            if self.config.router_check_finite:
                _check_router_tensor_is_finite(f"layer_{layer_idx}.full_hidden_states", full_hidden_states)
                _check_router_tensor_is_finite(f"layer_{layer_idx}.local_hidden_states", local_hidden_states)

        full_hidden_states = self.norm(full_hidden_states)
        local_hidden_states = self.norm(local_hidden_states)
        if self.config.router_check_finite:
            _check_router_tensor_is_finite("full_final_hidden_states", full_hidden_states)
            _check_router_tensor_is_finite("local_final_hidden_states", local_hidden_states)
        return full_hidden_states, local_hidden_states, inputs_embeds, position_ids, valid_token_mask

# FP32 router MLP head: normalize each feature, concatenate, then apply SiLU and a score projection.
class Qwen3RouterHead(nn.Module):
    """Predict whether Full attention is needed before the current action."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.feature_schema = tuple(config.router_feature_schema)
        self.feature_dimensions = _router_feature_dimensions(config)
        input_size = sum(self.feature_dimensions)
        self.hidden_size = config.hidden_size
        if "full_history" in self.feature_schema:
            self.history_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if "current_embedding" in self.feature_schema:
            self.embedding_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_projection = nn.Linear(input_size, config.router_hidden_size, bias=True)
        self.output_projection = nn.Linear(config.router_hidden_size, 1, bias=True)

    def forward(self, router_features: torch.Tensor) -> torch.Tensor:
        router_features = router_features.float()
        components = torch.split(router_features, self.feature_dimensions, dim=-1)
        normalized_components = []
        for name, component in zip(self.feature_schema, components):
            if name == "full_history":
                component = self.history_norm(component)
            elif name == "current_embedding":
                component = self.embedding_norm(component)
            normalized_components.append(component)
        router_features = torch.cat(normalized_components, dim=-1)
        router_features = nn.functional.silu(self.input_projection(router_features))
        return self.output_projection(router_features).squeeze(-1)


class Qwen3RouterSwiGLUBlock(nn.Module):
    """Pre-norm residual SwiGLU block used only by the larger Router head."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)
        if layer_scale_init is None:
            self.register_parameter("layer_scale", None)
        else:
            self.layer_scale = nn.Parameter(
                torch.tensor(float(layer_scale_init), dtype=torch.float32)
            )

    def forward_with_update(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(hidden_states)
        update = self.down_projection(
            nn.functional.silu(self.gate_projection(normalized))
            * self.up_projection(normalized)
        )
        if self.layer_scale is not None:
            update = update * self.layer_scale
        return hidden_states + update, update

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_update(hidden_states)
        return output


def _parameter_free_unit_rms(
    value: torch.Tensor, eps: float
) -> torch.Tensor:
    """Normalize one interaction group without introducing parameters."""

    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


def _token_rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().square().mean(dim=-1).sqrt()


def _token_cosine(
    left: torch.Tensor, right: torch.Tensor, eps: float
) -> torch.Tensor:
    return nn.functional.cosine_similarity(
        left.detach().float(), right.detach().float(), dim=-1, eps=eps
    )


class Qwen3InteractionRouterHead(nn.Module):
    """Two-tower interaction encoder followed by residual SwiGLU blocks.

    History and current-token embedding have different semantics, so each gets
    its own projection tower.  Elementwise product and absolute difference add
    explicit matching signals that a shallow MLP over a concatenation would
    otherwise need substantial width to approximate.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.feature_schema = tuple(config.router_feature_schema)
        self.feature_dimensions = _router_feature_dimensions(config)
        self.router_hidden_size = int(config.router_hidden_size)
        self.auxiliary_size = sum(
            dimension
            for name, dimension in zip(self.feature_schema, self.feature_dimensions)
            if name not in {"full_history", "current_embedding"}
        )

        self.history_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.history_projection = nn.Linear(
            config.hidden_size, self.router_hidden_size, bias=True
        )
        self.embedding_projection = nn.Linear(
            config.hidden_size, self.router_hidden_size, bias=True
        )
        self.fusion_projection = nn.Linear(
            4 * self.router_hidden_size + self.auxiliary_size,
            self.router_hidden_size,
            bias=True,
        )
        self.blocks = nn.ModuleList(
            Qwen3RouterSwiGLUBlock(
                self.router_hidden_size,
                int(config.router_intermediate_size),
                float(config.rms_norm_eps),
            )
            for _ in range(int(config.router_num_blocks))
        )
        self.output_norm = Qwen3RMSNorm(
            self.router_hidden_size, eps=config.rms_norm_eps
        )
        self.output_projection = nn.Linear(self.router_hidden_size, 1, bias=True)

    def forward(self, router_features: torch.Tensor) -> torch.Tensor:
        router_features = router_features.float()
        components = torch.split(router_features, self.feature_dimensions, dim=-1)
        feature_values = dict(zip(self.feature_schema, components))

        history = nn.functional.silu(
            self.history_projection(self.history_norm(feature_values["full_history"]))
        )
        embedding = nn.functional.silu(
            self.embedding_projection(
                self.embedding_norm(feature_values["current_embedding"])
            )
        )
        fusion_components = [
            history,
            embedding,
            history * embedding,
            torch.abs(history - embedding),
        ]
        auxiliary = [
            component
            for name, component in zip(self.feature_schema, components)
            if name not in {"full_history", "current_embedding"}
        ]
        if auxiliary:
            fusion_components.append(torch.cat(auxiliary, dim=-1))

        fused = nn.functional.silu(
            self.fusion_projection(torch.cat(fusion_components, dim=-1))
        )
        hidden_states = (history + embedding + fused) / math.sqrt(3.0)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_projection(self.output_norm(hidden_states)).squeeze(-1)


class Qwen3ThreeTowerInteractionRouterHead(nn.Module):
    """Three semantic towers with every pairwise matching interaction.

    The ordered inputs are fixed to history state, current token embedding,
    and the current token's final SWA-candidate hidden state.  Keeping the
    towers separate avoids treating three causally different representations
    as interchangeable slices of one flat vector.
    """

    expected_parameter_count_1_7b = 28_325_889

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.feature_schema = tuple(config.router_feature_schema)
        self.feature_dimensions = _router_feature_dimensions(config)
        expected_schema = ("history_state", "current_embedding", "current_swa_hidden")
        if self.feature_schema != expected_schema:
            raise ValueError(
                f"Three-tower Router requires {expected_schema}, got {self.feature_schema}."
            )

        hidden_size = int(config.hidden_size)
        router_size = int(config.router_hidden_size)
        self.router_hidden_size = router_size
        self.history_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.embedding_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.swa_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.history_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.embedding_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.swa_projection = nn.Linear(hidden_size, router_size, bias=True)

        # Three tower values plus product/absolute-difference for each of the
        # three unordered pairs: 3 + 2 * C(3, 2) = 9 components.
        self.fusion_projection = nn.Linear(9 * router_size, router_size, bias=True)
        self.blocks = nn.ModuleList(
            Qwen3RouterSwiGLUBlock(
                router_size,
                int(config.router_intermediate_size),
                float(config.rms_norm_eps),
            )
            for _ in range(int(config.router_num_blocks))
        )
        self.output_norm = Qwen3RMSNorm(router_size, eps=config.rms_norm_eps)
        self.output_projection = nn.Linear(router_size, 1, bias=True)

        from .recall_features import CONFIDENCE_NAMES, validate_recall_config
        self.recall_variant = validate_recall_config(config)
        if self.recall_variant is not None:
            self.confidence_projection = nn.Linear(len(CONFIDENCE_NAMES), router_size, bias=False)

    def forward(self, router_features: torch.Tensor) -> torch.Tensor:
        # The formal head is an FP32 island even when an outer trainer enables
        # autocast for the frozen backbone.
        with torch.autocast(device_type=router_features.device.type, enabled=False):
            values = router_features.float()
            confidence = None
            if self.recall_variant is not None:
                expected = sum(self.feature_dimensions) + self.confidence_projection.in_features
                if values.shape[-1] != expected:
                    raise ValueError("Recall head input dimension mismatch")
                values, confidence = values[..., :-6], values[..., -6:]
            components = torch.split(values, self.feature_dimensions, dim=-1)
            feature_values = dict(zip(self.feature_schema, components))
            history = nn.functional.silu(
                self.history_projection(self.history_norm(feature_values["history_state"]))
            )
            embedding = nn.functional.silu(
                self.embedding_projection(
                    self.embedding_norm(feature_values["current_embedding"])
                )
            )
            swa = nn.functional.silu(
                self.swa_projection(self.swa_norm(feature_values["current_swa_hidden"]))
            )

            fusion_components = [history, embedding, swa]
            for left, right in ((history, embedding), (history, swa), (embedding, swa)):
                fusion_components.extend((left * right, torch.abs(left - right)))
            fused = nn.functional.silu(
                self.fusion_projection(torch.cat(fusion_components, dim=-1))
            )
            hidden_states = (history + embedding + swa + fused) * 0.5
            if confidence is not None:
                hidden_states = hidden_states + self.confidence_projection(confidence)
            for block in self.blocks:
                hidden_states = block(hidden_states)
            return self.output_projection(self.output_norm(hidden_states)).squeeze(-1)


class Qwen3StabilizedThreeTowerInteractionRouterHead(nn.Module):
    """Cumulative three-tower stability ablations with an unchanged input API."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.feature_schema = tuple(config.router_feature_schema)
        self.feature_dimensions = _router_feature_dimensions(config)
        expected_schema = ("history_state", "current_embedding", "current_swa_hidden")
        if self.feature_schema != expected_schema:
            raise ValueError(
                f"Stabilized three-tower Router requires {expected_schema}, "
                f"got {self.feature_schema}."
            )

        hidden_size = int(config.hidden_size)
        router_size = int(config.router_hidden_size)
        self.router_hidden_size = router_size
        self.group_rms_enabled = bool(config.router_group_rms_enabled)
        self.group_rms_eps = float(config.router_group_rms_eps)
        self.learned_fusion_scales = bool(config.router_learned_fusion_scales)
        self.layer_scale_enabled = bool(config.router_layer_scale_enabled)

        self.history_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.embedding_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.swa_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.history_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.embedding_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.swa_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.fusion_projection = nn.Linear(9 * router_size, router_size, bias=True)
        if self.learned_fusion_scales:
            self.fusion_scales = nn.Parameter(
                torch.full(
                    (4,),
                    float(config.router_fusion_scale_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("fusion_scales", None)

        layer_scale_init = (
            float(config.router_layer_scale_init)
            if self.layer_scale_enabled
            else None
        )
        self.blocks = nn.ModuleList(
            Qwen3RouterSwiGLUBlock(
                router_size,
                int(config.router_intermediate_size),
                float(config.rms_norm_eps),
                layer_scale_init=layer_scale_init,
            )
            for _ in range(int(config.router_num_blocks))
        )
        self.output_norm = Qwen3RMSNorm(router_size, eps=config.rms_norm_eps)
        self.output_projection = nn.Linear(router_size, 1, bias=True)

    @staticmethod
    def _raw_fusion_components(
        history: torch.Tensor,
        embedding: torch.Tensor,
        swa: torch.Tensor,
    ) -> list[torch.Tensor]:
        values = [history, embedding, swa]
        for left, right in ((history, embedding), (history, swa), (embedding, swa)):
            values.extend((left * right, torch.abs(left - right)))
        return values

    def _forward_impl(
        self,
        router_features: torch.Tensor,
        collect_diagnostics: bool,
    ) -> Tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        with torch.autocast(device_type=router_features.device.type, enabled=False):
            components = torch.split(
                router_features.float(), self.feature_dimensions, dim=-1
            )
            feature_values = dict(zip(self.feature_schema, components))
            history = nn.functional.silu(
                self.history_projection(
                    self.history_norm(feature_values["history_state"])
                )
            )
            embedding = nn.functional.silu(
                self.embedding_projection(
                    self.embedding_norm(feature_values["current_embedding"])
                )
            )
            swa = nn.functional.silu(
                self.swa_projection(
                    self.swa_norm(feature_values["current_swa_hidden"])
                )
            )

            raw_groups = self._raw_fusion_components(history, embedding, swa)
            fusion_groups = (
                [
                    _parameter_free_unit_rms(value, self.group_rms_eps)
                    for value in raw_groups
                ]
                if self.group_rms_enabled
                else raw_groups
            )
            fused = nn.functional.silu(
                self.fusion_projection(torch.cat(fusion_groups, dim=-1))
            )
            if self.fusion_scales is None:
                hidden_states = (history + embedding + swa + fused) * 0.5
            else:
                hidden_states = (
                    self.fusion_scales[0] * history
                    + self.fusion_scales[1] * embedding
                    + self.fusion_scales[2] * swa
                    + self.fusion_scales[3] * fused
                )

            diagnostics = None
            if collect_diagnostics:
                diagnostics = {
                    "head_rms_history": _token_rms(history),
                    "head_rms_embedding": _token_rms(embedding),
                    "head_rms_swa": _token_rms(swa),
                    "head_rms_fused": _token_rms(fused),
                    "head_rms_residual_initial": _token_rms(hidden_states),
                    "head_cosine_history_embedding": _token_cosine(
                        history, embedding, self.group_rms_eps
                    ),
                    "head_cosine_history_swa": _token_cosine(
                        history, swa, self.group_rms_eps
                    ),
                    "head_cosine_embedding_swa": _token_cosine(
                        embedding, swa, self.group_rms_eps
                    ),
                    "head_cosine_history_fused": _token_cosine(
                        history, fused, self.group_rms_eps
                    ),
                    "head_cosine_embedding_fused": _token_cosine(
                        embedding, fused, self.group_rms_eps
                    ),
                    "head_cosine_swa_fused": _token_cosine(
                        swa, fused, self.group_rms_eps
                    ),
                }
                for index, (raw, normalized) in enumerate(
                    zip(raw_groups, fusion_groups)
                ):
                    diagnostics[f"head_group_{index}_pre_rms"] = _token_rms(raw)
                    diagnostics[f"head_group_{index}_post_rms"] = _token_rms(
                        normalized
                    )

            for index, block in enumerate(self.blocks):
                residual = hidden_states
                hidden_states, update = block.forward_with_update(hidden_states)
                if diagnostics is not None:
                    residual_rms = _token_rms(residual)
                    update_rms = _token_rms(update)
                    diagnostics[f"head_block_{index}_update_rms"] = update_rms
                    diagnostics[f"head_block_{index}_update_to_residual_rms"] = (
                        update_rms / residual_rms.clamp_min(self.group_rms_eps)
                    )

            score = self.output_projection(
                self.output_norm(hidden_states)
            ).squeeze(-1)
            return score, diagnostics

    def forward(self, router_features: torch.Tensor) -> torch.Tensor:
        score, _ = self._forward_impl(router_features, collect_diagnostics=False)
        return score

    def forward_with_diagnostics(
        self, router_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        score, diagnostics = self._forward_impl(
            router_features, collect_diagnostics=True
        )
        assert diagnostics is not None
        return score, diagnostics


def _build_router_head(config: Qwen3Config) -> nn.Module:
    if config.router_head_architecture == "shallow_mlp":
        return Qwen3RouterHead(config)
    if config.router_head_architecture == "interaction_swiglu":
        return Qwen3InteractionRouterHead(config)
    if config.router_head_architecture == "three_tower_interaction_swiglu":
        return Qwen3ThreeTowerInteractionRouterHead(config)
    if config.router_head_architecture == "three_tower_interaction_swiglu_stabilized":
        return Qwen3StabilizedThreeTowerInteractionRouterHead(config)
    raise AssertionError(
        f"Router head architecture was not validated: {config.router_head_architecture!r}"
    )

# LM wrapper with a frozen backbone and FP32 router; only router_head.* is trainable.
class Qwen3RouterForCausalLM(Qwen3RouterPreTrainedModel):
    """Train a causal Full-vs-Local router from parallel counterfactual oracles."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3Config):
        _set_router_config_defaults(config)
        super().__init__(config)
        self.model = Qwen3RouterModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_head = _build_router_head(config)
        self.post_init()
        self.router_head.float()

        # The Full and Local branches are label-generating oracles. Keeping only
        # the router head trainable also prevents accidental SFT of the parent.
        self.requires_grad_(False)
        self.router_head.requires_grad_(True)

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        if hasattr(self, "router_head"):
            self.router_head.float()
        return module

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def gradient_checkpointing_enable(self, *args, **kwargs):
        del args, kwargs
        logger.warning_once("Gradient checkpointing is disabled because the Qwen3 Router oracle runs under no_grad.")

    # Compute per-token KL(P_full || P_local) in chunks to limit peak memory.
    @torch.no_grad()
    def _compute_router_oracles(
        self,
        full_hidden_states: torch.Tensor,
        local_hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        valid_token_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        temperature = float(self.config.router_distill_temperature)
        chunk_size = int(self.config.router_kl_chunk_size)
        divergence_chunks = []
        full_nll_chunks = []
        local_nll_chunks = []
        output_weight = self.lm_head.weight.detach()

        aligned_labels = None
        aligned_label_mask = None
        if labels is not None:
            if valid_token_mask is None or position_ids is None:
                raise ValueError("Aligned token NLL requires valid_token_mask and position_ids.")
            if labels.shape != valid_token_mask.shape or labels.shape != position_ids.shape:
                raise ValueError("Aligned token NLL inputs must have identical [batch, sequence] shapes.")
            aligned_labels = torch.full_like(labels, -100)
            aligned_label_mask = torch.zeros_like(valid_token_mask)
            if labels.shape[1] > 1:
                aligned_labels[:, :-1] = labels[:, 1:]
                aligned_label_mask[:, :-1] = (
                    valid_token_mask[:, :-1]
                    & valid_token_mask[:, 1:]
                    & labels[:, 1:].ne(-100)
                    & position_ids[:, 1:].eq(position_ids[:, :-1] + 1)
                )

        for start in range(0, full_hidden_states.shape[1], chunk_size):
            end = min(start + chunk_size, full_hidden_states.shape[1])
            full_logits = nn.functional.linear(full_hidden_states[:, start:end], output_weight).float()
            local_logits = nn.functional.linear(local_hidden_states[:, start:end], output_weight).float()
            full_log_probs = nn.functional.log_softmax(full_logits / temperature, dim=-1)
            local_log_probs = nn.functional.log_softmax(local_logits / temperature, dim=-1)
            divergence = (full_log_probs.exp() * (full_log_probs - local_log_probs)).sum(dim=-1) * temperature**2
            divergence_chunks.append(divergence.clamp_min_(0.0))

            if aligned_labels is not None:
                target = aligned_labels[:, start:end]
                target_mask = aligned_label_mask[:, start:end]
                safe_target = target.clamp_min(0).unsqueeze(-1)
                full_nll = torch.logsumexp(full_logits, dim=-1) - full_logits.gather(-1, safe_target).squeeze(-1)
                local_nll = torch.logsumexp(local_logits, dim=-1) - local_logits.gather(-1, safe_target).squeeze(-1)
                nan = torch.full_like(full_nll, float("nan"))
                full_nll_chunks.append(torch.where(target_mask, full_nll, nan))
                local_nll_chunks.append(torch.where(target_mask, local_nll, nan))

        divergence = torch.cat(divergence_chunks, dim=1)
        if aligned_labels is None:
            return divergence, None, None
        return divergence, torch.cat(full_nll_chunks, dim=1), torch.cat(local_nll_chunks, dim=1)

    @torch.no_grad()
    def _compute_router_divergence(
        self,
        full_hidden_states: torch.Tensor,
        local_hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        divergence, _, _ = self._compute_router_oracles(full_hidden_states, local_hidden_states)
        return divergence

    # Build causal features: full_history[i] uses FullHidden[i-1], while current_embedding uses position i.
    def _build_router_features(
        self,
        input_embeddings: torch.Tensor,
        full_hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        valid_token_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        feature_schema = self.config.router_feature_schema
        feature_values = {}
        context_length = None
        if "full_history" in feature_schema:
            history = torch.zeros_like(full_hidden_states)
            history[:, 1:] = full_hidden_states[:, :-1]
            history = history.masked_fill(position_ids.eq(0)[..., None], 0.0)
            feature_values["full_history"] = history
        if "current_embedding" in feature_schema:
            feature_values["current_embedding"] = input_embeddings
        if "normalized_log_context" in feature_schema or "local_coverage" in feature_schema:
            context_length = position_ids.to(torch.float32).add(1.0).clamp_min_(1.0)
        if "normalized_log_context" in feature_schema:
            max_context_length = max(int(self.config.max_position_embeddings), 1)
            normalized_log_context = torch.log1p(context_length) / math.log1p(max_context_length)
            feature_values["normalized_log_context"] = normalized_log_context[..., None].to(input_embeddings.dtype)
        if "local_coverage" in feature_schema:
            local_coverage = _exact_local_coverage(
                context_length,
                int(self.config.router_sink_size),
                int(self.config.router_window_size),
            )
            feature_values["local_coverage"] = local_coverage[..., None].to(input_embeddings.dtype)

        router_features = torch.cat([feature_values[name] for name in feature_schema], dim=-1)
        return router_features.masked_fill(~valid_token_mask[..., None], 0.0)
    # Exclude segment starts; target-only eligibility at position i uses labels[i+1].
    def _get_router_loss_mask(
        self,
        valid_token_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
    ) -> torch.BoolTensor:
        if self.config.router_eligible_only:
            if labels is None or position_ids is None:
                raise ValueError("`router_eligible_only=True` requires labels and position_ids.")
            return build_router_eligible_mask(
                valid_token_mask,
                labels,
                position_ids,
                int(self.config.router_sink_size),
                int(self.config.router_window_size),
            )

        loss_mask = valid_token_mask if position_ids is None else valid_token_mask & position_ids.gt(0)
        if self.config.router_target_only:
            if labels is None:
                raise ValueError("`router_target_only=True` requires causal-LM labels.")
            prediction_target_mask = torch.zeros_like(loss_mask)
            prediction_target_mask[:, :-1] = labels[:, 1:].ne(-100)
            loss_mask = loss_mask & prediction_target_mask
        return loss_mask

    @staticmethod
    def _distributed_denominator(local_value: torch.Tensor) -> Tuple[torch.Tensor, int]:
        global_value = local_value.detach().float().clone()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_value, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
        return global_value, world_size

    @staticmethod
    def _context_masks(
        loss_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor],
    ):
        for batch_index in range(loss_mask.shape[0]):
            row_mask = loss_mask[batch_index]
            if not bool(row_mask.any()):
                continue
            if position_ids is None:
                yield batch_index, row_mask
                continue
            segment_ids = position_ids[batch_index].eq(0).to(torch.int64).cumsum(dim=0)
            for segment_id in torch.unique(segment_ids[row_mask], sorted=True):
                context_mask = row_mask & segment_ids.eq(segment_id)
                if bool(context_mask.any()):
                    yield batch_index, context_mask

    def _tail_weights(
        self,
        router_divergence: torch.Tensor,
        loss_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        weights = torch.zeros_like(router_divergence, dtype=torch.float32)
        for batch_index, context_mask in self._context_masks(loss_mask, position_ids):
            context_count = int(context_mask.sum().item())
            tail_count = max(1, int(math.ceil(context_count * float(self.config.router_tail_fraction))))
            context_indices = torch.nonzero(context_mask, as_tuple=False).flatten()
            tail_relative = torch.topk(
                router_divergence[batch_index, context_mask].float(),
                k=tail_count,
                largest=True,
                sorted=False,
            ).indices
            weights[batch_index, context_indices] = 1.0
            weights[batch_index, context_indices[tail_relative]] = float(self.config.router_tail_weight)
        return weights

    def _weighted_bce_loss(
        self,
        router_logits: torch.Tensor,
        router_targets: torch.Tensor,
        loss_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        loss_mask_fp32 = loss_mask.float()
        local_counts = torch.stack(
            (
                loss_mask_fp32.sum(),
                (router_targets.ge(0.5) & loss_mask).float().sum(),
            )
        )
        global_counts = local_counts.detach().clone()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_counts, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
        global_token_count = global_counts[0]
        if not bool(global_token_count > 0):
            return router_logits.float().sum() * 0.0, router_logits.new_tensor(1.0, dtype=torch.float32)

        global_positive_count = global_counts[1]
        global_negative_count = global_token_count - global_positive_count
        positive_weight = router_logits.new_tensor(float(self.config.router_positive_weight), dtype=torch.float32)
        if self.config.router_auto_positive_weight:
            has_both_classes = bool((global_positive_count > 0) & (global_negative_count > 0))
            if has_both_classes:
                positive_weight = global_negative_count / global_positive_count

        per_token_loss = nn.functional.binary_cross_entropy_with_logits(
            router_logits.float(), router_targets.float(), reduction="none"
        )
        sample_weight = router_targets.float() * positive_weight + (1.0 - router_targets.float()) * float(
            self.config.router_negative_weight
        )
        loss = (per_token_loss * sample_weight * loss_mask_fp32).sum() * (float(world_size) / global_token_count)
        return loss, positive_weight.detach()

    def _tail_log_kl_huber_loss(
        self,
        router_logits: torch.Tensor,
        router_divergence: torch.Tensor,
        loss_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        weights = self._tail_weights(router_divergence, loss_mask, position_ids)
        global_weight, world_size = self._distributed_denominator(weights.sum())
        if not bool(global_weight > 0):
            return router_logits.float().sum() * 0.0
        target = torch.log(router_divergence.float() + float(self.config.router_log_kl_epsilon))
        per_token_loss = nn.functional.huber_loss(
            router_logits.float(),
            target,
            reduction="none",
            delta=float(self.config.router_huber_delta),
        )
        return (per_token_loss * weights).sum() * (float(world_size) / global_weight)

    def _kl_mass_listwise_loss(
        self,
        router_logits: torch.Tensor,
        router_divergence: torch.Tensor,
        loss_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        local_loss = router_logits.float().sum() * 0.0
        local_context_count = router_logits.new_zeros((), dtype=torch.float32)
        epsilon = float(self.config.router_log_kl_epsilon)
        temperature = float(self.config.router_listwise_temperature)
        for batch_index, context_mask in self._context_masks(loss_mask, position_ids):
            target_mass = router_divergence[batch_index, context_mask].float() + epsilon
            target_distribution = target_mass / target_mass.sum()
            log_probability = nn.functional.log_softmax(
                router_logits[batch_index, context_mask].float() / temperature,
                dim=0,
            )
            local_loss = local_loss - (target_distribution * log_probability).sum()
            local_context_count = local_context_count + 1.0

        global_context_count, world_size = self._distributed_denominator(local_context_count)
        if not bool(global_context_count > 0):
            return router_logits.float().sum() * 0.0
        return local_loss * (float(world_size) / global_context_count)

    def _training_prediction_mask(
        self,
        router_logits: torch.Tensor,
        loss_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.BoolTensor:
        objective = self.config.router_objective
        if objective == "weighted_bce":
            return router_logits.ge(0.0) & loss_mask
        if objective == "tail_log_kl_huber":
            threshold = math.log(float(self.config.router_kl_threshold) + float(self.config.router_log_kl_epsilon))
            return router_logits.ge(threshold) & loss_mask

        predicted_full = torch.zeros_like(loss_mask)
        for batch_index, context_mask in self._context_masks(loss_mask, position_ids):
            context_count = int(context_mask.sum().item())
            full_count = max(1, int(math.ceil(context_count * float(self.config.router_tail_fraction))))
            context_indices = torch.nonzero(context_mask, as_tuple=False).flatten()
            selected_relative = torch.topk(
                router_logits[batch_index, context_mask].float(),
                k=full_count,
                largest=True,
                sorted=False,
            ).indices
            predicted_full[batch_index, context_indices[selected_relative]] = True
        return predicted_full

    # DDP averaging yields global token/context normalization for class statistics and the objective.
    def _compute_router_loss(
        self,
        router_logits: torch.Tensor,
        router_targets: torch.Tensor,
        router_divergence: torch.Tensor,
        valid_token_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, dict]:
        loss_mask = self._get_router_loss_mask(valid_token_mask, labels, position_ids)
        loss_mask_fp32 = loss_mask.float()
        predicted_full = self._training_prediction_mask(router_logits, loss_mask, position_ids)
        predicted_local = loss_mask & ~predicted_full
        if self.config.router_objective == "weighted_bce":
            soft_local_weight = torch.sigmoid(-router_logits.float()) * loss_mask_fp32
        else:
            soft_local_weight = predicted_local.float()
        local_statistics = torch.stack(
            (
                loss_mask_fp32.sum(),
                (router_targets.float() * loss_mask_fp32).sum(),
                (router_targets.ge(0.5) & loss_mask).float().sum(),
                predicted_full.float().sum(),
                (router_divergence.float() * loss_mask_fp32).sum(),
                (router_divergence.float() * predicted_local.float()).sum(),
                (router_divergence.float() * soft_local_weight).sum(),
                predicted_local.float().sum(),
                (router_targets.ge(0.5) & predicted_full).float().sum(),
                (router_targets.lt(0.5) & predicted_full).float().sum(),
                (router_targets.ge(0.5) & predicted_local).float().sum(),
                (router_targets.lt(0.5) & predicted_local).float().sum(),
            )
        ).detach()
        global_statistics = local_statistics.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_statistics, op=torch.distributed.ReduceOp.SUM)

        global_token_count = global_statistics[0].clamp_min(1.0)
        objective = self.config.router_objective
        positive_weight = router_logits.new_tensor(1.0, dtype=torch.float32)
        if objective == "weighted_bce":
            loss, positive_weight = self._weighted_bce_loss(router_logits, router_targets, loss_mask)
        elif objective == "tail_log_kl_huber":
            loss = self._tail_log_kl_huber_loss(router_logits, router_divergence, loss_mask, position_ids)
        elif objective == "kl_mass_listwise":
            loss = self._kl_mass_listwise_loss(router_logits, router_divergence, loss_mask, position_ids)
        else:
            huber_loss = self._tail_log_kl_huber_loss(
                router_logits, router_divergence, loss_mask, position_ids
            )
            listwise_loss = self._kl_mass_listwise_loss(
                router_logits, router_divergence, loss_mask, position_ids
            )
            alpha = float(self.config.router_hybrid_alpha)
            loss = alpha * huber_loss + (1.0 - alpha) * listwise_loss
        metrics = {
            "target_rate": global_statistics[1] / global_token_count,
            "prediction_rate_at_0_5": global_statistics[3] / global_token_count,
            "mean_divergence": global_statistics[4] / global_token_count,
            "effective_positive_weight": positive_weight.detach(),
            "effective_hard_divergence": global_statistics[5] / global_token_count,
            "effective_soft_divergence": global_statistics[6] / global_token_count,
            "local_mean_divergence": global_statistics[5] / global_statistics[7].clamp_min(1.0),
            "accuracy": (global_statistics[8] + global_statistics[11]) / global_token_count,
            "precision": global_statistics[8] / (global_statistics[8] + global_statistics[9]).clamp_min(1.0),
            "recall": global_statistics[8] / (global_statistics[8] + global_statistics[10]).clamp_min(1.0),
        }
        metrics["f1"] = (
            2.0
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"]).clamp_min(1.0e-12)
        )
        return loss, metrics
    # Complete training forward pass, including targets, loss, and probe outputs.
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Qwen3RouterCausalLMOutput:
        del logits_to_keep
        full_hidden_states, local_hidden_states, input_embeddings, position_ids, valid_token_mask = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        return_token_nll = bool(self.config.router_return_token_nll)
        if return_token_nll and labels is None:
            raise ValueError("`router_return_token_nll=True` requires causal-LM labels.")
        router_divergence, full_token_nll, local_token_nll = self._compute_router_oracles(
            full_hidden_states,
            local_hidden_states,
            labels=labels if return_token_nll else None,
            valid_token_mask=valid_token_mask if return_token_nll else None,
            position_ids=position_ids if return_token_nll else None,
        )
        if self.config.router_check_finite:
            _check_router_tensor_is_finite("router_divergence", router_divergence)
        if self.config.router_use_soft_targets:
            router_targets = torch.sigmoid(
                (router_divergence - float(self.config.router_kl_threshold))
                / float(self.config.router_target_temperature)
            )
        else:
            router_targets = router_divergence.gt(float(self.config.router_kl_threshold)).float()

        router_features = self._build_router_features(
            input_embeddings,
            full_hidden_states,
            position_ids,
            valid_token_mask,
        )
        router_logits = self.router_head(router_features)
        if self.config.router_check_finite:
            _check_router_tensor_is_finite("router_targets", router_targets)
            _check_router_tensor_is_finite("router_logits", router_logits)
        router_loss, router_metrics = self._compute_router_loss(
            router_logits,
            router_targets,
            router_divergence,
            valid_token_mask,
            labels,
            position_ids,
        )
        router_loss_mask = self._get_router_loss_mask(valid_token_mask, labels, position_ids)
        loss = router_loss * float(self.config.router_loss_weight)

        if return_dict is False:
            return (loss, router_logits, router_targets, router_divergence)
        return_hidden_states = bool(self.config.router_return_hidden_states)
        probe_tensors = {
            "router_divergence": router_divergence.detach(),
            "router_logits": router_logits.detach(),
            "router_targets": router_targets.detach(),
            "router_feature_l2": router_features.detach().float().norm(dim=-1),
            "position_ids": position_ids.detach(),
            "valid_token_mask": router_loss_mask.detach(),
            **({"input_ids": input_ids.detach()} if input_ids is not None else {}),
        }
        if self.config.router_objective == "weighted_bce":
            probe_tensors["router_full_probability"] = torch.sigmoid(router_logits.detach())
        if full_token_nll is not None:
            probe_tensors["full_token_nll"] = full_token_nll.detach()
            probe_tensors["local_token_nll"] = local_token_nll.detach()
        return Qwen3RouterCausalLMOutput(
            loss=loss,
            router_loss=router_loss,
            router_logits=router_logits,
            router_targets=router_targets,
            router_divergence=router_divergence,
            full_token_nll=full_token_nll,
            local_token_nll=local_token_nll,
            router_target_rate=router_metrics["target_rate"],
            router_prediction_rate_at_0_5=router_metrics["prediction_rate_at_0_5"],
            router_mean_divergence=router_metrics["mean_divergence"],
            router_effective_positive_weight=router_metrics["effective_positive_weight"],
            router_effective_hard_divergence=router_metrics["effective_hard_divergence"],
            router_effective_soft_divergence=router_metrics["effective_soft_divergence"],
            router_local_mean_divergence=router_metrics["local_mean_divergence"],
            router_accuracy=router_metrics["accuracy"],
            router_precision=router_metrics["precision"],
            router_recall=router_metrics["recall"],
            router_f1=router_metrics["f1"],
            training_metrics={
                "router/target_rate": router_metrics["target_rate"],
                "router/prediction_rate_at_0_5": router_metrics["prediction_rate_at_0_5"],
                "router/mean_kl": router_metrics["mean_divergence"],
                "router/oracle_mean_kl": router_metrics["mean_divergence"],
                "router/effective_hard_kl": router_metrics["effective_hard_divergence"],
                "router/effective_soft_kl": router_metrics["effective_soft_divergence"],
                "router/local_mean_kl": router_metrics["local_mean_divergence"],
                "router/positive_weight": router_metrics["effective_positive_weight"],
                "router/accuracy": router_metrics["accuracy"],
                "router/precision": router_metrics["precision"],
                "router/recall": router_metrics["recall"],
                "router/f1": router_metrics["f1"],
            },
            training_probe_tensors=probe_tensors,
            full_hidden_states=full_hidden_states if return_hidden_states else None,
            local_hidden_states=local_hidden_states if return_hidden_states else None,
        )


class Qwen3FullLocalForCausalLM(Qwen3RouterForCausalLM):
    """Inference-only official-Qwen wrapper for Full/Local prior observation.

    The module reuses the weight-sharing Full/Local backbone and language-model
    head, but deliberately removes the Router head.  Its forward path therefore
    cannot consume a trained Router checkpoint or produce a routing decision.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        del self.router_head
        self.requires_grad_(False)

    @torch.no_grad()
    def _compute_full_local_prior_metrics(
        self,
        full_hidden_states: torch.Tensor,
        local_hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        valid_token_mask: torch.BoolTensor,
        position_ids: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute target quality, Top-1 shape, bidirectional KL, and JS in FP32."""

        if (
            input_ids.shape != labels.shape
            or labels.shape != valid_token_mask.shape
            or labels.shape != position_ids.shape
        ):
            raise ValueError("Full/Local prior metric inputs must have identical [batch, sequence] shapes.")
        if full_hidden_states.shape != local_hidden_states.shape:
            raise ValueError("Full and Local hidden-state tensors must have identical shapes.")
        if full_hidden_states.shape[:2] != labels.shape:
            raise ValueError("Full/Local hidden states must align with labels.")

        aligned_labels = torch.full_like(labels, -100)
        aligned_all_tokens = torch.full_like(input_ids, -100)
        aligned_target_mask = torch.zeros_like(valid_token_mask)
        structural_aligned_target_mask = torch.zeros_like(valid_token_mask)
        if labels.shape[1] > 1:
            aligned_labels[:, :-1] = labels[:, 1:]
            aligned_all_tokens[:, :-1] = input_ids[:, 1:]
            structural_aligned_target_mask[:, :-1] = (
                valid_token_mask[:, :-1]
                & valid_token_mask[:, 1:]
                & position_ids[:, 1:].eq(position_ids[:, :-1] + 1)
            )
            aligned_target_mask[:, :-1] = (
                structural_aligned_target_mask[:, :-1] & labels[:, 1:].ne(-100)
            )

        chunk_size = int(self.config.router_kl_chunk_size)
        output_weight = self.lm_head.weight.detach()
        full_to_local_chunks = []
        local_to_full_chunks = []
        js_chunks = []
        full_nll_chunks = []
        local_nll_chunks = []
        all_full_nll_chunks = []
        all_local_nll_chunks = []
        full_top1_token_chunks = []
        local_top1_token_chunks = []
        full_top1_probability_chunks = []
        local_top1_probability_chunks = []
        full_top1_margin_chunks = []
        local_top1_margin_chunks = []
        full_entropy_chunks = []
        local_entropy_chunks = []
        log_two = math.log(2.0)
        for start in range(0, full_hidden_states.shape[1], chunk_size):
            end = min(start + chunk_size, full_hidden_states.shape[1])
            full_logits = nn.functional.linear(full_hidden_states[:, start:end], output_weight).float()
            local_logits = nn.functional.linear(local_hidden_states[:, start:end], output_weight).float()
            full_log_probs = nn.functional.log_softmax(full_logits, dim=-1)
            local_log_probs = nn.functional.log_softmax(local_logits, dim=-1)
            full_probs = full_log_probs.exp()
            local_probs = local_log_probs.exp()

            full_to_local = (full_probs * (full_log_probs - local_log_probs)).sum(dim=-1)
            local_to_full = (local_probs * (local_log_probs - full_log_probs)).sum(dim=-1)
            log_mixture = torch.logaddexp(full_log_probs, local_log_probs) - log_two
            js_divergence = 0.5 * (
                (full_probs * (full_log_probs - log_mixture)).sum(dim=-1)
                + (local_probs * (local_log_probs - log_mixture)).sum(dim=-1)
            )
            full_to_local_chunks.append(full_to_local.clamp_min_(0.0))
            local_to_full_chunks.append(local_to_full.clamp_min_(0.0))
            js_chunks.append(js_divergence.clamp_min_(0.0))

            full_top2_log_probs, full_top2_ids = torch.topk(full_log_probs, k=2, dim=-1)
            local_top2_log_probs, local_top2_ids = torch.topk(local_log_probs, k=2, dim=-1)
            full_top2_probs = full_top2_log_probs.exp()
            local_top2_probs = local_top2_log_probs.exp()
            full_top1_token_chunks.append(full_top2_ids[..., 0])
            local_top1_token_chunks.append(local_top2_ids[..., 0])
            full_top1_probability_chunks.append(full_top2_probs[..., 0])
            local_top1_probability_chunks.append(local_top2_probs[..., 0])
            full_top1_margin_chunks.append(full_top2_probs[..., 0] - full_top2_probs[..., 1])
            local_top1_margin_chunks.append(local_top2_probs[..., 0] - local_top2_probs[..., 1])
            full_entropy_chunks.append(-(full_probs * full_log_probs).sum(dim=-1))
            local_entropy_chunks.append(-(local_probs * local_log_probs).sum(dim=-1))

            target = aligned_labels[:, start:end]
            target_mask = aligned_target_mask[:, start:end]
            safe_target = target.clamp_min(0).unsqueeze(-1)
            full_nll = torch.logsumexp(full_logits, dim=-1) - full_logits.gather(-1, safe_target).squeeze(-1)
            local_nll = torch.logsumexp(local_logits, dim=-1) - local_logits.gather(-1, safe_target).squeeze(-1)
            nan = torch.full_like(full_nll, float("nan"))
            full_nll_chunks.append(torch.where(target_mask, full_nll, nan))
            local_nll_chunks.append(torch.where(target_mask, local_nll, nan))

            all_target = aligned_all_tokens[:, start:end]
            all_target_mask = structural_aligned_target_mask[:, start:end]
            safe_all_target = all_target.clamp_min(0).unsqueeze(-1)
            all_full_nll = torch.logsumexp(full_logits, dim=-1) - full_logits.gather(
                -1, safe_all_target
            ).squeeze(-1)
            all_local_nll = torch.logsumexp(local_logits, dim=-1) - local_logits.gather(
                -1, safe_all_target
            ).squeeze(-1)
            all_full_nll_chunks.append(torch.where(all_target_mask, all_full_nll, nan))
            all_local_nll_chunks.append(torch.where(all_target_mask, all_local_nll, nan))

        return {
            "full_to_local_kl": torch.cat(full_to_local_chunks, dim=1),
            "local_to_full_kl": torch.cat(local_to_full_chunks, dim=1),
            "js_divergence": torch.cat(js_chunks, dim=1),
            "full_token_nll": torch.cat(full_nll_chunks, dim=1),
            "local_token_nll": torch.cat(local_nll_chunks, dim=1),
            "all_full_token_nll": torch.cat(all_full_nll_chunks, dim=1),
            "all_local_token_nll": torch.cat(all_local_nll_chunks, dim=1),
            "full_top1_token_id": torch.cat(full_top1_token_chunks, dim=1),
            "local_top1_token_id": torch.cat(local_top1_token_chunks, dim=1),
            "full_top1_probability": torch.cat(full_top1_probability_chunks, dim=1),
            "local_top1_probability": torch.cat(local_top1_probability_chunks, dim=1),
            "full_top1_margin": torch.cat(full_top1_margin_chunks, dim=1),
            "local_top1_margin": torch.cat(local_top1_margin_chunks, dim=1),
            "full_entropy": torch.cat(full_entropy_chunks, dim=1),
            "local_entropy": torch.cat(local_entropy_chunks, dim=1),
            "aligned_target_mask": aligned_target_mask,
            "structural_aligned_target_mask": structural_aligned_target_mask,
        }

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Qwen3FullLocalCausalLMOutput:
        del logits_to_keep
        if labels is None:
            raise ValueError("Full/Local prior observation requires causal-LM labels.")
        if input_ids is None:
            raise ValueError("Full/Local token statistics require input_ids for all-token sensitivity.")
        if return_dict is False:
            raise ValueError("Full/Local prior observation requires return_dict=True.")

        full_hidden_states, local_hidden_states, _, position_ids, valid_token_mask = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        metrics = self._compute_full_local_prior_metrics(
            full_hidden_states,
            local_hidden_states,
            input_ids,
            labels,
            valid_token_mask,
            position_ids,
        )
        eligible_token_mask = build_router_eligible_mask(
            valid_token_mask,
            labels,
            position_ids,
            int(self.config.router_sink_size),
            int(self.config.router_window_size),
        )
        structural_eligible_token_mask = build_router_eligible_mask(
            valid_token_mask,
            input_ids,
            position_ids,
            int(self.config.router_sink_size),
            int(self.config.router_window_size),
        )
        return Qwen3FullLocalCausalLMOutput(
            **metrics,
            position_ids=position_ids,
            eligible_token_mask=eligible_token_mask,
            structural_eligible_token_mask=structural_eligible_token_mask,
        )


if is_liger_kernel_available():
    apply_rotary_pos_emb = liger_rotary_pos_emb
    Qwen3RMSNorm = LigerRMSNorm
    Qwen3MLP = LigerSwiGLUMLP
    logger.info_rank0("Apply liger kernel to Qwen3.")

if is_torch_npu_available():
    from veomni_kernels.rms_norm.npu_ttx_rmsnorm_kernel import triton_rmsnorm_npu
    from veomni_kernels.rope.npu_ttx_rope_kernel import triton_rope_npu

    from veomni.ops.attention import npu_flash_attention

    apply_rotary_pos_emb = triton_rope_npu
    rms_norm = triton_rmsnorm_npu
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", npu_flash_attention)
    logger.info_rank0("Apply npu kernels to Qwen3.")

ModelClass = Qwen3RouterForCausalLM

__all__ = [
    "Qwen3FullLocalCausalLMOutput",
    "Qwen3FullLocalForCausalLM",
    "Qwen3RouterForCausalLM",
    "Qwen3RouterModel",
    "Qwen3RouterPreTrainedModel",
    "build_router_eligible_mask",
    "create_router_past_block_mask",
    "online_softmax_merge_local_self",
]
