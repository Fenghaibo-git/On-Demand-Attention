"""Scale the qualified hybrid three-stream implementation to 27B's layout.

Every token mixer, candidate calculation, mask, feature and objective is reused.
Only construction/validation is separate because the historical constructor
intentionally accepts exactly the 2B 24-layer layout. No module-global patching.
"""

from copy import deepcopy

from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

from ..qwen3_5_three_stream_router.modeling_qwen3_5_three_stream_router import (
    _DEFAULTS,
    Qwen3_5ThreeStreamModel,
    Qwen3_5ThreeStreamNllSelectedRouterForCausalLM,
    qwen3_5_three_stream_schema_sha256,
)
from ..qwen3_one_step_router.modeling_qwen3_one_step_router import _set_one_step_config_defaults
from ..qwen3_router.modeling_qwen3_router import _build_router_head, _set_router_config_defaults
from ..qwen3_large_robust_margin_router.objective import (
    RobustMarginObjectiveMixin, finish_contract, set_robust_defaults,
)


def set_scaled_hybrid_defaults(config):
    set_robust_defaults(config)
    for name, value in _DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, deepcopy(value))
    layers = int(config.num_hidden_layers)
    pattern = ["linear_attention"] * 3 + ["full_attention"]
    if layers <= 0 or layers % 4 or list(config.layer_types) != pattern * (layers // 4):
        raise ValueError("scaled hybrid layout must repeat three DeltaNet layers and one Full attention layer")
    # Keep all execution/feature contracts from the qualified implementation.
    customizable = {
        "router_expected_head_parameter_count", "router_hidden_size", "router_intermediate_size",
        "router_return_hidden_states", "router_sink_size", "router_window_size",
        "router_control_threshold_status", "router_control_p999_nll_threshold",
    }
    for name, expected in _DEFAULTS.items():
        if name in customizable or name in ("router_history_nll_threshold",):
            continue
        if getattr(config, name) != expected:
            raise ValueError(f"scaled hybrid execution contract drift: {name}")
    config.router_objective = "weighted_bce"
    try:
        _set_one_step_config_defaults(config)
        _set_router_config_defaults(config)
    finally:
        config.router_objective = "robust_decision_margin_huber"
    observed = qwen3_5_three_stream_schema_sha256(config)
    if getattr(config, "router_schema_sha256", None) not in (None, "", observed):
        raise ValueError("scaled hybrid structural schema hash mismatch")
    config.router_schema_sha256 = observed


class Qwen3_5LargeRobustMarginRouterForCausalLM(
    RobustMarginObjectiveMixin, Qwen3_5ThreeStreamNllSelectedRouterForCausalLM
):
    def __init__(self, config):
        set_scaled_hybrid_defaults(config)
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5ThreeStreamModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_head = _build_router_head(config)
        self.post_init()
        self.router_head.float()
        self.requires_grad_(False)
        self.router_head.requires_grad_(True)
        config.router_head_parameter_count = sum(p.numel() for p in self.router_head.parameters())
        config.router_head_tensor_count = len(self.router_head.state_dict())
        finish_contract(self)


ModelClass = Qwen3_5LargeRobustMarginRouterForCausalLM
