"""Three-stream Qwen3 training with a previous threshold-selected hidden.

The Full trunk remains the formal history.  Independent Full and SWA
current-token counterfactual streams produce the signed-NLL target
``g_i = NLL_local(i) - NLL_full(i)``.  Only the Router history feature is
selected: Local when ``g_i < 0.001`` and Full trunk otherwise.  The selected
hidden is shifted by one position in the shared three-stream implementation,
so the target token never leaks into the Router action that predicted it.
"""

from __future__ import annotations

import math

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..qwen3_three_stream_router.modeling_qwen3_three_stream_router import (
    Qwen3ThreeStreamRouterForCausalLM,
)


_SELECTED_DEFAULTS = {
    "router_schema_version": 4,
    "router_training_history_mode": "full",
    "router_runtime_history_mode": "full_cache_nll_selected",
    "router_training_attention_backend": "flex_attention",
    "router_swa_all_layers": True,
    "router_window_includes_current": True,
    "router_feature_schema": [
        "history_state",
        "current_embedding",
        "current_swa_hidden",
    ],
    "router_head_architecture": "three_tower_interaction_swiglu",
    "router_hidden_size": 1024,
    "router_num_blocks": 2,
    "router_intermediate_size": 2048,
    "router_expected_head_parameter_count": 28_325_889,
    "router_expected_head_tensor_count": 22,
    "router_history_selector": "local_when_signed_nll_lt_threshold",
    "router_history_nll_threshold": 0.001,
    "router_control_p999_nll_threshold": 0.2687559947967575,
    "router_one_step_harm_threshold": 0.2687559947967575,
}


class Qwen3ThreeStreamNllSelectedRouterForCausalLM(
    Qwen3ThreeStreamRouterForCausalLM
):
    """Exact A0 Head trained from three counterfactual computation streams."""

    def __init__(self, config: Qwen3Config):
        for name, value in _SELECTED_DEFAULTS.items():
            if not hasattr(config, name):
                setattr(config, name, value)
        super().__init__(config)

        if int(config.router_schema_version) != 4:
            raise ValueError("Threshold-selected Router requires schema version 4.")
        if config.router_training_history_mode != "full":
            raise ValueError("Threshold-selected Router requires Full training history.")
        if config.router_runtime_history_mode != "full_cache_nll_selected":
            raise ValueError("Threshold-selected Router requires Full runtime K/V.")
        if config.router_history_selector != (
            "local_when_signed_nll_lt_threshold"
        ):
            raise ValueError("Unknown threshold-selected history rule.")
        threshold = float(config.router_history_nll_threshold)
        if not math.isfinite(threshold):
            raise ValueError("`router_history_nll_threshold` must be finite.")
        control_threshold = float(config.router_control_p999_nll_threshold)
        if not math.isfinite(control_threshold) or control_threshold <= 0.0:
            raise ValueError(
                "`router_control_p999_nll_threshold` must be finite and positive."
            )
        if float(config.router_one_step_harm_threshold) != control_threshold:
            raise ValueError(
                "The control-aware metric threshold must equal the configured P99.9."
            )


ModelClass = Qwen3ThreeStreamNllSelectedRouterForCausalLM
