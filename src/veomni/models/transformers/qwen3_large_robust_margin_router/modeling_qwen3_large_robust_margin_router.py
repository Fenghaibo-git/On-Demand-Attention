"""Qwen3 A0 three-stream robust margin, independent of Qwen3.5 dependencies."""

from ..qwen3_three_stream_nll_selected_router.modeling_qwen3_three_stream_nll_selected_router import _SELECTED_DEFAULTS
from ..qwen3_three_stream_router.modeling_qwen3_three_stream_router import Qwen3ThreeStreamRouterForCausalLM
from .objective import RobustMarginObjectiveMixin, finish_contract, set_robust_defaults


class Qwen3LargeRobustMarginRouterForCausalLM(RobustMarginObjectiveMixin, Qwen3ThreeStreamRouterForCausalLM):
    def __init__(self, config):
        set_robust_defaults(config)
        for name, value in _SELECTED_DEFAULTS.items():
            if not hasattr(config, name):
                setattr(config, name, value)
        # The historical shared constructor has an older public enum.
        # Loss dispatch uses the methods above, never its legacy BCE branch.
        config.router_objective = "weighted_bce"
        try:
            super().__init__(config)
        finally:
            config.router_objective = "robust_decision_margin_huber"
        finish_contract(self)


ModelClass = Qwen3LargeRobustMarginRouterForCausalLM
