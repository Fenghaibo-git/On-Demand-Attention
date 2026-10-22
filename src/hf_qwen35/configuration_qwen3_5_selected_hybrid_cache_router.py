# Modified for the portable Router v4.2 release; see NOTICE.
"""Configuration for the Qwen3.5 selected-hybrid-cache Router runtime."""

import os

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class Qwen3_5SelectedHybridCacheRouterConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_selected_hybrid_cache_router"

    def __init__(
        self,
        router_feature_schema=None,
        router_head_architecture="three_tower_interaction_swiglu",
        router_hidden_size=1024,
        router_num_blocks=2,
        router_intermediate_size=2048,
        router_expected_head_parameter_count=28_325_889,
        router_expected_head_tensor_count=22,
        router_sink_size=4,
        router_window_size=2048,
        router_action_threshold=0.0,
        router_checkpoint=None,
        router_audit=None,
        router_summary_path=None,
        router_base_model_path=None,
        router_base_weights_manifest_sha256=(
            "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
        ),
        router_base_config_sha256=(
            "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
        ),
        router_expected_training_schema_sha256=(
            "d17934038878169f856ae60dbf7503961358f2260a752d92c8321dcabf8f9789"
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.router_feature_schema = list(
            router_feature_schema
            or ["history_state", "current_embedding", "current_swa_hidden"]
        )
        self.router_head_architecture = str(router_head_architecture)
        self.router_hidden_size = int(router_hidden_size)
        self.router_num_blocks = int(router_num_blocks)
        self.router_intermediate_size = int(router_intermediate_size)
        self.router_expected_head_parameter_count = int(
            router_expected_head_parameter_count
        )
        self.router_expected_head_tensor_count = int(
            router_expected_head_tensor_count
        )
        self.router_sink_size = int(router_sink_size)
        self.router_window_size = int(router_window_size)
        self.router_action_threshold = float(router_action_threshold)
        self.router_checkpoint = router_checkpoint
        self.router_audit = router_audit
        self.router_summary_path = router_summary_path
        self.router_base_model_path = str(
            router_base_model_path
            or os.path.join(os.environ.get("ROUTER_MODEL_ROOT", "/path/to/models"), "Qwen3.5-2B")
        )
        self.router_base_weights_manifest_sha256 = str(
            router_base_weights_manifest_sha256
        )
        self.router_base_config_sha256 = str(router_base_config_sha256)
        self.router_expected_training_schema_sha256 = str(
            router_expected_training_schema_sha256
        )


__all__ = ["Qwen3_5SelectedHybridCacheRouterConfig"]
