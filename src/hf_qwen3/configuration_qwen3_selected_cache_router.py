# Modified for On Demand Attention; see NOTICE.
"""Configuration for the causal selected-mixed-cache Qwen3 Router runtime."""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3SelectedCacheRouterConfig(Qwen3Config):
    model_type = "qwen3_selected_cache_router"

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
        router_base_model_path="Qwen/Qwen3-8B",
        router_base_weights_manifest_sha256=(
            "f9bbbc07c97e9be1b545c383464d7c1392c66a7d74fe7173bb40ea6f786841dc"
        ),
        router_base_config_sha256=(
            "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
        ),
        router_expected_training_schema_sha256=(
            "72e8d00db25c2d1f729c9eb6e9768a66a27fb3ea9db17b923b0564a66358484a"
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
        self.router_base_model_path = str(router_base_model_path)
        self.router_base_weights_manifest_sha256 = str(
            router_base_weights_manifest_sha256
        )
        self.router_base_config_sha256 = str(router_base_config_sha256)
        self.router_expected_training_schema_sha256 = str(
            router_expected_training_schema_sha256
        )


__all__ = ["Qwen3SelectedCacheRouterConfig"]
