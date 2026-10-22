"""Native Gemma4-unified text backbone with isolated three-stream supervision.

Only global-attention layers are restricted by the Router's Local action.
Native sliding1024 layers, K=V projections, proportional RoPE, all four norms,
embedding scale, layer scalar and final logit softcap are preserved.
"""
from copy import deepcopy
import hashlib
import json

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedPreTrainedModel, Gemma4UnifiedTextModel, apply_rotary_pos_emb, repeat_kv,
)

from ..qwen3_router.modeling_qwen3_router import (
    Qwen3RouterForCausalLM, _build_router_head,
)
from ..qwen3_one_step_router.modeling_qwen3_one_step_router import Qwen3OneStepRouterForCausalLM
from ..qwen3_three_stream_router.modeling_qwen3_three_stream_router import (
    Qwen3ThreeStreamRouterForCausalLM, create_three_stream_masks, online_softmax_merge_local_self,
)
from ..qwen3_large_robust_margin_router.objective import (
    RobustMarginObjectiveMixin, set_robust_defaults, finish_contract,
)

_FLEX = {}
DEFAULTS = {
    'router_schema_version': 1,
    'router_training_history_mode': 'full',
    'router_runtime_history_mode': 'selected_hybrid_cache',
    'router_training_attention_backend': 'flex_attention_native_gemma4_unified',
    'router_local_definition': 'sink_window_global_layers_preserve_native_sliding',
    'router_equivalent_stream_canonicalization': 'full_cf_to_trunk_each_layer_local_before2053',
    'router_history_selector': 'local_when_signed_nll_lt_threshold',
    'router_sink_size': 4, 'router_window_size': 2048, 'router_window_includes_current': True,
    'router_feature_schema': ['history_state','current_embedding','current_swa_hidden'],
    'router_head_architecture': 'three_tower_interaction_swiglu',
    'router_hidden_size': 1024, 'router_intermediate_size': 2048, 'router_num_blocks': 2,
    'router_return_hidden_states': False, 'router_check_finite': True,
    'router_kl_chunk_size': 16, 'router_group_rms_enabled': False,
    'router_learned_fusion_scales': False, 'router_layer_scale_enabled': False,
    'router_one_step_sign_aux_temperature': .1, 'router_one_step_sign_near_zero_threshold': .0001,
    'router_control_threshold_status': 'not_used_robust_margin', 'router_control_p999_nll_threshold': 0.,
}


def set_defaults(config):
    set_robust_defaults(config)
    for k,v in DEFAULTS.items():
        if not hasattr(config,k): setattr(config,k,deepcopy(v))
    if (config.num_kv_shared_layers != 0 or getattr(config,'enable_moe_block',False)
            or getattr(config,'hidden_size_per_layer_input',0) != 0
            or config.attention_dropout != 0 or config.use_bidirectional_attention == 'all'):
        raise ValueError('only causal dense Gemma4 unified without shared-layer KV is qualified')
    if len(config.layer_types) != config.num_hidden_layers or not set(config.layer_types) <= {'full_attention','sliding_attention'}:
        raise ValueError('invalid Gemma layer layout')
    if (config.router_training_history_mode != 'full'
            or config.router_history_selector != 'local_when_signed_nll_lt_threshold'
            or config.router_head_architecture != 'three_tower_interaction_swiglu'
            or config.router_feature_schema != ['history_state','current_embedding','current_swa_hidden']):
        raise ValueError('Gemma v4.2 feature/history contract changed')
    payload = {
        'format':'gemma4-unified-three-stream-v1',
        'native':{k:getattr(config,k,None) for k in ('model_type','hidden_size','intermediate_size',
            'num_hidden_layers','num_attention_heads','num_key_value_heads','num_global_key_value_heads',
            'head_dim','global_head_dim','attention_k_eq_v','layer_types','sliding_window',
            'rope_parameters','rms_norm_eps','final_logit_softcapping','tie_word_embeddings')},
        'router':{k:getattr(config,k) for k in DEFAULTS if k not in ('router_return_hidden_states','router_check_finite')},
        'head_norm':'Qwen_A0_RMSNorm_unmodified', 'embedding_feature':'native_scaled_token_embedding',
        'full_trunk':'native_Gemma_QKV_MLP_norms_with_equivalent_flex_attention',
        'candidate_history':'past_only_full_trunk_plus_isolated_current_candidate_KV',
    }
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if getattr(config,'router_schema_sha256',None) not in (None,'',digest):
        raise ValueError('Gemma structural schema mismatch')
    config.router_schema_sha256=digest
    return payload


def run_flex(query,key,value,mask,role,groups,with_lse=False):
    options={'ROWS_GUARANTEED_SAFE':False}
    if query.size(-1)>256:
        # The native global heads are512-wide. Torch2.7's default tile exceeds
        # A100 shared memory; only the tile/staging changes, never mask/scale.
        options.update(BLOCK_M=32,BLOCK_N=32,num_stages=1,num_warps=4)
    if query.device.type == 'cuda':
        if role not in _FLEX:
            _FLEX[role]=torch.compile(flex_attention,dynamic=True,fullgraph=True)
        call=_FLEX[role]
        # Two head dimensions, native sliding/global masks and isolated
        # candidates require more than Torch2.7's eight shared-code variants.
        # Keep the override local and fail closed instead of silently switching
        # to dense eager attention at long sequence lengths.
        with torch._dynamo.config.patch(recompile_limit=64):
            return call(query,key,value,block_mask=mask,scale=1.0,enable_gqa=groups>1,
                        return_lse=with_lse,kernel_options=options)
    call=flex_attention
    return call(query,key,value,block_mask=mask,scale=1.0,enable_gqa=groups>1,
                return_lse=with_lse,kernel_options=options)


def project(attn,hidden,position_embeddings):
    shape=(*hidden.shape[:-1],-1,attn.head_dim)
    cos,sin=position_embeddings
    q=attn.q_norm(attn.q_proj(hidden).view(shape))
    q=apply_rotary_pos_emb(q,cos,sin,unsqueeze_dim=2).transpose(1,2)
    raw_k=attn.k_proj(hidden).view(shape)
    raw_v=attn.v_proj(hidden).view(shape) if attn.v_proj is not None else raw_k
    k=apply_rotary_pos_emb(attn.k_norm(raw_k),cos,sin,unsqueeze_dim=2).transpose(1,2)
    v=attn.v_norm(raw_v).transpose(1,2)
    return q,k,v


def candidate(attn,q,trunk_k,trunk_v,k,v,mask,positions,valid,role):
    past,lse=run_flex(q,trunk_k,trunk_v,mask,role,attn.num_key_value_groups,True)
    self_k=repeat_kv(k,attn.num_key_value_groups)
    self_v=repeat_kv(v,attn.num_key_value_groups)
    self_score=(q.float()*self_k.float()).sum(-1) # Native Gemma attention scale is1.
    merged,_=online_softmax_merge_local_self(past,lse,self_score,self_v,
        empty_past_mask=~(valid & positions.gt(0)),valid_token_mask=valid)
    return merged


def finish_layer(layer,hidden,attention,valid):
    b,_,t,_=attention.shape
    attention=attention.masked_fill(~valid[:,None,:,None],0.)
    attention=layer.self_attn.o_proj(attention.transpose(1,2).reshape(b,t,-1).contiguous())
    hidden=hidden+layer.post_attention_layernorm(attention)
    hidden=hidden+layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(hidden)))
    return hidden*layer.layer_scalar


class Gemma4UnifiedThreeStreamModel(Gemma4UnifiedTextModel):
    def __init__(self, config):
        super().__init__(config)
        # Native Gemma registers its nonpersistent RoPE buffers by iterating a
        # set of layer types. Python hash randomization can give each DDP rank
        # a different order. DDP broadcasts buffers positionally, so normalize
        # registration order without changing any value or persistence flag.
        rope = self.rotary_emb
        rope._buffers = {name: rope._buffers[name] for name in sorted(rope._buffers)}

    @torch.no_grad()
    def forward(self,input_ids=None,attention_mask=None,position_ids=None,inputs_embeds=None,
                past_key_values=None,use_cache=None,**kwargs):
        if past_key_values is not None or use_cache:
            raise ValueError('training candidates cannot commit a runtime cache')
        if inputs_embeds is None:
            if input_ids is None: raise ValueError('token input required')
            inputs_embeds=self.embed_tokens(input_ids)
        b,t,_=inputs_embeds.shape
        valid=torch.ones((b,t),dtype=torch.bool,device=inputs_embeds.device) if attention_mask is None else attention_mask.bool()
        if valid.shape!=(b,t): raise ValueError('training expects padded2D attention masks')
        if position_ids is None: position_ids=torch.arange(t,device=inputs_embeds.device).expand(b,-1)
        if position_ids.shape[0]==1: position_ids=position_ids.expand(b,-1)
        if position_ids.shape!=(b,t): raise ValueError('position shape mismatch')
        global_masks=create_three_stream_masks(position_ids,valid,self.config.router_sink_size,self.config.router_window_size,'full')
        sliding_masks=create_three_stream_masks(position_ids,valid,0,self.config.sliding_window,'swa')
        embeddings={kind:self.rotary_emb(inputs_embeds,position_ids,kind) for kind in set(self.config.layer_types)}
        trunk=full=local=inputs_embeds
        equivalent=valid & position_ids.add(1).le(self.config.router_sink_size+self.config.router_window_size)
        for i,layer in enumerate(self.layers):
            kind=self.config.layer_types[i]
            masks=global_masks if kind=='full_attention' else (sliding_masks[0],sliding_masks[2],sliding_masks[2])
            attn=layer.self_attn
            q,k,v=project(attn,layer.input_layernorm(trunk),embeddings[kind])
            fq,fk,fv=project(attn,layer.input_layernorm(full),embeddings[kind])
            lq,lk,lv=project(attn,layer.input_layernorm(local),embeddings[kind])
            trunk_attention=run_flex(q,k,v,masks[0],'trunk',attn.num_key_value_groups)
            full_attention=candidate(attn,fq,k,v,fk,fv,masks[1],position_ids,valid,'full_candidate')
            local_attention=candidate(attn,lq,k,v,lk,lv,masks[2],position_ids,valid,'local_candidate')
            trunk=finish_layer(layer,trunk,trunk_attention,valid)
            full=finish_layer(layer,full,full_attention,valid)
            local=finish_layer(layer,local,local_attention,valid)
            # Full current action against native Full history is analytically the
            # trunk. Match v4.2's equivalent-stream canonicalization explicitly.
            full=trunk
            local=torch.where(equivalent[...,None],trunk,local)
            if self.config.router_check_finite and (not bool(trunk.isfinite().all()) or not bool(local.isfinite().all())):
                raise FloatingPointError(f'nonfinite Gemma stream at layer{i}')
        return self.norm(trunk),self.norm(full),self.norm(local),inputs_embeds,position_ids,valid


class _Gemma4TrainingBase(Gemma4UnifiedPreTrainedModel):
    """Reuse feature/objective plumbing, never a Qwen backbone implementation."""
    config_class=Gemma4UnifiedTextConfig
    _tied_weights_keys={'lm_head.weight':'model.embed_tokens.weight'}
    _checkpoint_conversion_mapping={r'^model\.language_model\.':'model.'}
    _get_router_loss_mask=Qwen3RouterForCausalLM._get_router_loss_mask
    _distributed_denominator=staticmethod(Qwen3RouterForCausalLM._distributed_denominator)
    _aligned_next_token_labels=staticmethod(Qwen3OneStepRouterForCausalLM._aligned_next_token_labels)
    _compute_one_step_loss=Qwen3OneStepRouterForCausalLM._compute_one_step_loss
    _build_three_stream_features=Qwen3ThreeStreamRouterForCausalLM._build_three_stream_features
    _history_selection_metrics=Qwen3ThreeStreamRouterForCausalLM._history_selection_metrics
    _select_threshold_history_source=Qwen3ThreeStreamRouterForCausalLM._select_threshold_history_source
    forward=Qwen3ThreeStreamRouterForCausalLM.forward

    def get_input_embeddings(self): return self.model.embed_tokens
    def get_output_embeddings(self): return self.lm_head

    @torch.no_grad()
    def _compute_token_nll_only(self,full,local,labels,valid,positions):
        aligned,mask=self._aligned_next_token_labels(labels,valid,positions)
        results=[]
        # Preserve native BF16 projection and softcap BEFORE FP32 logsumexp.
        for hidden in (full,local):
            chunks=[]
            for start in range(0,hidden.shape[1],self.config.router_kl_chunk_size):
                end=min(start+self.config.router_kl_chunk_size,hidden.shape[1])
                logits=self.lm_head(hidden[:,start:end])
                cap=self.config.final_logit_softcapping
                if cap is not None: logits=torch.tanh(logits/cap)*cap
                logits=logits.float()
                target=aligned[:,start:end].clamp_min(0).unsqueeze(-1)
                nll=torch.logsumexp(logits,-1)-logits.gather(-1,target).squeeze(-1)
                chunks.append(torch.where(mask[:,start:end],nll,torch.full_like(nll,float('nan'))))
            results.append(torch.cat(chunks,1))
        return tuple(results)


class Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM(RobustMarginObjectiveMixin,_Gemma4TrainingBase):
    def __init__(self,config):
        set_defaults(config)
        super().__init__(config)
        self.model=Gemma4UnifiedThreeStreamModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.router_head=_build_router_head(config)
        self.post_init()
        self.router_head.float()
        self.requires_grad_(False)
        self.router_head.requires_grad_(True)
        config.router_head_parameter_count=sum(p.numel() for p in self.router_head.parameters())
        config.router_head_tensor_count=len(self.router_head.state_dict())
        if not hasattr(config,'router_expected_head_parameter_count'):
            config.router_expected_head_parameter_count=config.router_head_parameter_count
        if not hasattr(config,'router_expected_head_tensor_count'):
            config.router_expected_head_tensor_count=22
        finish_contract(self)

    def validate_distributed_runtime_state(self):
        """Fail closed on unequal/nonfinite nonpersistent RoPE state across ranks.

        Invoked before/after DDP wrapping and after DCP restore by the trainer.
        This checks exact replica identity, not a floating-point approximation
        against a separately computed reference. It never repairs live values.
        """
        signature = []
        for name, buffer in self.model.rotary_emb.named_buffers():
            value = buffer.detach().contiguous().cpu()
            signature.append({
                'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                'finite': bool(value.isfinite().all()),
                'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            })
        if dist.is_available() and dist.is_initialized():
            signatures = [None] * dist.get_world_size()
            dist.all_gather_object(signatures, signature)
        else:
            signatures = [signature]
        if (not signature or any(not item['finite'] for sig in signatures for item in sig)
                or any(sig != signatures[0] for sig in signatures)):
            raise RuntimeError('Gemma RoPE buffers differ across ranks or contain nonfinite values')
        return signature

    def _apply(self,fn,recurse=True):
        super()._apply(fn,recurse=recurse)
        if hasattr(self,'router_head'): self.router_head.float()
        return self


ModelClass=Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM
