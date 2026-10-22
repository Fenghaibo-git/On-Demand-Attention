import pytest
pytest.importorskip("transformers.models.gemma4_unified")
from copy import deepcopy

import pytest
import torch
from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import apply_rotary_pos_emb

from veomni.models.transformers.gemma4_unified_three_stream_router.modeling_gemma4_unified_three_stream_router import (
    Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM as Router,
)


def tiny_config(layers=6):
    cfg=Gemma4UnifiedTextConfig(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,
        num_global_key_value_heads=1,head_dim=8,global_head_dim=16,
        attention_k_eq_v=True,layer_types=(['sliding_attention']*5+['full_attention'])*(layers//6),
        sliding_window=3,num_kv_shared_layers=0,max_position_embeddings=128,
        use_double_wide_mlp=False,use_bidirectional_attention='vision',
        hidden_size_per_layer_input=0,enable_moe_block=False,attention_dropout=0.,
        final_logit_softcapping=30.,tie_word_embeddings=True,
        rope_parameters={'full_attention':{'rope_type':'proportional','rope_theta':1e6,'partial_rotary_factor':.25},
                         'sliding_attention':{'rope_type':'default','rope_theta':10000.}})
    cfg._attn_implementation='eager'
    cfg.router_hidden_size=16
    cfg.router_intermediate_size=32
    cfg.router_num_blocks=2
    cfg.router_sink_size=1
    cfg.router_window_size=4
    cfg.router_return_hidden_states=True
    return cfg


def models(layers=6):
    torch.set_num_threads(1)
    torch.manual_seed(912)
    cfg=tiny_config(layers)
    native=Gemma4UnifiedForCausalLM(deepcopy(cfg)).eval()
    router=Router(cfg).eval()
    missing,unexpected=router.load_state_dict(native.state_dict(),strict=False)
    assert all(k.startswith('router_head.') for k in missing) and not unexpected
    return native,router


def test_native_full_parity_and_shared_lm_embedding():
    native,router=models()
    ids=torch.randint(3,97,(2,14))
    expected=native(ids,output_hidden_states=True,use_cache=False)
    actual=router(input_ids=ids,labels=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    torch.testing.assert_close(actual.trunk_hidden_states,expected.hidden_states[-1],rtol=3e-5,atol=3e-6)
    assert torch.equal(actual.trunk_hidden_states,actual.full_candidate_hidden_states)
    assert router.lm_head.weight is router.model.embed_tokens.weight
    target=ids[:,1:]
    logits=expected.logits[:,:-1].float()
    nll=torch.logsumexp(logits,-1)-logits.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual.full_token_nll[:,:-1],nll,rtol=3e-5,atol=3e-6)


def test_rope_buffer_order_preserves_native_values_and_nonpersistence():
    native,router=models()
    actual=dict(router.model.rotary_emb.named_buffers())
    expected=dict(native.model.rotary_emb.named_buffers())
    assert list(actual)==sorted(actual)
    assert actual.keys()==expected.keys()
    for name in expected:
        assert torch.equal(actual[name],expected[name])
    assert not router.model.rotary_emb.state_dict()
    assert len(router.validate_distributed_runtime_state())==len(actual)


def test_rope_guard_rejects_nonfinite_state_without_silently_repairing():
    _,router=models()
    buffer=router.model.rotary_emb.full_attention_inv_freq
    buffer[0]=float('nan')
    with pytest.raises(RuntimeError,match='RoPE buffers'):
        router.validate_distributed_runtime_state()
    assert torch.isnan(buffer[0])


@torch.no_grad()
def independent_local_reference(native,ids,router_cfg):
    """One current candidate against independently captured native Full inputs.

    Ordinary dense softmax concatenates only selected past K/V and its own
    current K/V; no production Flex mask/online-merge/finish helper is used.
    """
    observed=native.model(input_ids=ids,output_hidden_states=True,use_cache=False)
    b,t=ids.shape
    positions=torch.arange(t).expand(b,-1)
    embeddings=native.model.embed_tokens(ids)
    rope={kind:native.model.rotary_emb(embeddings,positions,kind) for kind in set(router_cfg.layer_types)}
    results=[]
    for j in range(t):
        current=embeddings[:,j:j+1]
        for i,layer in enumerate(native.model.layers):
            attn=layer.self_attn
            kind=router_cfg.layer_types[i]
            normalized=layer.input_layernorm(current)
            full_normalized=layer.input_layernorm(observed.hidden_states[i])
            past_indices=[k for k in range(j) if (
                k>=j-router_cfg.sliding_window+1 if kind=='sliding_attention'
                else k<router_cfg.router_sink_size or k>=j-router_cfg.router_window_size+1)]
            kv_inputs=torch.cat([full_normalized[:,past_indices],normalized],dim=1)
            actual_indices=past_indices+[j]
            cos,sin=rope[kind]
            q=attn.q_norm(attn.q_proj(normalized).view(b,1,-1,attn.head_dim))
            q=apply_rotary_pos_emb(q,cos[:,j:j+1],sin[:,j:j+1],unsqueeze_dim=2).transpose(1,2)
            k0=attn.k_proj(kv_inputs).view(b,len(actual_indices),-1,attn.head_dim)
            v0=attn.v_proj(kv_inputs).view_as(k0) if attn.v_proj is not None else k0
            k=apply_rotary_pos_emb(attn.k_norm(k0),cos[:,actual_indices],sin[:,actual_indices],unsqueeze_dim=2).transpose(1,2)
            v=attn.v_norm(v0).transpose(1,2)
            k=k.repeat_interleave(attn.num_key_value_groups,dim=1)
            v=v.repeat_interleave(attn.num_key_value_groups,dim=1)
            out=(q @ k.transpose(-1,-2)).float().softmax(-1).to(q.dtype) @ v
            out=attn.o_proj(out.transpose(1,2).reshape(b,1,-1))
            current=current+layer.post_attention_layernorm(out)
            current=current+layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(current)))
            current=current*layer.layer_scalar
        results.append(native.model.norm(current))
    return torch.cat(results,dim=1)


def test_isolated_local_candidate_matches_independent_dense_native_reference():
    native,router=models(12)
    ids=torch.randint(3,97,(1,13))
    actual=router(input_ids=ids,labels=ids,use_cache=False)
    expected=independent_local_reference(native,ids,router.config)
    torch.testing.assert_close(actual.swa_candidate_hidden_states,expected,rtol=5e-5,atol=5e-6)
    torch.testing.assert_close(actual.swa_candidate_hidden_states[:,:5],actual.trunk_hidden_states[:,:5],rtol=0,atol=0)


def test_only_fp32_head_trains_and_current_label_does_not_leak():
    _,router=models()
    ids=torch.randint(3,97,(1,12))
    features=[]
    hook=router.router_head.register_forward_pre_hook(lambda m,a:features.append(a[0].detach().clone()))
    first=router(input_ids=ids,labels=ids,use_cache=False)
    labels=ids.clone(); labels[:,9]=(labels[:,9]+1)%97
    router(input_ids=ids,labels=labels,use_cache=False)
    hook.remove()
    assert torch.equal(features[0][:,8],features[1][:,8])
    p=first.training_probe_tensors
    mask=p['valid_token_mask'].bool()
    gap=p['router_raw_signed_nll_gap']
    assert torch.equal(p['history_local_selected_mask'],(gap<0)&mask)
    assert torch.equal(p['router_full_target'],(gap>.001)&mask)
    first.loss.backward()
    assert all(p.grad is None for n,p in router.named_parameters() if not n.startswith('router_head.'))
    assert all(p.grad is not None and p.grad.dtype==torch.float32 and p.grad.isfinite().all() for p in router.router_head.parameters())


def test_new_native_model_rejects_unsupported_shared_kv_and_training_cache():
    cfg=tiny_config(); cfg.num_kv_shared_layers=2
    with pytest.raises(ValueError,match='shared-layer'):
        Router(cfg)
    _,router=models()
    ids=torch.ones(1,8,dtype=torch.long)
    with pytest.raises(ValueError,match='cache'):
        router(input_ids=ids,labels=ids,use_cache=True)


def test_a100_large_head_tile_preserves_mask_scale_and_gqa(monkeypatch):
    from veomni.models.transformers.gemma4_unified_three_stream_router import modeling_gemma4_unified_three_stream_router as implementation
    seen=[]
    monkeypatch.setattr(implementation,'flex_attention',lambda q,k,v,**kwargs:seen.append(kwargs))
    mask=object()
    for dim in (256,512):
        q=torch.zeros(1,2,3,dim)
        implementation.run_flex(q,q,q,mask,'cpu_test',2,True)
    assert all(v['block_mask'] is mask and v['scale']==1.0 and v['enable_gqa'] and v['return_lse'] for v in seen)
    assert seen[0]['kernel_options']=={'ROWS_GUARANTEED_SAFE':False}
    assert seen[1]['kernel_options']=={'ROWS_GUARANTEED_SAFE':False,'BLOCK_M':32,'BLOCK_N':32,'num_stages':1,'num_warps':4}


def test_cuda_flex_requires_fullgraph_and_scoped_cache_budget(monkeypatch):
    from types import SimpleNamespace
    from veomni.models.transformers.gemma4_unified_three_stream_router import modeling_gemma4_unified_three_stream_router as implementation
    compile_calls=[]
    previous=torch._dynamo.config.recompile_limit
    def compiled(*args,**kwargs):
        assert torch._dynamo.config.recompile_limit==64
        return 'compiled_only'
    def compile_spy(function,**kwargs):
        compile_calls.append(kwargs)
        return compiled
    monkeypatch.setattr(implementation,'_FLEX',{})
    monkeypatch.setattr(torch,'compile',compile_spy)
    query=SimpleNamespace(device=SimpleNamespace(type='cuda'),size=lambda dim:512)
    assert implementation.run_flex(query,query,query,object(),'trunk',16)=='compiled_only'
    assert compile_calls==[{'dynamic':True,'fullgraph':True}]
    assert torch._dynamo.config.recompile_limit==previous
