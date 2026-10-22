"""Production large-Qwen Huber + whole-step accumulation, without model weights."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from veomni.models.transformers.qwen3_large_robust_margin_router.modeling_qwen3_large_robust_margin_router import Qwen3LargeRobustMarginRouterForCausalLM
from veomni.models.transformers.qwen3_large_robust_margin_router.objective import ROBUST_DEFAULTS, objective_payload, objective_sha256
from veomni.utils.router_step_token_mean import RouterStepTokenMean, validate_step_token_mean_config


def config(arch='Qwen3LargeRobustMarginRouterForCausalLM'):
    return SimpleNamespace(**ROBUST_DEFAULTS, architectures=[arch], model_type='qwen3',
        router_schema_sha256='a'*64, router_one_step_sign_aux_temperature=.1,
        router_one_step_sign_near_zero_threshold=.0001,
        router_optimizer_step_reduction='global_eligible_token_mean_v1',
        router_optimization_protocol_sha256='b'*64)


@pytest.mark.parametrize('arch', ['Qwen3LargeRobustMarginRouterForCausalLM', 'Qwen3_5LargeRobustMarginRouterForCausalLM',
                                 'Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM'])
def test_large_architecture_requires_exact_objective_and_hash(arch):
    c = config(arch)
    args = SimpleNamespace(data_parallel_mode='ddp', ulysses_parallel_size=1, wandb_comparison_protocol_hash='b'*64)
    assert validate_step_token_mean_config(c, args)
    c.router_one_step_positive_weight = 2.
    with pytest.raises(ValueError):
        validate_step_token_mean_config(c, args)


def test_legacy_objective_identity_not_relabelled():
    c = config()
    new = objective_sha256(c)
    assert objective_payload(c)['optimizer_reduction'] == 'global_eligible_token_mean_v1'
    del c.router_optimizer_step_reduction
    assert objective_payload(c)['optimizer_reduction'] == 'mean_of_global_microbatch_means_as_matched_control'
    assert objective_sha256(c) != new


def test_unknown_architecture_remains_fail_closed():
    with pytest.raises(ValueError, match='qualified'):
        validate_step_token_mean_config(config('UnqualifiedForCausalLM'), SimpleNamespace())


@pytest.mark.parametrize('micro', [1, 2, 8, 64])
def test_large_huber_whole_step_gradient_clip_and_adam(micro):
    objective = object.__new__(Qwen3LargeRobustMarginRouterForCausalLM)
    torch.nn.Module.__init__(objective)
    objective.config = config()
    objective._last_robust_margin_metrics = {}
    torch.manual_seed(20260827)
    reference = torch.nn.Sequential(torch.nn.Linear(5, 12), torch.nn.SiLU(), torch.nn.Linear(12, 1))
    candidate = deepcopy(reference)
    x, gap = torch.randn(64, 7, 5), torch.randn(64, 7) * 3
    mask = torch.rand(64, 7) > .3
    mask[0] = False
    y = (gap-.001).sign() * torch.log1p((gap-.001).abs())
    expected = torch.nn.functional.huber_loss(reference(x).squeeze(-1)[mask], y[mask], delta=1.)
    expected.backward()
    chunks = [slice(i, i+micro) for i in range(0,64,micro)]
    plan = RouterStepTokenMean.from_counts([int(mask[c].sum()) for c in chunks], device='cpu')
    total = 0.
    for i,c in enumerate(chunks):
        loss, _ = objective._compute_one_step_loss(candidate(x[c]).squeeze(-1), gap[c], y[c], mask[c])
        scaled = plan.scale_loss(loss, mask[c], i)
        scaled.backward()
        total += float(scaled.detach())
    assert total == pytest.approx(float(expected.detach()), rel=2e-6, abs=2e-7)
    for p,q in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-7)
    for model in (reference,candidate):
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=0.).step()
    for p,q in zip(reference.parameters(),candidate.parameters()):
        torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-7)
