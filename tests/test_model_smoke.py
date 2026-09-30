"""Testes de fumaça do modelo: rodam em CPU, em segundos, com uma configuração minúscula."""

import pytest
import torch

from model.model_lora import apply_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

VOCAB = 128


def tiny_model(**overrides):
    kwargs = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": VOCAB,
        "max_position_embeddings": 256,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return MiniMindForCausalLM(MiniMindConfig(**kwargs))


def random_ids(batch=2, length=16):
    return torch.randint(0, VOCAB, (batch, length), generator=torch.Generator().manual_seed(1))


def test_forward_returns_logits_and_trainable_loss():
    model = tiny_model()
    ids = random_ids()
    out = model(ids, labels=ids)
    assert out.logits.shape == (*ids.shape, VOCAB)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.lm_head.weight.grad is not None


def test_kv_cache_matches_full_forward():
    model = tiny_model().eval()
    ids = random_ids(batch=1)
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        prefix = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(step.logits[:, -1], full, atol=1e-4, rtol=1e-4)


def test_greedy_generate_is_deterministic():
    model = tiny_model().eval()
    ids = random_ids(batch=1, length=4)
    first = model.generate(ids, max_new_tokens=6, do_sample=False, eos_token_id=None)
    second = model.generate(ids, max_new_tokens=6, do_sample=False, eos_token_id=None)
    assert first.shape == (1, 10)
    assert torch.equal(first, second)


@pytest.mark.parametrize("experts_per_token", [1, 2])
def test_moe_forward_reports_aux_loss(experts_per_token):
    model = tiny_model(use_moe=True, num_experts=4, num_experts_per_tok=experts_per_token).train()
    ids = random_ids()
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss)
    assert out.aux_loss.item() > 0


def test_lora_starts_as_identity():
    model = tiny_model().eval()
    ids = random_ids(batch=1)
    with torch.no_grad():
        before = model(ids).logits
        apply_lora(model, rank=4)
        after = model(ids).logits
    assert any(hasattr(module, "lora") for module in model.modules())
    torch.testing.assert_close(after, before)
