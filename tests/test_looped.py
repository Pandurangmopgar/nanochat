"""
Tests for the Looped Transformer architecture.

Verifies all components work correctly on CPU before any GPU training.
Run with: uv run python -m pytest tests/test_looped.py -v
"""

import pytest
import torch
import math

from nanochat.looped_transformer import (
    SandwichBlock,
    SandwichSelfAttention,
    SandwichMLP,
    ConcatAdapter,
    LoRAAdapter,
    MoDRouter,
    RecurrentCore,
    sandwich_norm,
    kl_divergence_early_exit,
)
from nanochat.looped_gpt import LoopedGPT, LoopedGPTConfig


# Test dimensions
B, T, D = 2, 32, 64  # small batch, sequence, embedding for CPU tests
N_HEAD = 4
N_KV_HEAD = 2


# ---------------------------------------------------------------------------
# Component Tests
# ---------------------------------------------------------------------------

class TestSandwichNorm:
    def test_output_shape(self):
        x = torch.randn(B, T, D)
        y = sandwich_norm(x)
        assert y.shape == x.shape

    def test_unit_rms(self):
        x = torch.randn(B, T, D)
        y = sandwich_norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestSandwichBlock:
    def test_output_shape(self):
        block = SandwichBlock(D, N_HEAD, N_KV_HEAD)
        x = torch.randn(B, T, D)
        cos = torch.randn(1, T, 1, D // N_HEAD // 2)
        sin = torch.randn(1, T, 1, D // N_HEAD // 2)
        y = block(x, (cos, sin))
        assert y.shape == (B, T, D)

    def test_gradient_flow(self):
        block = SandwichBlock(D, N_HEAD, N_KV_HEAD)
        x = torch.randn(B, T, D, requires_grad=True)
        cos = torch.randn(1, T, 1, D // N_HEAD // 2)
        sin = torch.randn(1, T, 1, D // N_HEAD // 2)
        y = block(x, (cos, sin))
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (residual adds something)."""
        block = SandwichBlock(D, N_HEAD, N_KV_HEAD)
        x = torch.randn(B, T, D)
        cos = torch.randn(1, T, 1, D // N_HEAD // 2)
        sin = torch.randn(1, T, 1, D // N_HEAD // 2)
        y = block(x, (cos, sin))
        assert not torch.allclose(x, y, atol=1e-6)


class TestConcatAdapter:
    def test_output_shape(self):
        adapter = ConcatAdapter(D)
        state = torch.randn(B, T, D)
        emb = torch.randn(B, T, D)
        y = adapter(state, emb)
        assert y.shape == (B, T, D)

    def test_param_count(self):
        adapter = ConcatAdapter(D)
        # Linear(2D, D) = 2D*D params (no bias)
        assert sum(p.numel() for p in adapter.parameters()) == 2 * D * D


class TestLoRAAdapter:
    def test_output_shape(self):
        lora = LoRAAdapter(D, rank=8)
        x = torch.randn(B, T, D)
        y = lora(x)
        assert y.shape == (B, T, D)

    def test_small_initial_output(self):
        """LoRA output should be small (scale * up(down(x)))."""
        lora = LoRAAdapter(D, rank=8, scale=0.1)
        # Init up to zeros for near-zero output
        torch.nn.init.zeros_(lora.up.weight)
        x = torch.randn(B, T, D)
        y = lora(x)
        assert y.abs().max() < 1e-6

    def test_param_count(self):
        rank = 8
        lora = LoRAAdapter(D, rank=rank)
        # down: D*rank + up: rank*D = 2*D*rank
        assert sum(p.numel() for p in lora.parameters()) == 2 * D * rank


class TestMoDRouter:
    def test_output_shapes(self):
        router = MoDRouter(D)
        x = torch.randn(B, T, D)
        scores, mask, indices = router(x, capacity_ratio=0.5)
        assert scores.shape == (B, T, 1)
        assert mask.shape == (B, T, 1)
        K = max(1, int(T * 0.5))
        assert indices.shape == (B, K)

    def test_mask_sum(self):
        """Mask should select exactly K tokens."""
        router = MoDRouter(D)
        x = torch.randn(B, T, D)
        capacity = 0.5
        _, mask, _ = router(x, capacity_ratio=capacity)
        K = max(1, int(T * capacity))
        assert mask.sum(dim=1).squeeze().allclose(torch.tensor(float(K)))

    def test_full_capacity(self):
        """At capacity=1.0, all tokens selected."""
        router = MoDRouter(D)
        x = torch.randn(B, T, D)
        _, mask, indices = router(x, capacity_ratio=1.0)
        assert mask.sum(dim=1).squeeze().allclose(torch.tensor(float(T)))


class TestRecurrentCore:
    def _make_core(self, **kwargs):
        defaults = dict(
            n_embd=D, n_head=N_HEAD, n_kv_head=N_KV_HEAD,
            n_recurrent_layers=2, max_recurrence=4,
            mean_recurrence=2, backprop_depth=2,
            state_init_std=0.02, use_lora=True, lora_rank=8,
            use_mod=True,
        )
        defaults.update(kwargs)
        return RecurrentCore(**defaults)

    def _make_cos_sin(self):
        head_dim = D // N_HEAD
        cos = torch.randn(1, T, 1, head_dim // 2)
        sin = torch.randn(1, T, 1, head_dim // 2)
        return cos, sin

    def test_output_shape(self):
        core = self._make_core()
        emb = torch.randn(B, T, D)
        state, info = core(emb, self._make_cos_sin(), training=True)
        assert state.shape == (B, T, D)

    def test_info_dict(self):
        core = self._make_core()
        emb = torch.randn(B, T, D)
        _, info = core(emb, self._make_cos_sin(), training=True)
        assert 'n_steps' in info
        assert info['n_steps'] >= 1
        assert info['n_steps'] <= 4  # max_recurrence

    def test_gradient_flow(self):
        core = self._make_core()
        emb = torch.randn(B, T, D, requires_grad=True)
        state, _ = core(emb, self._make_cos_sin(), training=True)
        loss = state.sum()
        loss.backward()
        assert emb.grad is not None

    def test_no_lora_no_mod(self):
        """Should work with both LoRA and MoD disabled."""
        core = self._make_core(use_lora=False, use_mod=False)
        emb = torch.randn(B, T, D)
        state, info = core(emb, self._make_cos_sin(), training=True)
        assert state.shape == (B, T, D)

    def test_deterministic_eval(self):
        """In eval mode, should use mean_recurrence consistently."""
        core = self._make_core(mean_recurrence=3)
        emb = torch.randn(B, T, D)
        _, info1 = core(emb, self._make_cos_sin(), training=False)
        _, info2 = core(emb, self._make_cos_sin(), training=False)
        assert info1['n_steps'] == 3
        assert info2['n_steps'] == 3


# ---------------------------------------------------------------------------
# Full Model Tests
# ---------------------------------------------------------------------------

class TestLoopedGPT:
    def _make_config(self, **kwargs):
        defaults = dict(
            vocab_size=256, sequence_len=64, n_embd=D,
            n_head=N_HEAD, n_kv_head=N_KV_HEAD,
            n_prelude=1, n_recurrent=2, n_coda=1,
            mean_recurrence=2, max_recurrence=4,
            backprop_depth=2, state_init_std=0.02,
            use_lora=True, lora_rank=8, use_mod=True,
            embedding_scale=1.0,
        )
        defaults.update(kwargs)
        return LoopedGPTConfig(**defaults)

    def _make_model(self, **kwargs):
        config = self._make_config(**kwargs)
        # Use meta device then init
        model = LoopedGPT(config, pad_vocab_size_to=16)
        model.init_weights()
        return model

    def test_forward_shape(self):
        model = self._make_model()
        idx = torch.randint(0, 256, (B, T))
        logits = model(idx)
        assert logits.shape == (B, T, 256)

    def test_forward_loss(self):
        model = self._make_model()
        idx = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        loss = model(idx, targets=targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_backward(self):
        model = self._make_model()
        idx = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        loss = model(idx, targets=targets)
        loss.backward()
        # Check gradients exist for key parameters
        assert model.wte.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_generate(self):
        model = self._make_model()
        model.eval()
        tokens = [1, 2, 3]
        generated = list(model.generate(tokens, max_tokens=5, temperature=1.0, top_k=10))
        assert len(generated) == 5
        assert all(isinstance(t, int) for t in generated)
        assert all(0 <= t < 256 for t in generated)

    def test_param_count(self):
        model = self._make_model()
        counts = model.num_params()
        assert counts['total'] == sum(p.numel() for p in model.parameters())
        assert counts['embedding'] > 0
        assert counts['prelude'] > 0
        assert counts['recurrent_block'] > 0
        assert counts['coda'] > 0
        assert counts['lm_head'] > 0
        # LoRA and MoD should be non-zero when enabled
        assert counts['lora'] > 0
        assert counts['mod'] > 0

    def test_no_lora_no_mod(self):
        model = self._make_model(use_lora=False, use_mod=False)
        counts = model.num_params()
        assert counts['lora'] == 0
        assert counts['mod'] == 0
        # Should still work
        idx = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        loss = model(idx, targets=targets)
        assert not torch.isnan(loss)

    def test_training_loss_decreases(self):
        """Sanity check: loss should decrease over a few training steps."""
        torch.manual_seed(42)
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            idx = torch.randint(0, 256, (B, T))
            targets = torch.randint(0, 256, (B, T))
            loss = model(idx, targets=targets)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (first > last)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_param_efficiency(self):
        """LoopedGPT should have fewer params than equivalent depth GPT."""
        config = self._make_config()
        model = self._make_model()
        looped_params = sum(p.numel() for p in model.parameters())

        # Equivalent standard GPT would have (n_prelude + n_recurrent*mean_recurrence + n_coda) layers
        equiv_layers = config.n_prelude + config.n_recurrent * config.mean_recurrence + config.n_coda
        # Each layer has: attn(4*D*D) + MLP(4*D*D + D*4D) ≈ 8*D*D params
        approx_standard_params = equiv_layers * 8 * D * D + 256 * D * 2  # + embedding + lm_head

        print(f"\n  LoopedGPT params: {looped_params:,}")
        print(f"  Equivalent standard params (approx): {approx_standard_params:,}")
        print(f"  Ratio: {looped_params / approx_standard_params:.2f}x")
        assert looped_params < approx_standard_params


# ---------------------------------------------------------------------------
# KL Divergence Early Exit
# ---------------------------------------------------------------------------

class TestKLDivergenceExit:
    def test_identical_logits_converge(self):
        logits = torch.randn(B, T, 256)
        assert kl_divergence_early_exit(logits, logits, threshold=5e-4)

    def test_different_logits_no_converge(self):
        logits1 = torch.randn(B, T, 256)
        logits2 = torch.randn(B, T, 256)
        assert not kl_divergence_early_exit(logits1, logits2, threshold=5e-4)

    def test_similar_logits_converge(self):
        logits1 = torch.randn(B, T, 256) * 10  # sharp distribution
        logits2 = logits1 + torch.randn_like(logits1) * 0.001  # tiny perturbation
        assert kl_divergence_early_exit(logits1, logits2, threshold=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
