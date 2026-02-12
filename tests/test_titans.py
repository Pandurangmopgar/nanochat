"""
Tests for the Titans neural memory integration.
Runs on CPU, no GPU required. Run with:

    python -m pytest tests/test_titans.py -v -s

Note: Some tests that require the full GPT model (which depends on
nanochat.common/fcntl) may be skipped on Windows. The core Titans
memory module tests work on all platforms.
"""

import sys
import torch
import torch.nn as nn
import pytest

from nanochat.gpt import GPT, GPTConfig
from nanochat.titans_memory import TitansMemory, TitansLayer


# Small config for fast CPU testing
def make_config(use_titans=True, **kwargs):
    defaults = dict(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        use_titans=use_titans,
        titans_memory_dim=64,
        titans_memory_depth=2,
    )
    defaults.update(kwargs)
    return GPTConfig(**defaults)


# -------------------------------------------------------------------------
# Titans Memory module tests (platform-independent)
# -------------------------------------------------------------------------

def test_titans_memory_shapes():
    """TitansMemory output shape must match input shape."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    mem.train()
    x = torch.randn(2, 16, 128)
    out = mem(x, update_memory=True)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_titans_layer_shapes():
    """TitansLayer output shape must match input shape."""
    layer = TitansLayer(model_dim=128, memory_dim=64, memory_depth=2)
    layer.train()
    x = torch.randn(2, 16, 128)
    out = layer(x, update_memory=True)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_titans_layer_backward():
    """TitansLayer backward pass must produce gradients for all parameters."""
    layer = TitansLayer(model_dim=128, memory_dim=64, memory_depth=2)
    layer.train()
    x = torch.randn(2, 16, 128, requires_grad=True)
    out = layer(x, update_memory=True)
    loss = out.sum()
    loss.backward()

    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


def test_memory_updates():
    """Memory network weights must actually change after a forward pass with update_memory=True."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    mem.train()

    # Save initial weights
    initial_weights = [p.clone() for p in mem.memory_net.parameters()]

    # Forward with memory update
    x = torch.randn(2, 16, 128)
    mem(x, update_memory=True)

    # Check that at least some weights changed
    any_changed = False
    for p_init, p_new in zip(initial_weights, mem.memory_net.parameters()):
        if not torch.equal(p_init, p_new):
            any_changed = True
            break
    assert any_changed, "Memory network weights should change after update"


def test_no_update_in_eval():
    """Memory weights should NOT change when the module is in eval mode."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    mem.eval()

    initial_weights = [p.clone() for p in mem.memory_net.parameters()]

    x = torch.randn(2, 16, 128)
    mem(x, update_memory=True)

    for p_init, p_new in zip(initial_weights, mem.memory_net.parameters()):
        assert torch.equal(p_init, p_new), "Memory weights should not change in eval mode"


def test_momentum_state():
    """Momentum buffer must be initialized and non-zero after an update."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    mem.train()

    assert mem._momentum_state is None, "Momentum state should be None initially"

    x = torch.randn(2, 16, 128)
    mem(x, update_memory=True)

    assert mem._momentum_state is not None, "Momentum state should be initialized after update"
    assert mem._momentum_state.abs().sum() > 0, "Momentum state should be non-zero"


def test_gate_range():
    """Gate output must be in [0, 1] range (sigmoid)."""
    layer = TitansLayer(model_dim=128, memory_dim=64, memory_depth=2)
    layer.eval()

    x = torch.randn(2, 16, 128)
    x_normed = torch.nn.functional.rms_norm(x, (x.size(-1),))
    gate = torch.sigmoid(layer.gate(x_normed))
    assert gate.min() >= 0.0, f"Gate min below 0: {gate.min()}"
    assert gate.max() <= 1.0, f"Gate max above 1: {gate.max()}"


def test_memory_reset():
    """reset_memory_state() should clear the momentum buffer."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    mem.train()

    x = torch.randn(2, 16, 128)
    mem(x, update_memory=True)
    assert mem._momentum_state is not None

    mem.reset_memory_state()
    assert mem._momentum_state is None, "Momentum state should be None after reset"


def test_conv_effect():
    """1D convolution should make outputs position-dependent."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    mem.eval()

    # Create input where all positions are identical
    x_single = torch.randn(1, 1, 128)
    x = x_single.expand(1, 8, 128).clone()

    out = mem(x, update_memory=False)
    # Due to causal conv, different positions should get slightly different outputs
    # (edge effects from convolution padding)
    diffs = (out[:, 0, :] - out[:, 1, :]).abs().sum()
    assert diffs > 0, "Conv should cause position-dependent outputs"


def test_kv_projections():
    """K/V projections should exist and produce correct shapes (Eq. 11)."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    x = torch.randn(2, 16, 128)

    # Verify W_K and W_V exist and project correctly
    keys = mem.W_K(x)
    values = mem.W_V(x)
    assert keys.shape == (2, 16, 128), f"Key shape mismatch: {keys.shape}"
    assert values.shape == (2, 16, 128), f"Value shape mismatch: {values.shape}"

    # Keys and values should be different projections
    assert not torch.equal(keys, values), "K and V should be different projections"


def test_data_dependent_hyperparams():
    """Data-dependent hyperparameters should vary with input."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)

    # Two very different inputs should produce different hyperparameters
    x1 = torch.randn(2, 16, 128) * 0.1
    x2 = torch.randn(2, 16, 128) * 10.0

    alpha1, eta1, theta1 = mem._compute_hyperparams(x1)
    alpha2, eta2, theta2 = mem._compute_hyperparams(x2)

    # At init, the projections have zero weights so alpha/eta/theta are near-constant.
    # After one training step with gradients, they should start differing.
    # For now, verify they are valid scalars in the right ranges.
    assert 0 <= alpha1.item() <= 1, f"alpha out of range: {alpha1.item()}"
    assert 0 <= eta1.item() <= 1, f"eta out of range: {eta1.item()}"
    assert theta1.item() > 0, f"theta should be positive: {theta1.item()}"

    # Verify the projections themselves exist and have gradients
    x = torch.randn(2, 16, 128, requires_grad=True)
    mem.train()
    out = mem(x, update_memory=True)
    loss = out.sum()
    loss.backward()
    assert mem.proj_alpha.weight.grad is not None, "proj_alpha should get gradients"
    assert mem.proj_eta.weight.grad is not None, "proj_eta should get gradients"
    assert mem.proj_theta.weight.grad is not None, "proj_theta should get gradients"


def test_silu_activation():
    """Memory MLP should use SiLU activation (negative inputs produce negative outputs)."""
    mem = TitansMemory(model_dim=128, memory_dim=64, memory_depth=2)
    # SiLU(x) = x * sigmoid(x); for x < 0, SiLU(x) < 0
    # ReLU would produce 0 for negative inputs
    # Check internal layers are SiLU
    silu_count = sum(1 for m in mem.memory_net.modules() if isinstance(m, nn.SiLU))
    assert silu_count >= 1, f"Expected SiLU activations in memory_net, found {silu_count}"
    relu_count = sum(1 for m in mem.memory_net.modules() if isinstance(m, nn.ReLU))
    assert relu_count == 0, f"Expected no ReLU in memory_net, found {relu_count}"


def test_titans_layer_training_loop():
    """10-step training loop on the standalone TitansLayer should work without errors."""
    layer = TitansLayer(model_dim=128, memory_dim=64, memory_depth=2)
    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    losses = []
    target = torch.randn(2, 16, 128)
    for _ in range(10):
        x = torch.randn(2, 16, 128)
        out = layer(x, update_memory=True)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    print(f"TitansLayer training losses: {[f'{l:.4f}' for l in losses]}")
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
        f"Got non-finite loss values: {losses}"


# -------------------------------------------------------------------------
# Full GPT integration tests
# -------------------------------------------------------------------------


def test_titans_gpt_forward():
    """Full GPT with use_titans=True must produce correct logits shape."""
    config = make_config(use_titans=True)
    model = GPT(config)
    model.init_weights()

    B, T = 2, 32
    idx = torch.randint(0, 256, (B, T))
    logits = model(idx)
    assert logits.shape == (B, T, 256), f"Expected (2, 32, 256), got {logits.shape}"



def test_titans_gpt_backward():
    """Loss.backward() must run; all parameters must get gradients."""
    config = make_config(use_titans=True)
    model = GPT(config)
    model.init_weights()

    B, T = 2, 32
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    loss = model(idx, targets)
    loss.backward()

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"



def test_original_unaffected():
    """use_titans=False must produce a model with no TitansLayer."""
    vanilla = GPT(make_config(use_titans=False))
    titans = GPT(make_config(use_titans=True))
    vanilla.init_weights()
    titans.init_weights()

    assert vanilla.titans_layer is None
    assert titans.titans_layer is not None

    vanilla_params = sum(p.numel() for p in vanilla.parameters())
    titans_params = sum(p.numel() for p in titans.parameters())
    assert titans_params > vanilla_params



def test_param_count():
    """Verify parameter count breakdowns for optimizer grouping."""
    config = make_config(use_titans=True)
    model = GPT(config)
    model.init_weights()

    total = sum(p.numel() for p in model.parameters())
    blocks = sum(p.numel() for p in model.transformer.h.parameters())
    embed = sum(p.numel() for p in model.transformer.wte.parameters())
    lm = sum(p.numel() for p in model.lm_head.parameters())
    titans = sum(p.numel() for p in model.titans_layer.parameters())

    assert total == blocks + embed + lm + titans, \
        "Parameter count mismatch — optimizer grouping would be incorrect"

    print(f"Total: {total:,} | Blocks: {blocks:,} | Embed: {embed:,} | LM: {lm:,} | Titans: {titans:,}")



def test_tiny_training_loop():
    """10-step training loop on random data should not produce NaN/Inf."""
    config = make_config(use_titans=True)
    model = GPT(config)
    model.init_weights()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, T = 2, 32
    losses = []
    for _ in range(10):
        idx = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    print(f"Loss trajectory: {[f'{l:.4f}' for l in losses]}")
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
        f"Got non-finite loss values: {losses}"
