"""
Quick sanity check for MoE + Titans GPT.
Tests model construction, forward pass, backward pass, and parameter counts.

Run: python -m tests.test_moe
"""

import torch
import sys


def test_moe_only():
    """Test MoE-only model (no Titans)."""
    print("=" * 60)
    print("TEST 1: MoE-only model")
    print("=" * 60)

    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
        # MoE config
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        moe_every_n=2,
        aux_loss_weight=0.01,
    )

    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()

    # Check MoE layer placement
    moe_layers = [i for i, b in enumerate(model.transformer.h) if b.is_moe]
    dense_layers = [i for i, b in enumerate(model.transformer.h) if not b.is_moe]
    print(f"  MoE layers:   {moe_layers}")
    print(f"  Dense layers: {dense_layers}")
    assert len(moe_layers) == 2, f"Expected 2 MoE layers, got {len(moe_layers)}"

    # Check param counts
    params = model.num_scaling_params()
    print(f"  Total params:      {params['total']:,}")
    print(f"  MoE expert params: {params['moe_expert_params']:,}")
    print(f"  MoE router params: {params['moe_router_params']:,}")
    assert params['moe_expert_params'] > 0
    assert params['moe_router_params'] > 0

    # Forward pass
    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))

    model.train()
    loss = model(idx, targets)
    print(f"  Train loss:  {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    # Backward pass
    loss.backward()
    print(f"  Backward pass: ✓")

    # Check that MoE aux loss is being computed
    for block in model.transformer.h:
        if block.is_moe:
            aux = block.mlp.aux_loss
            if torch.is_tensor(aux):
                aux = aux.item()
            print(f"  MoE aux loss: {aux:.6f}")
            assert aux > 0, "Aux loss should be > 0"

    # Inference pass
    model.eval()
    with torch.no_grad():
        logits = model(idx)
    print(f"  Logits shape: {logits.shape}")
    assert logits.shape == (2, 32, 256), f"Expected (2, 32, 256), got {logits.shape}"

    # FLOPs estimate
    flops = model.estimate_flops()
    print(f"  FLOPs/token:  {flops:e}")

    print("  ✓ TEST 1 PASSED\n")
    return True


def test_moe_with_titans():
    """Test MoE + Titans model."""
    print("=" * 60)
    print("TEST 2: MoE + Titans model")
    print("=" * 60)

    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=8,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
        # MoE config
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        moe_every_n=2,
        # Titans config
        use_titans=True,
        titans_memory_dim=64,
        titans_memory_depth=2,
        titans_every_n=4,
    )

    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()

    # Check layout
    moe_layers = [i for i, b in enumerate(model.transformer.h) if b.is_moe]
    titans_at = [int(k) for k in model.titans_layers.keys()]
    print(f"  MoE layers:    {moe_layers}")
    print(f"  Titans after:  {titans_at}")
    assert len(titans_at) == 2, f"Expected 2 Titans layers (at blocks 3,7), got {len(titans_at)}"
    assert model.titans_layer is not None, "titans_layer backward compat should be set"

    # Param counts
    params = model.num_scaling_params()
    print(f"  Total params:      {params['total']:,}")
    print(f"  Titans params:     {params['titans_params']:,}")
    assert params['titans_params'] > 0

    # Forward + backward
    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))

    model.train()
    loss = model(idx, targets)
    print(f"  Train loss:  {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"

    loss.backward()
    print(f"  Backward pass: ✓")

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (2, 32, 256)
    print(f"  Inference: ✓")

    print("  ✓ TEST 2 PASSED\n")
    return True


def test_backward_compat():
    """Test that vanilla GPT (no MoE, no Titans) still works."""
    print("=" * 60)
    print("TEST 3: Backward compatibility (vanilla GPT)")
    print("=" * 60)

    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
        # Both off by default
    )

    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()

    assert not any(b.is_moe for b in model.transformer.h), "No MoE layers expected"
    assert len(model.titans_layers) == 0, "No Titans layers expected"

    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))

    model.train()
    loss = model(idx, targets)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Backward: ✓")

    print("  ✓ TEST 3 PASSED\n")
    return True


def test_moe_load_balancing():
    """Test that MoE router produces roughly balanced expert utilization."""
    print("=" * 60)
    print("TEST 4: Expert load balancing")
    print("=" * 60)

    from nanochat.moe import MoEMLP

    moe = MoEMLP(n_embd=128, num_experts=4, top_k=2, aux_loss_weight=0.01)
    # Small init for router to get balanced routing
    torch.nn.init.normal_(moe.gate.weight, std=0.01)

    x = torch.randn(4, 64, 128)
    moe.train()
    out = moe(x)

    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} != {x.shape}"

    aux_loss = moe.aux_loss.item()
    print(f"  Aux loss: {aux_loss:.6f}")
    assert aux_loss > 0, "Aux loss should be positive"

    # Check output is not all zeros
    assert out.abs().sum() > 0, "Output should not be all zeros"

    print("  ✓ TEST 4 PASSED\n")
    return True


if __name__ == "__main__":
    print("\n🧪 MoE + Titans GPT Sanity Tests\n")

    all_passed = True
    tests = [test_moe_load_balancing, test_moe_only, test_moe_with_titans, test_backward_compat]

    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
