"""
Tests for multi-timescale ContinuumMemory.

Run: python -m tests.test_continuum_memory
"""

import torch
import sys


def test_continuum_memory_standalone():
    """Test ContinuumMemory in isolation."""
    print("=" * 60)
    print("TEST 1: ContinuumMemory standalone")
    print("=" * 60)

    from nanochat.titans_memory import ContinuumMemory

    mem = ContinuumMemory(
        model_dim=128,
        memory_dim=192,  # will be split: 192 // 3 = 64 per timescale
        memory_depth=2,
        num_timescales=3,
        update_intervals=(1, 4, 16),
    )

    print(f"  Num timescales: {mem.num_timescales}")
    print(f"  Update intervals: {mem.update_intervals}")
    print(f"  Memories: {len(mem.memories)}")
    num_params = sum(p.numel() for p in mem.parameters())
    print(f"  Total params: {num_params:,}")

    # Forward pass
    x = torch.randn(2, 32, 128)
    mem.train()
    out = mem(x, update_memory=True)
    print(f"  Output shape: {out.shape}")
    assert out.shape == (2, 32, 128), f"Shape mismatch: {out.shape}"

    # Check step counter incremented
    assert mem._step_count == 1, f"Expected step_count=1, got {mem._step_count}"

    # Run a few more steps to test medium/slow timescale activation
    for _ in range(16):
        out = mem(x, update_memory=True)
    assert mem._step_count == 17

    # Reset
    mem.reset_memory_state()
    assert mem._step_count == 0
    for m in mem.memories:
        assert m._momentum_state is None

    print("  ✓ TEST 1 PASSED\n")
    return True


def test_continuum_titans_layer():
    """Test ContinuumTitansLayer as drop-in replacement for TitansLayer."""
    print("=" * 60)
    print("TEST 2: ContinuumTitansLayer (drop-in for TitansLayer)")
    print("=" * 60)

    from nanochat.titans_memory import ContinuumTitansLayer

    layer = ContinuumTitansLayer(
        model_dim=128,
        memory_dim=192,
        memory_depth=2,
        num_timescales=3,
        update_intervals=(1, 4, 16),
    )

    x = torch.randn(2, 32, 128)
    layer.train()
    out = layer(x, update_memory=True)
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape

    # Backward pass
    loss = out.sum()
    loss.backward()
    print(f"  Backward: ✓")

    # Check gradients flow through gate
    assert layer.gate.weight.grad is not None
    print(f"  Gate gradients: ✓")

    # Reset
    layer.reset_memory_state()
    print(f"  Reset: ✓")

    print("  ✓ TEST 2 PASSED\n")
    return True


def test_gpt_with_continuum_memory():
    """Test GPT with MoE + ContinuumTitansLayer."""
    print("=" * 60)
    print("TEST 3: GPT with MoE + Continuum Memory")
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
        # MoE
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        moe_every_n=2,
        # Continuum Titans
        use_titans=True,
        titans_memory_dim=96,
        titans_memory_depth=2,
        titans_every_n=4,
        titans_continuum=True,
        titans_num_timescales=3,
        titans_update_intervals=(1, 4, 16),
    )

    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()

    # Check layer types
    titans_at = [int(k) for k in model.titans_layers.keys()]
    print(f"  Continuum Titans after blocks: {titans_at}")
    assert len(titans_at) == 2  # blocks 3 and 7

    # Verify it's ContinuumTitansLayer, not TitansLayer
    from nanochat.titans_memory import ContinuumTitansLayer
    for idx in titans_at:
        assert isinstance(model.titans_layers[str(idx)], ContinuumTitansLayer), \
            f"Expected ContinuumTitansLayer at block {idx}"
    print(f"  Layer type: ContinuumTitansLayer ✓")

    # Param counts
    params = model.num_scaling_params()
    print(f"  Total params: {params['total']:,}")
    print(f"  Titans params: {params['titans_params']:,}")
    assert params['titans_params'] > 0

    # Forward + backward
    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))

    model.train()
    loss = model(idx, targets)
    print(f"  Loss: {loss.item():.4f}")
    assert not torch.isnan(loss)

    loss.backward()
    print(f"  Backward: ✓")

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (2, 32, 256)
    print(f"  Inference: ✓")

    print("  ✓ TEST 3 PASSED\n")
    return True


def test_single_vs_continuum_param_comparison():
    """Compare param counts between single and continuum memory."""
    print("=" * 60)
    print("TEST 4: Single vs Continuum param comparison")
    print("=" * 60)

    from nanochat.titans_memory import TitansLayer, ContinuumTitansLayer

    single = TitansLayer(model_dim=128, memory_dim=192, memory_depth=2)
    continuum = ContinuumTitansLayer(model_dim=128, memory_dim=192, memory_depth=2,
                                     num_timescales=3, update_intervals=(1, 4, 16))

    single_params = sum(p.numel() for p in single.parameters())
    continuum_params = sum(p.numel() for p in continuum.parameters())
    ratio = continuum_params / single_params

    print(f"  Single TitansLayer params:    {single_params:,}")
    print(f"  ContinuumTitansLayer params:  {continuum_params:,}")
    print(f"  Ratio (continuum/single):     {ratio:.2f}x")

    # Continuum should be larger (3 memories) but not 3x due to dim scaling
    assert ratio > 1.0, "Continuum should have more params"
    assert ratio < 3.5, "Continuum shouldn't be more than 3.5x params"

    print("  ✓ TEST 4 PASSED\n")
    return True


if __name__ == "__main__":
    print("\n🧪 Multi-Timescale ContinuumMemory Tests\n")

    all_passed = True
    tests = [
        test_continuum_memory_standalone,
        test_continuum_titans_layer,
        test_gpt_with_continuum_memory,
        test_single_vs_continuum_param_comparison,
    ]

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
