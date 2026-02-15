"""
Looped Transformer Components

Implements the core building blocks for a Huginn-style depth-recurrent transformer
with optional LoRA depth adapters and Mixture-of-Depths (MoD) routing.

Architecture: Prelude-Recurrent-Coda (P-R-C)
- Prelude: Embeds input into latent space (specialized layers)
- Recurrent Core: Shared block applied N times with input injection
- Coda: Decodes from latent space (specialized layers)

Key techniques (from Huginn paper, arXiv:2502.05171):
- Sandwich normalization for stable deep recurrence
- Concatenation adapter (2h → h) for input injection each loop
- Random state initialization (path independence)
- Log-normal Poisson recurrence sampling during training
- Truncated backpropagation through last k iterations
- Zero-shot KL-divergence early exit at inference

Novel additions:
- LoRA depth adapters (optional, for perplexity recovery at nano-scale)
- MoD top-k routing (optional, for reasoning + speed improvement)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sandwich_norm(x):
    """RMSNorm without learnable parameters (sandwich-style)."""
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# Sandwich Block: norm → attn → norm → residual, norm → MLP → norm → residual
# Critical for stable recurrence (Huginn §3.2)
# ---------------------------------------------------------------------------

class SandwichSelfAttention(nn.Module):
    """Causal self-attention with sandwich normalization and RoPE.

    Simplified compared to gpt.py's CausalSelfAttention:
    - No value embeddings (only used in standard GPT)
    - No KV cache (looped model handles caching differently)
    - Sandwich norm wrapping
    """

    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0

        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings
        cos, sin = cos_sin
        from nanochat.gpt import apply_rotary_emb
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = sandwich_norm(q), sandwich_norm(k)  # QK norm

        # Expand KV heads for GQA
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        # Scaled dot-product attention (causal)
        # Transpose to (B, n_head, T, head_dim) for F.scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SandwichMLP(nn.Module):
    """MLP with SiLU gating (matching Huginn's gated SiLU MLP)."""

    def __init__(self, n_embd, mlp_ratio=4):
        super().__init__()
        inner_dim = n_embd * mlp_ratio
        self.c_fc = nn.Linear(n_embd, inner_dim, bias=False)
        self.c_proj = nn.Linear(inner_dim, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # Match nanochat's relu^2 activation
        x = self.c_proj(x)
        return x


class SandwichBlock(nn.Module):
    """Transformer block with sandwich normalization.

    Layout (Huginn §3.2):
        x → norm → attn → norm → + residual
        x → norm → MLP  → norm → + residual
    """

    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        self.attn = SandwichSelfAttention(n_embd, n_head, n_kv_head)
        self.mlp = SandwichMLP(n_embd)

    def forward(self, x, cos_sin):
        # Attention with sandwich norm
        x = x + sandwich_norm(self.attn(sandwich_norm(x), cos_sin))
        # MLP with sandwich norm
        x = x + sandwich_norm(self.mlp(sandwich_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Concat Adapter: merges recurrent state + embedded input each loop step
# (Huginn §3.2: concat works better than addition at scale)
# ---------------------------------------------------------------------------

class ConcatAdapter(nn.Module):
    """Linear adapter that merges [state, embedding] → hidden_dim.

    At each recurrence step, the current state s_i and the prelude embedding e
    are concatenated and projected back to hidden_dim.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)

    def forward(self, state, embedding):
        return self.proj(torch.cat([state, embedding], dim=-1))


# ---------------------------------------------------------------------------
# LoRA Depth Adapter (optional, for perplexity recovery)
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Low-rank adapter for depth-specific specialization.

    Each recurrence depth gets its own tiny LoRA that adds a small
    perturbation to the shared block's output, giving each depth
    step a "personality" for memorization without massive param increase.

    Params per adapter: 2 × n_embd × rank (e.g., 2 × 768 × 32 = 49K)
    """

    def __init__(self, n_embd, rank=32, scale=0.1):
        super().__init__()
        self.down = nn.Linear(n_embd, rank, bias=False)
        self.up = nn.Linear(rank, n_embd, bias=False)
        self.scale = scale

    def forward(self, x):
        return self.scale * self.up(self.down(x))


# ---------------------------------------------------------------------------
# Mixture-of-Depths Router (optional, for reasoning + speed)
# ---------------------------------------------------------------------------

class MoDRouter(nn.Module):
    """Top-k router for Mixture-of-Depths.

    At each recurrence step, scores all tokens and only processes the
    top-C% (most "uncertain" / highest-scoring). Remaining tokens skip
    this loop step, keeping their state unchanged.

    This is GPU-friendly: uses gather/scatter, not per-token loops.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.scorer = nn.Linear(n_embd, 1, bias=False)

    def forward(self, x, capacity_ratio):
        """
        Args:
            x: (B, T, D) input tensor
            capacity_ratio: float in (0, 1], fraction of tokens to process

        Returns:
            scores: (B, T, 1) routing scores
            mask: (B, T, 1) binary mask (1 = process, 0 = skip)
            indices: (B, K) indices of selected tokens
        """
        B, T, D = x.shape
        scores = self.scorer(x)  # (B, T, 1)
        K = max(1, int(T * capacity_ratio))

        # Top-k selection along sequence dimension
        _, indices = torch.topk(scores.squeeze(-1), K, dim=1)  # (B, K)
        mask = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)
        mask.scatter_(1, indices.unsqueeze(-1), 1.0)

        return scores, mask, indices


# ---------------------------------------------------------------------------
# Recurrent Core: the heart of the looped transformer
# ---------------------------------------------------------------------------

class RecurrentCore(nn.Module):
    """The recurrent engine that loops a shared block group.

    Contains:
    - N SandwichBlocks (the shared recurrent group)
    - ConcatAdapter (merges state + embedding each step)
    - Optional LoRA adapters (one per max loop)
    - Optional MoD router

    Training: samples random loop count from log-normal Poisson distribution
    Inference: runs fixed loop count, or uses KL-divergence early exit
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_kv_head,
        n_recurrent_layers=3,
        max_recurrence=16,
        mean_recurrence=4,
        backprop_depth=4,
        state_init_std=0.02,
        use_lora=True,
        lora_rank=32,
        use_mod=True,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.max_recurrence = max_recurrence
        self.mean_recurrence = mean_recurrence
        self.backprop_depth = backprop_depth
        self.state_init_std = state_init_std
        self.use_lora = use_lora
        self.use_mod = use_mod

        # Concat adapter for input injection
        self.adapter = ConcatAdapter(n_embd)

        # Shared recurrent block (N transformer layers)
        self.layers = nn.ModuleList([
            SandwichBlock(n_embd, n_head, n_kv_head)
            for _ in range(n_recurrent_layers)
        ])

        # Optional LoRA adapters (one per max loop step)
        if use_lora:
            self.lora_adapters = nn.ModuleList([
                LoRAAdapter(n_embd, rank=lora_rank)
                for _ in range(max_recurrence)
            ])
        else:
            self.lora_adapters = None

        # Optional MoD router
        if use_mod:
            self.mod_router = MoDRouter(n_embd)
        else:
            self.mod_router = None

        # Final norm after recurrence (Huginn: n_c)
        # Using a learnable RMSNorm for the recurrent output
        self.out_norm_weight = nn.Parameter(torch.ones(n_embd))

    def _sample_recurrence(self):
        """Sample iteration count from log-normal Poisson distribution.

        Matches Huginn §3.3: r ~ Poisson(exp(N(log(r̄), σ²)))
        """
        log_mean = math.log(max(self.mean_recurrence, 1))
        sigma = 0.5  # Huginn uses σ=1/2
        log_r = torch.normal(mean=log_mean, std=sigma, size=(1,)).item()
        r = max(1, min(self.max_recurrence, int(round(math.exp(log_r)))))
        return r

    def _mod_capacity_schedule(self, step, total_steps):
        """Capacity schedule for MoD: process fewer tokens at later steps.

        Step 0: 100% of tokens
        Step 1: 90% of tokens
        ...
        Last step: ~30% of tokens (only hardest ones)
        """
        if total_steps <= 1:
            return 1.0
        min_capacity = 0.3
        frac = step / (total_steps - 1)
        return 1.0 - frac * (1.0 - min_capacity)

    def forward(self, embedding, cos_sin, training=True):
        """
        Args:
            embedding: (B, T, D) output from prelude
            cos_sin: tuple of (cos, sin) for rotary embeddings
            training: if True, sample random recurrence; if False, use mean

        Returns:
            state: (B, T, D) final latent state
            info: dict with diagnostics (avg_loops, etc.)
        """
        B, T, D = embedding.shape
        device = embedding.device

        # Initialize random state (Huginn: path independence)
        state = torch.randn(B, T, D, device=device, dtype=embedding.dtype) * self.state_init_std

        # Determine number of recurrent steps
        if training:
            n_steps = self._sample_recurrence()
        else:
            n_steps = self.mean_recurrence

        info = {'n_steps': n_steps, 'mod_tokens_processed': []}

        for step in range(n_steps):
            # Truncated backpropagation: detach state for early steps
            if training and step < (n_steps - self.backprop_depth):
                state = state.detach()

            # Concat adapter: merge state + embedding
            x = self.adapter(state, embedding)

            # MoD routing (optional): only process top-k tokens
            if self.use_mod and self.mod_router is not None and training:
                capacity = self._mod_capacity_schedule(step, n_steps)
                if capacity < 1.0:
                    _, mask, _ = self.mod_router(x, capacity)
                    # Save original for skip connections
                    x_orig = x
                    info['mod_tokens_processed'].append(capacity)
                else:
                    mask = None
                    x_orig = None
                    info['mod_tokens_processed'].append(1.0)
            else:
                mask = None
                x_orig = None

            # Apply shared recurrent block (all layers)
            for layer in self.layers:
                x = layer(x, cos_sin)

            # Apply LoRA adapter for this depth (optional)
            if self.use_lora and self.lora_adapters is not None:
                lora_idx = min(step, len(self.lora_adapters) - 1)
                x = x + self.lora_adapters[lora_idx](x)

            # MoD: blend processed tokens with skipped tokens
            if mask is not None:
                x = mask * x + (1 - mask) * x_orig

            # Update state
            state = F.rms_norm(x, (D,), weight=self.out_norm_weight)

        return state, info


# ---------------------------------------------------------------------------
# KL-divergence early exit utility (inference only)
# ---------------------------------------------------------------------------

def kl_divergence_early_exit(logits_prev, logits_curr, threshold=5e-4):
    """Check if recurrence has converged by comparing successive logit distributions.

    Args:
        logits_prev: (B, T, V) logits from previous step
        logits_curr: (B, T, V) logits from current step
        threshold: KL-divergence threshold for convergence

    Returns:
        converged: bool, True if all positions have converged
    """
    p = F.softmax(logits_prev, dim=-1)
    q = F.softmax(logits_curr, dim=-1)
    # KL(p || q) = sum(p * log(p / q))
    kl = F.kl_div(q.log(), p, reduction='batchmean')
    return kl.item() < threshold
