"""
LoopedGPT: Depth-Recurrent Transformer for Nanochat

A Huginn-style (arXiv:2502.05171) looped transformer with optional
LoRA depth adapters and Mixture-of-Depths routing.

Architecture: Prelude → RecurrentCore → Coda → LM Head
- Same API as GPT (forward, generate, setup_optimizer)
- Separate class to keep standard GPT fully functional
"""

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.looped_transformer import (
    SandwichBlock,
    RecurrentCore,
    kl_divergence_early_exit,
    sandwich_norm,
)


@dataclass
class LoopedGPTConfig:
    """Configuration for the Looped Transformer.

    Designed for nano-scale experiments (~8M params) with
    Huginn-style P-R-C architecture.
    """
    # Standard dimensions
    vocab_size: int = 32768
    sequence_len: int = 2048
    n_embd: int = 768
    n_head: int = 6
    n_kv_head: int = 6

    # P-R-C layer counts
    n_prelude: int = 2          # layers in prelude (embed → latent)
    n_recurrent: int = 3        # layers in shared recurrent block
    n_coda: int = 1             # layers in coda (latent → logits)

    # Recurrence parameters
    mean_recurrence: int = 4    # r̄ during training (log-normal Poisson mean)
    max_recurrence: int = 16    # safety cap
    backprop_depth: int = 4     # truncated backprop through last k iterations
    state_init_std: float = 0.02  # σ for random state init

    # LoRA depth adapters (optional)
    use_lora: bool = True
    lora_rank: int = 32

    # Mixture-of-Depths routing (optional)
    use_mod: bool = True

    # Embedding scale (Huginn: γ)
    embedding_scale: float = 1.0


class LoopedGPT(nn.Module):
    """Depth-Recurrent Transformer with Huginn-style P-R-C architecture.

    Structure:
        tokens → Embedding × γ → Prelude layers → [RecurrentCore × N] → Coda layers → LM Head

    The RecurrentCore is the key innovation: it applies a shared block
    multiple times with input injection, optional LoRA adapters, and
    optional MoD routing.
    """

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # --- Embedding ---
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)

        # --- Prelude: specialized input layers ---
        self.prelude = nn.ModuleList([
            SandwichBlock(config.n_embd, config.n_head, config.n_kv_head)
            for _ in range(config.n_prelude)
        ])

        # --- Recurrent Core ---
        self.recurrent_core = RecurrentCore(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_recurrent_layers=config.n_recurrent,
            max_recurrence=config.max_recurrence,
            mean_recurrence=config.mean_recurrence,
            backprop_depth=config.backprop_depth,
            state_init_std=config.state_init_std,
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            use_mod=config.use_mod,
        )

        # --- Coda: specialized output layers ---
        self.coda = nn.ModuleList([
            SandwichBlock(config.n_embd, config.n_head, config.n_kv_head)
            for _ in range(config.n_coda)
        ])

        # --- LM Head (untied from embedding) ---
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # --- Rotary embeddings ---
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights explicitly."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # uniform bound for same std as normal

        # Embedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

        # LM Head
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Prelude layers
        for block in self.prelude:
            self._init_sandwich_block(block, s)

        # Coda layers
        for block in self.coda:
            self._init_sandwich_block(block, s)

        # Recurrent core layers
        for block in self.recurrent_core.layers:
            self._init_sandwich_block(block, s)

        # Concat adapter
        torch.nn.init.uniform_(self.recurrent_core.adapter.proj.weight, -s, s)

        # LoRA adapters (init down to small, up to zero for near-zero initial output)
        if self.recurrent_core.lora_adapters is not None:
            for lora in self.recurrent_core.lora_adapters:
                torch.nn.init.normal_(lora.down.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(lora.up.weight)

        # MoD router
        if self.recurrent_core.mod_router is not None:
            torch.nn.init.zeros_(self.recurrent_core.mod_router.scorer.weight)

        # Recurrent output norm
        self.recurrent_core.out_norm_weight.fill_(1.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embedding to bf16 if on CUDA
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)

    def _init_sandwich_block(self, block, s):
        """Initialize a SandwichBlock's weights."""
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Precompute rotary embeddings (same as GPT)."""
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.wte.weight.device

    def num_params(self):
        """Return detailed parameter counts."""
        embedding = self.wte.weight.numel()
        prelude = sum(p.numel() for p in self.prelude.parameters())
        recurrent_block = sum(p.numel() for layer in self.recurrent_core.layers for p in layer.parameters())
        adapter = sum(p.numel() for p in self.recurrent_core.adapter.parameters())
        lora = sum(p.numel() for p in self.recurrent_core.lora_adapters.parameters()) if self.recurrent_core.lora_adapters else 0
        mod = sum(p.numel() for p in self.recurrent_core.mod_router.parameters()) if self.recurrent_core.mod_router else 0
        coda = sum(p.numel() for p in self.coda.parameters())
        lm_head = self.lm_head.weight.numel()
        other = self.recurrent_core.out_norm_weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return {
            'embedding': embedding,
            'prelude': prelude,
            'recurrent_block': recurrent_block,
            'adapter': adapter,
            'lora': lora,
            'mod': mod,
            'coda': coda,
            'lm_head': lm_head,
            'other': other,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """Setup MuonAdamW optimizer (same pattern as GPT)."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect parameter groups
        # Matrix params: all attention + MLP weights in prelude, recurrent, coda
        matrix_params = []
        for block in self.prelude:
            matrix_params.extend(list(block.parameters()))
        for block in self.recurrent_core.layers:
            matrix_params.extend(list(block.parameters()))
        for block in self.coda:
            matrix_params.extend(list(block.parameters()))

        # AdamW groups: embedding, lm_head, adapter, LoRA, MoD, scalars
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        adapter_params = list(self.recurrent_core.adapter.parameters())
        scalar_params = [self.recurrent_core.out_norm_weight]

        lora_params = list(self.recurrent_core.lora_adapters.parameters()) if self.recurrent_core.lora_adapters else []
        mod_params = list(self.recurrent_core.mod_router.parameters()) if self.recurrent_core.mod_router else []

        # Verify we have all parameters
        all_param_lists = (matrix_params + embedding_params + lm_head_params +
                          adapter_params + scalar_params + lora_params + mod_params)
        assert len(list(self.parameters())) == len(all_param_lists), \
            f"Parameter count mismatch: {len(list(self.parameters()))} != {len(all_param_lists)}"

        dmodel_lr_scale = (model_dim / 768) ** -0.5

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=adapter_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=scalar_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        if lora_params:
            param_groups.append(dict(kind='adamw', params=lora_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        if mod_params:
            param_groups.append(dict(kind='adamw', params=mod_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        # Muon groups (matrix params, grouped by shape)
        for shape in sorted({p.shape for p in matrix_params}):
            group = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, loss_reduction='mean'):
        """
        Forward pass of the Looped Transformer.

        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (None for inference)
            loss_reduction: 'mean' or 'none'

        Returns:
            If targets: loss (scalar)
            If no targets: logits (B, T, vocab_size)
        """
        B, T = idx.size()
        training = targets is not None

        # Rotary embeddings
        assert T <= self.cos.size(1), f"Seq len {T} > rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Embed + scale
        x = self.wte(idx) * self.config.embedding_scale
        x = sandwich_norm(x)

        # Prelude: embed into latent space
        for block in self.prelude:
            x = block(x, cos_sin)

        embedding = x  # save for input injection in recurrent core

        # Recurrent Core: loop with input injection
        x, recurrence_info = self.recurrent_core(embedding, cos_sin, training=training)

        # Coda: decode from latent space
        for block in self.coda:
            x = block(x, cos_sin)

        x = sandwich_norm(x)

        # LM Head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42,
                 use_early_exit=False, early_exit_threshold=5e-4):
        """Autoregressive generation with optional KL-divergence early exit.

        Args:
            tokens: list of token ids (prompt)
            max_tokens: how many tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            seed: random seed
            use_early_exit: if True, use KL-div early exit in recurrent core
            early_exit_threshold: KL-div threshold for convergence
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]   # (B, vocab_size) — last token

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
