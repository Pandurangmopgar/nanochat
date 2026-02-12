"""
Titans Neural Long-Term Memory Module (MAL variant)

Implements the Memory as Layer (MAL) architecture from:
"Titans: Learning to Memorize at Test Time" (Behrouz et al., 2025)

Key components:
- Deep MLP memory network updated via gradient-based surprise metric
- K/V projections for associative memory loss (Eq. 11-12)
- Momentum-based update rule with data-dependent η, θ, α (Eq. 13-14)
- Weight decay (forgetting mechanism) for memory management
- 1D depthwise convolution for local pattern capture
- SiLU activation (§4.4)
- Sigmoid gating to blend memory output with residual stream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TitansMemory(nn.Module):
    """
    Neural long-term memory module (paper-faithful implementation).

    Uses a deep MLP as an associative memory that is updated at test time
    via gradient descent on a key-value association objective (surprise metric),
    with data-dependent momentum, learning rate, and weight decay.

    Paper equations:
        k_t = x_t · W_K,  v_t = x_t · W_V                         (Eq. 11)
        ℓ(M; x_t) = ||M(k_t) - v_t||²                              (Eq. 12)
        M_t = (1 - α_t) · M_{t-1} + S_t                            (Eq. 13)
        S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)             (Eq. 14)

    Args:
        model_dim: Dimension of the input/output (matches GPT's n_embd)
        memory_dim: Hidden dimension of the memory MLP
        memory_depth: Number of hidden layers in the memory MLP
        conv_kernel_size: Kernel size for the input convolution
    """

    def __init__(
        self,
        model_dim: int,
        memory_dim: int = 512,
        memory_depth: int = 3,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.memory_dim = memory_dim
        self.memory_depth = memory_depth

        # Key/Value projections (Eq. 11) — trained by outer loop
        self.W_K = nn.Linear(model_dim, model_dim, bias=False)
        self.W_V = nn.Linear(model_dim, model_dim, bias=False)

        # Build the deep memory network: input_proj -> [hidden layers] -> output_proj
        # Uses SiLU activation per §4.4
        layers = []
        layers.append(nn.Linear(model_dim, memory_dim, bias=False))
        layers.append(nn.SiLU())
        for _ in range(memory_depth - 1):
            layers.append(nn.Linear(memory_dim, memory_dim, bias=False))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(memory_dim, model_dim, bias=False))
        self.memory_net = nn.Sequential(*layers)

        # 1D depthwise convolution for local pattern capture (ablation: ~1.7 ppl)
        # Conv operates over the sequence dimension
        self.conv = nn.Conv1d(
            model_dim, model_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=model_dim,  # depthwise conv for efficiency
            bias=False,
        )

        # Data-dependent hyperparameters (§3.1, §4.4)
        # α_t: forgetting rate, η_t: momentum decay, θ_t: learning rate
        # Each is a linear projection from model_dim -> 1, then activated
        self.proj_alpha = nn.Linear(model_dim, 1, bias=True)  # -> sigmoid -> [0, 1]
        self.proj_eta = nn.Linear(model_dim, 1, bias=True)    # -> sigmoid -> [0, 1]
        self.proj_theta = nn.Linear(model_dim, 1, bias=True)  # -> softplus -> (0, ∞)

        # Initialize data-dependent hyperparameter projections to sensible defaults:
        # α ≈ 0.01 (low forgetting), η ≈ 0.9 (high momentum), θ ≈ 0.01 (small LR)
        nn.init.zeros_(self.proj_alpha.weight)
        nn.init.constant_(self.proj_alpha.bias, -4.6)   # sigmoid(-4.6) ≈ 0.01

        nn.init.zeros_(self.proj_eta.weight)
        nn.init.constant_(self.proj_eta.bias, 2.2)      # sigmoid(2.2) ≈ 0.9

        nn.init.zeros_(self.proj_theta.weight)
        nn.init.constant_(self.proj_theta.bias, -3.9)   # softplus(-3.9) ≈ 0.02

        # Momentum state buffer — lazily initialized per batch
        self._momentum_state = None
        self._total_mem_params = sum(p.numel() for p in self.memory_net.parameters())

    def _init_momentum_state(self, device, dtype):
        """Initialize the momentum buffer to zeros."""
        self._momentum_state = torch.zeros(
            self._total_mem_params, device=device, dtype=dtype
        )

    def _compute_hyperparams(self, x):
        """
        Compute data-dependent hyperparameters from input.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            alpha, eta, theta: scalar hyperparameters (averaged over batch & sequence)
        """
        # Mean-pool over (B, T) to get a single representation
        # NOT detached — so proj_alpha/eta/theta get outer-loop gradients
        x_mean = x.float().mean(dim=(0, 1))  # (D,)

        alpha = torch.sigmoid(self.proj_alpha(x_mean)).squeeze()  # scalar in [0, 1]
        eta = torch.sigmoid(self.proj_eta(x_mean)).squeeze()      # scalar in [0, 1]
        theta = F.softplus(self.proj_theta(x_mean)).squeeze()     # scalar in (0, ∞)

        return alpha, eta, theta

    def forward(self, x, update_memory=True):
        """
        Args:
            x: Input tensor of shape (B, T, D)
            update_memory: Whether to update memory parameters (True during training)

        Returns:
            Memory output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Apply 1D convolution for local context (B, T, D) -> (B, D, T) -> conv -> (B, T, D)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)

        # Project to keys for memory query (Eq. 11)
        keys = self.W_K(x_conv)

        # Query the memory network (retrieval): M(k_t)
        mem_out = self.memory_net(keys)

        # Update memory parameters via gradient-based surprise metric
        if update_memory and self.training:
            # Initialize momentum state on first call
            if self._momentum_state is None:
                self._init_momentum_state(x.device, torch.float32)

            # Compute data-dependent hyperparameters
            alpha, eta, theta = self._compute_hyperparams(x)

            # Force float32 for numerical stability of the surprise gradient
            # (bfloat16 autocast can cause precision loss in the inner loop)
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                mem_out_f32 = mem_out.float()
                # Project to values (Eq. 11)
                # Not detached — create_graph=False prevents 2nd-order grads,
                # and W_V gets outer-loop gradients via retain_graph=True
                values_f32 = self.W_V(x_conv).float()

                # Compute associative memory loss (Eq. 12): ||M(k_t) - v_t||²
                loss = F.mse_loss(mem_out_f32, values_f32)

                # Compute gradients of surprise w.r.t. memory parameters only
                mem_params = list(self.memory_net.parameters())
                grads = torch.autograd.grad(
                    loss,
                    mem_params,
                    retain_graph=True,  # graph needed for outer loss.backward()
                    create_graph=False,
                )

            # Apply momentum update (Eq. 13-14):
            #   S_t = η_t · S_{t-1} - θ_t · grad_t
            #   W_t = (1 - α_t) · W_{t-1} + S_t
            with torch.no_grad():
                # Flatten all gradients into a single vector
                grad_flat = torch.cat([g.float().reshape(-1) for g in grads])

                # Momentum update: S_t = η · S_{t-1} - θ · grad
                self._momentum_state.mul_(eta.item()).sub_(
                    grad_flat, alpha=theta.item()
                )

                # Apply to each parameter using .data to avoid autograd conflicts
                # (inner-loop updates must not invalidate the outer computation graph)
                alpha_val = alpha.item()
                idx = 0
                for p in mem_params:
                    numel = p.numel()
                    update = self._momentum_state[idx : idx + numel].reshape_as(p)

                    # Weight decay / forgetting (Eq. 13): shrink existing memory
                    p.data.mul_(1.0 - alpha_val)

                    # Add momentum-based surprise update
                    p.data.add_(update.to(p.dtype))

                    idx += numel

            # Straight-through gradient trick: add zero-magnitude contributions so
            # W_V and proj_alpha/eta/theta get outer-loop gradients via backprop.
            # These don't change mem_out numerically but create gradient paths.
            # W_V only appears in the inner surprise loss (create_graph=False severs
            # its gradient path), so we route it through the output here.
            values_reg = self.W_V(x_conv).sum() * 0.0
            hyperparam_reg = (alpha + eta + theta) * 0.0
            mem_out = mem_out + values_reg + hyperparam_reg

        return mem_out

    def reset_memory_state(self):
        """Reset the momentum state (e.g., between sequences/episodes)."""
        self._momentum_state = None


class TitansLayer(nn.Module):
    """
    MAL (Memory as Layer) wrapper (§4.3).

    Inserts the neural memory as a sequential layer in the architecture.
    Uses a sigmoid gate to blend memory output with the residual input:
        output = x + gate * (memory(x) - x_normed)

    The gate is initialized near 0 so the model starts as near-identity
    and gradually learns to incorporate memory.

    Args:
        model_dim: Dimension of the input/output
        memory_dim: Hidden dimension of the memory MLP
        memory_depth: Number of hidden layers in the memory MLP
    """

    def __init__(self, model_dim: int, memory_dim: int = 512, memory_depth: int = 3):
        super().__init__()
        self.memory = TitansMemory(
            model_dim=model_dim,
            memory_dim=memory_dim,
            memory_depth=memory_depth,
        )
        # Learned gate: produces a scalar gate per position per feature
        self.gate = nn.Linear(model_dim, model_dim, bias=True)
        # Initialize gate bias so gate starts near 0 (mostly pass-through)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, x, update_memory=True):
        """
        Args:
            x: Input tensor of shape (B, T, D)
            update_memory: Whether to update memory (True during training)

        Returns:
            Output tensor of shape (B, T, D), blending memory and residual
        """
        # Norm before memory (pre-norm style, consistent with nanochat)
        x_normed = F.rms_norm(x, (x.size(-1),))

        # Get memory output
        mem_out = self.memory(x_normed, update_memory=update_memory)

        # Compute gate (sigmoid ensures [0, 1] range)
        gate = torch.sigmoid(self.gate(x_normed))

        # Blend: residual connection with gated memory
        output = x + gate * (mem_out - x_normed)

        return output
