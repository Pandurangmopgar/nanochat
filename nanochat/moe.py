"""
Mixture-of-Experts (MoE) MLP Module for nanochat GPT.

Replaces the standard dense MLP with a sparse mixture of expert MLPs.
Each token is routed to top-k experts (default: top-2 out of 4),
giving the model more capacity (total params) while keeping active
FLOPs per token roughly the same as a dense model.

Key components:
- ExpertMLP: Individual expert (same architecture as gpt.MLP but half width)
- MoEMLP: Router + expert dispatcher with load balancing auxiliary loss
- load_balance_loss: Prevents expert collapse (Switch Transformer, Eq. 4)

References:
- Switch Transformer (Fedus et al., 2022)
- Mixtral of Experts (Jiang et al., 2024)
- DeepSeek-V2 MoE (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """
    A single expert MLP. Same architecture as gpt.MLP but with configurable
    hidden dimension (default: 2 * n_embd, half of the dense 4 * n_embd).

    This keeps active FLOPs roughly constant: with top-2 routing and half-width
    experts, each token sees 2 * (2 * n_embd) = 4 * n_embd effective width.
    """

    def __init__(self, n_embd, expert_dim=None):
        super().__init__()
        if expert_dim is None:
            expert_dim = 2 * n_embd  # half of standard 4 * n_embd
        self.c_fc = nn.Linear(n_embd, expert_dim, bias=False)
        self.c_proj = nn.Linear(expert_dim, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² activation (same as gpt.MLP)
        x = self.c_proj(x)
        return x


class MoEMLP(nn.Module):
    """
    Mixture-of-Experts MLP layer.

    Routes each token to top-k experts and combines their outputs with
    learned routing weights. Includes auxiliary load-balancing loss to
    prevent expert collapse.

    Args:
        n_embd: Model embedding dimension
        num_experts: Total number of expert MLPs (default: 4)
        top_k: Number of experts active per token (default: 2)
        expert_dim: Hidden dimension per expert (default: 2 * n_embd)
        aux_loss_weight: Weight for the load balancing loss (default: 0.01)
    """

    def __init__(self, n_embd, num_experts=4, top_k=2, expert_dim=None, aux_loss_weight=0.01):
        super().__init__()
        assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # Router: linear projection from hidden dim to num_experts scores
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

        # Expert MLPs
        self.experts = nn.ModuleList([
            ExpertMLP(n_embd, expert_dim) for _ in range(num_experts)
        ])

        # Store auxiliary loss for the training loop to read
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            output: Tensor of shape (B, T, D) — weighted combination of expert outputs
        """
        B, T, D = x.shape

        # 1. Compute routing scores
        router_logits = self.gate(x)  # (B, T, num_experts)

        # 2. Select top-k experts per token
        routing_weights = F.softmax(router_logits, dim=-1)  # (B, T, num_experts)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (B, T, top_k)

        # Re-normalize weights so they sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Compute auxiliary load balancing loss (Switch Transformer Eq. 4)
        #    Encourages uniform expert utilization
        if self.training:
            self.aux_loss = self._load_balance_loss(router_logits, routing_weights)

        # 4. Dispatch tokens to experts and combine outputs
        # Use the simple loop approach (works well for small num_experts)
        output = torch.zeros_like(x)

        for k_idx in range(self.top_k):
            expert_indices = top_indices[:, :, k_idx]  # (B, T) — which expert for this k
            expert_weights = top_weights[:, :, k_idx]   # (B, T) — weight for this k

            for expert_idx in range(self.num_experts):
                # Create mask for tokens routed to this expert at this k position
                mask = (expert_indices == expert_idx)  # (B, T)

                if not mask.any():
                    continue

                # Gather tokens for this expert
                # Flatten B,T for efficient gathering
                flat_mask = mask.reshape(-1)  # (B*T,)
                flat_x = x.reshape(-1, D)     # (B*T, D)

                expert_input = flat_x[flat_mask]  # (num_selected, D)

                if expert_input.shape[0] == 0:
                    continue

                # Run through expert
                expert_output = self.experts[expert_idx](expert_input)  # (num_selected, D)

                # Scatter back with weights
                flat_weights = expert_weights.reshape(-1)[flat_mask].unsqueeze(-1)  # (num_selected, 1)
                weighted_output = expert_output * flat_weights

                # Add to output
                flat_output = output.reshape(-1, D)
                flat_output[flat_mask] += weighted_output
                output = flat_output.reshape(B, T, D)

        return output

    def _load_balance_loss(self, router_logits, routing_weights):
        """
        Compute the auxiliary load balancing loss (Switch Transformer, Eq. 4-5).

        Goal: Each expert should receive ~1/num_experts fraction of tokens.

        L_aux = num_experts * Σ_i (f_i · P_i)
        where:
            f_i = fraction of tokens routed to expert i
            P_i = mean routing probability for expert i

        This loss is minimized when f_i = P_i = 1/num_experts (uniform).
        """
        # f_i: fraction of tokens dispatched to each expert (from hard top-k assignments)
        _, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (B, T, top_k)
        # One-hot encode and sum to get per-expert counts
        one_hot = F.one_hot(top_indices, self.num_experts).float()  # (B, T, top_k, num_experts)
        one_hot = one_hot.sum(dim=2)  # (B, T, num_experts) — each token may select same expert twice
        f = one_hot.mean(dim=(0, 1))  # (num_experts,) — fraction of tokens per expert

        # P_i: mean routing probability for each expert
        P = routing_weights.mean(dim=(0, 1))  # (num_experts,)

        # Auxiliary loss
        aux_loss = self.num_experts * (f * P).sum()
        return self.aux_loss_weight * aux_loss


def init_moe_weights(moe_layer, n_embd):
    """Initialize MoE weights following the same scheme as gpt.py."""
    # Router: small init for balanced initial routing
    nn.init.normal_(moe_layer.gate.weight, mean=0.0, std=0.01)

    # Expert weights: same as standard MLP init
    s = 3**0.5 * n_embd**-0.5
    for expert in moe_layer.experts:
        nn.init.uniform_(expert.c_fc.weight, -s, s)
        nn.init.zeros_(expert.c_proj.weight)
