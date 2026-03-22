"""
FASCSS - Framework for Adaptive Self-Configuring Systems
"""

import torch
import torch.nn as nn
import random


class SelfConfiguringModule(nn.Module):
    """
    A module that can dynamically configure its own architecture.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_branches: int = 4
    ):
        super().__init__()

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_branches)
            ]
        )

        self.gate = nn.Linear(input_dim, num_branches)

    def forward(self, x: torch.Tensor):
        gate_weights = torch.softmax(self.gate(x), dim=-1)

        outputs = torch.stack([branch(x) for branch in self.branches], dim=1)

        weighted = gate_weights.unsqueeze(-1) * outputs
        return weighted.sum(dim=1)


class FASCSS(nn.Module):
    """
    Framework for Adaptive Self-Configuring Systems.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_modules: int = 3
    ):
        super().__init__()

        self.modules = nn.ModuleList(
            [
                SelfConfiguringModule(input_dim, hidden_dim, hidden_dim)
                for _ in range(num_modules)
            ]
        )

        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        for module in self.modules:
            x = module(x)
        return self.output(x)

    def adapt(self, x: torch.Tensor, y: torch.Tensor, steps: int = 10):
        """
        Adapt the architecture based on data.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.forward(x), y)
            loss.backward()
            optimizer.step()
