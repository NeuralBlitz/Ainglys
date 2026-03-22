"""
Core module for ADAG framework.
"""

import torch
import torch.nn as nn


class CoreModule(nn.Module):
    """Core module for ADAG framework."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc(x)))
