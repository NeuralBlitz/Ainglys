"""
Adaptive Learning Framework (ALF) Implementation
"""

import torch
import torch.nn as nn


class AdaptiveLearningFramework(nn.Module):
    """
    ALF - A meta-learning framework for adaptive model optimization.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_tasks: int = 10
    ):
        super().__init__()

        self.meta_encoder = nn.Linear(input_dim, hidden_dim)

        self.task_specific = nn.ModuleDict(
            {f"task_{i}": nn.Linear(hidden_dim, hidden_dim) for i in range(num_tasks)}
        )

        self.adaptor = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, task_id: int = 0):
        x = torch.relu(self.meta_encoder(x))

        if f"task_{task_id}" in self.task_specific:
            x = self.task_specific[f"task_{task_id}"](x)

        x = self.adaptor(x)
        return self.output(x)
