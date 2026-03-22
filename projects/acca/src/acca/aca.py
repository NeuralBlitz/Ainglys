"""
Attentive Causal Automata (ACA) Module

Implements the meta-architecture that learns both structure and function
via self-supervised causal discovery and gradient-based topology optimization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from .mmdg import MultiModalDynamicGraph


class CausalDiscovery(nn.Module):
    """
    Self-supervised causal discovery module.
    Learns causal relationships between variables without explicit supervision.
    """

    def __init__(
        self,
        num_variables: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_variables = num_variables

        self.encoder = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variables * num_variables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Infer causal structure from input data.

        Args:
            x: Input data [batch, num_variables, feature_dim]

        Returns:
            Adjacency matrix representing causal graph
        """
        batch_size = x.shape[0]

        encoded = self.encoder(x.mean(dim=-1))
        adj_matrix = encoded.view(batch_size, self.num_variables, self.num_variables)
        adj_matrix = torch.sigmoid(adj_matrix)

        return adj_matrix

    def causal_loss(
        self, adj_pred: torch.Tensor, x: torch.Tensor, lambda_sparse: float = 0.1
    ) -> torch.Tensor:
        """
        Compute causal discovery loss with sparsity regularization.

        Args:
            adj_pred: Predicted adjacency matrix
            x: Input data
            lambda_sparse: Sparsity coefficient

        Returns:
            Loss value
        """
        reconstruction = torch.matmul(
            torch.matmul(x, adj_pred), adj_pred.transpose(-2, -1)
        )

        reconstruction_loss = nn.functional.mse_loss(reconstruction, x)

        sparsity_loss = lambda_sparse * torch.sum(torch.abs(adj_pred))

        return reconstruction_loss + sparsity_loss


class TopologyOptimizer(nn.Module):
    """
    Gradient-based topology optimization for network architecture.
    """

    def __init__(
        self,
        num_nodes: int,
        max_edges: int = 100,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_edges = max_edges

        self.edge_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self) -> torch.Tensor:
        """
        Get optimized edge weights.

        Returns:
            Edge weight matrix
        """
        return torch.sigmoid(self.edge_logits)

    def get_sparse_topology(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Extract sparse topology by thresholding.

        Args:
            threshold: Edge weight threshold

        Returns:
            Binary adjacency matrix
        """
        edges = self.forward()
        return (edges > threshold).float()


class AttentiveCausalAutomata(nn.Module):
    """
    Attentive Causal Automata (ACA)

    A meta-architecture that learns both structure and function via
    self-supervised causal discovery and gradient-based topology optimization.
    """

    def __init__(
        self,
        num_nodes: int = 16,
        node_dim: int = 64,
        num_modalities: int = 1,
        num_classes: int = 10,
        attention_heads: int = 8,
    ):
        super().__init__()

        self.mmdg = MultiModalDynamicGraph(
            num_nodes=num_nodes,
            node_dim=node_dim,
            num_modalities=num_modalities,
            attention_heads=attention_heads,
        )

        self.causal_discovery = CausalDiscovery(
            num_variables=num_nodes,
            hidden_dim=node_dim,
        )

        self.topology_optimizer = TopologyOptimizer(
            num_nodes=num_nodes,
        )

        self.classifier = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(node_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, return_topology: bool = False) -> Dict[str, Any]:
        """
        Forward pass through ACA.

        Args:
            x: Input tensor [batch, num_nodes, node_dim]
            return_topology: Whether to return learned topology

        Returns:
            Dictionary with logits and optional topology
        """
        node_features, attention_weights = self.mmdg(x)

        causal_structure = self.causal_discovery(node_features)

        optimized_topology = self.topology_optimizer()

        pooled = node_features.mean(dim=1)
        logits = self.classifier(pooled)

        output = {"logits": logits}

        if return_topology:
            output.update(
                {
                    "attention_weights": attention_weights,
                    "causal_structure": causal_structure,
                    "optimized_topology": optimized_topology,
                }
            )

        return output

    def learn_topology(
        self, x: torch.Tensor, num_steps: int = 100, lr: float = 0.01
    ) -> torch.Tensor:
        """
        Learn optimal topology through gradient-based optimization.

        Args:
            x: Input data for topology learning
            num_steps: Number of optimization steps
            lr: Learning rate

        Returns:
            Learned topology adjacency matrix
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            node_features, _ = self.mmdg(x)
            causal_structure = self.causal_discovery(node_features)

            causal_loss = self.causal_discovery.causal_loss(
                causal_structure, node_features
            )

            topology_loss = -torch.sum(
                torch.log(causal_structure + 1e-8) * causal_structure
            )

            total_loss = causal_loss + 0.1 * topology_loss
            total_loss.backward()
            optimizer.step()

        return self.topology_optimizer()
