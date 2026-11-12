"""
Multi-Modal Dynamic Graph (MMDG) Module

Implements the time-dependent directed graph structure where:
- Nodes represent computational units (attention heads, layers, data sources)
- Edges encode information flow and dependency
- Dynamic weights are governed by attention-driven differential equations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import networkx as nx


class MultiModalDynamicGraph(nn.Module):
    """
    Multi-Modal Dynamic Graph (MMDG) for ML/AI architecture representation.

    G(t) = (V(t), E(t), A(t)) where:
    - V(t): set of nodes (computational units)
    - E(t): set of edges (information flow)
    - A(t): attention matrix governing edge weights
    """

    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        num_modalities: int = 1,
        attention_heads: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_modalities = num_modalities
        self.attention_heads = attention_heads

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, node_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=attention_heads, batch_first=True
        )

        self.edge_weights = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(
        self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dynamic graph.

        Args:
            x: Input tensor [batch, num_nodes, node_dim]
            edge_index: Optional edge connectivity [2, num_edges]

        Returns:
            output: Transformed node features
            attention_weights: Attention matrix
        """
        if x.ndim != 3 or x.shape[-1] != self.node_dim:
            raise ValueError(
                f"Expected [batch, nodes, {self.node_dim}] input, got {tuple(x.shape)}"
            )
        batch_size, supplied_nodes, _ = x.shape
        if supplied_nodes < self.num_nodes:
            padding = x.new_zeros(batch_size, self.num_nodes - supplied_nodes, self.node_dim)
            x = torch.cat([x, padding], dim=1)
        elif supplied_nodes > self.num_nodes:
            x = x[:, : self.num_nodes, :]

        nodes = x + self.node_embeddings.unsqueeze(0)
        adjacency = torch.softmax(self.edge_weights, dim=-1)
        graph_messages = torch.matmul(adjacency.unsqueeze(0), nodes)
        attended_nodes = nodes + graph_messages

        attn_output, attn_weights = self.attention(
            attended_nodes, attended_nodes, attended_nodes
        )
        return attn_output, attn_weights

    def update_dynamics(self, t: float, node_states: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic update using attention-driven differential equations.

        dV/dt = f(V, A(t))

        Args:
            t: Current time step
            node_states: Current node states

        Returns:
            Updated node states
        """
        _, attn_weights = self.forward(node_states.unsqueeze(0))
        dynamics = torch.matmul(attn_weights.squeeze(0), node_states)

        return dynamics

    def get_topology(self) -> nx.DiGraph:
        """
        Extract current graph topology as NetworkX graph.
        """
        G = nx.DiGraph()

        for i in range(self.num_nodes):
            G.add_node(i, embedding=self.node_embeddings[i].detach().cpu().numpy())

        edge_weights = torch.sigmoid(self.edge_weights).detach().cpu().numpy()
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if edge_weights[i, j] > 0.5:
                    G.add_edge(i, j, weight=edge_weights[i, j])

        return G
