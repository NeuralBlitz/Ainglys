"""Focused invariants for ACCA graph attention and topology optimization."""
import torch

from acca import AttentiveCausalAutomata, CausalDiscovery, MultiModalDynamicGraph


def test_graph_output_responds_to_data():
    torch.manual_seed(11)
    graph = MultiModalDynamicGraph(num_nodes=4, node_dim=16, attention_heads=4)
    graph.eval()
    zeros, _ = graph(torch.zeros(2, 4, 16))
    ones, _ = graph(torch.ones(2, 4, 16))
    assert zeros.shape == (2, 4, 16)
    assert ones.shape == (2, 4, 16)
    assert not torch.allclose(zeros, ones)


def test_causal_discovery_shapes_and_gradients():
    torch.manual_seed(12)
    model = CausalDiscovery(num_variables=4, hidden_dim=16)
    x = torch.randn(2, 4, 16)
    adjacency = model(x)
    assert adjacency.shape == (2, 4, 4)
    loss = model.causal_loss(adjacency, x)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_full_automata_forward_backward():
    torch.manual_seed(13)
    model = AttentiveCausalAutomata(
        num_nodes=4, node_dim=16, attention_heads=4, num_classes=3
    )
    x = torch.randn(2, 4, 16)
    result = model(x, return_topology=True)
    assert result["logits"].shape == (2, 3)
    assert result["causal_structure"].shape == (2, 4, 4)
    assert result["optimized_topology"].shape == (4, 4)
    loss = result["logits"].square().mean()
    loss.backward()
    assert model.mmdg.node_embeddings.grad is not None


def test_topology_logits_are_optimized():
    torch.manual_seed(14)
    model = AttentiveCausalAutomata(
        num_nodes=4, node_dim=16, attention_heads=4, num_classes=3
    )
    x = torch.randn(2, 4, 16)
    before = model.topology_optimizer.edge_logits.detach().clone()
    after = model.learn_topology(x, num_steps=3, lr=0.01)
    assert after.shape == (4, 4)
    assert not torch.allclose(before, model.topology_optimizer.edge_logits.detach())
