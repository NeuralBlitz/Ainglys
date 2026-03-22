#!/usr/bin/env python3
"""
Performance Benchmarks for AI Hub & Research Packages
======================================================

Run with: pytest tests/benchmarks/ -v --benchmark-only

Note: These tests are skipped by default. To run:
  pytest tests/benchmarks/ -v -m benchmark
"""

import pytest
import torch
import numpy as np
import time

# Try to import research packages, skip if not available
try:
    from acml import (
        FisherInformationEstimator,
        FunctorialLayer,
        EnrichedCategory,
        CategoricalMetaLearner,
        ACMLConfig,
    )

    ACML_AVAILABLE = True
except ImportError:
    ACML_AVAILABLE = False

try:
    from aether_calc import AetherOptimizer, AetherConfig

    AETHER_AVAILABLE = True
except ImportError:
    AETHER_AVAILABLE = False

try:
    from gaav import Granule, GranularOperators

    GAAV_AVAILABLE = True
except ImportError:
    GAAV_AVAILABLE = False

try:
    from gaft import ArithmonicGranule, GranuleType, GranuleState, GranularSimulation

    GAFT_AVAILABLE = True
except ImportError:
    GAFT_AVAILABLE = False


@pytest.mark.benchmark
@pytest.mark.skipif(not ACML_AVAILABLE, reason="ACML not installed")
class TestACMLBenchmarks:
    """Benchmarks for ACML meta-learning."""

    def test_fisher_information_estimation(self, benchmark):
        """Benchmark Fisher information matrix estimation."""
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10)
        )
        data = torch.randn(1000, 100)
        estimator = FisherInformationEstimator(num_samples=50)

        result = benchmark(estimator.estimate, model, data)
        assert result is not None

    def test_functorial_layer_forward(self, benchmark):
        """Benchmark functorial layer forward pass."""
        layer = FunctorialLayer(784, 256, EnrichedCategory("metric"))
        x = torch.randn(32, 784)

        result = benchmark(layer, x)
        assert result.shape == (32, 256)

    def test_meta_learning_step(self, benchmark):
        """Benchmark meta-learning training step."""
        config = ACMLConfig(
            encoder_dims=[50, 30, 20],
            decoder_dims=[20, 10],
            inner_steps=3,
            use_natural_gradient=False,
        )
        model = CategoricalMetaLearner(config)

        task_batch = [
            {
                "support_x": torch.randn(5, 50),
                "support_y": torch.randint(0, 10, (5,)),
                "query_x": torch.randn(15, 50),
                "query_y": torch.randint(0, 10, (15,)),
            }
            for _ in range(4)
        ]

        # Skip actual training in benchmark
        result = benchmark(lambda: sum(t["support_x"].sum() for t in task_batch))
        assert result is not None


@pytest.mark.benchmark
@pytest.mark.skipif(not AETHER_AVAILABLE, reason="Aether-Calc not installed")
class TestAetherCalcBenchmarks:
    """Benchmarks for Aether-Calculus optimizer."""

    def test_volatility_computation(self, benchmark):
        """Benchmark volatility surface computation."""
        param = torch.nn.Parameter(torch.randn(100))
        param.grad = torch.randn(100)

        optimizer = AetherOptimizer([param], AetherConfig())
        result = benchmark(optimizer.compute_volatility)
        assert isinstance(result, float)

    def test_aether_step(self, benchmark):
        """Benchmark Aether optimizer step."""
        params = [torch.nn.Parameter(torch.randn(1000)) for _ in range(10)]
        for p in params:
            p.grad = torch.randn(1000)

        optimizer = AetherOptimizer(params, AetherConfig())

        def step_all():
            optimizer.zero_grad()
            for p in params:
                p.grad = torch.randn(1000)
            optimizer.step()

        benchmark(step_all)


@pytest.mark.benchmark
@pytest.mark.skipif(not GAAV_AVAILABLE, reason="GAAV not installed")
class TestGAAVBenchmarks:
    """Benchmarks for Granular Arithmetic."""

    def test_granule_operations(self, benchmark):
        """Benchmark granular operations."""
        g1 = Granule(value=10.0, uncertainty={"mu": 0.0, "sigma": 1.0}, context="test")
        g2 = Granule(value=5.0, uncertainty={"mu": 0.0, "sigma": 0.5}, context="test")

        def chain_operations():
            result = GranularOperators.add(g1, g2)
            result = GranularOperators.multiply(result, g1)
            return result

        result = benchmark(chain_operations)
        assert result is not None

    def test_uncertainty_sampling(self, benchmark):
        """Benchmark uncertainty sampling."""
        g = Granule(value=0.0, uncertainty={"mu": 0.0, "sigma": 1.0}, context="test")

        result = benchmark(g.sample, n_samples=1000)
        assert len(result) == 1000


@pytest.mark.benchmark
@pytest.mark.skipif(not GAFT_AVAILABLE, reason="GAFT not installed")
class TestGAFTBenchmarks:
    """Benchmarks for Granular Field Theory."""

    def test_granule_interaction(self, benchmark):
        """Benchmark granule interaction computation."""
        g1 = ArithmonicGranule(
            value=1,
            granule_type=GranuleType.VALUE,
            state=GranuleState.ACTIVE,
            potential_charge=1.0,
            coherence=1.0,
        )
        g2 = ArithmonicGranule(
            value=1,
            granule_type=GranuleType.VALUE,
            state=GranuleState.ACTIVE,
            potential_charge=1.0,
            coherence=1.0,
        )

        result = benchmark(g1.compute_interaction_potential, g2)
        assert isinstance(result, float)

    def test_simulation_step(self, benchmark):
        """Benchmark simulation step."""
        sim = GranularSimulation()
        sim.granularize_expression("1 + 1")

        benchmark(sim.step, dt=0.1)


class TestMemoryBenchmarks:
    """Memory usage benchmarks."""

    def test_large_model_memory(self):
        """Test memory usage with large models."""
        import torch

        # Create increasingly large models
        sizes = [100, 500, 1000]
        memory_usage = []

        for size in sizes:
            model = torch.nn.Linear(size, size)
            # Get memory usage
            mem = sum(p.numel() * p.element_size() for p in model.parameters())
            memory_usage.append((size, mem / 1024 / 1024))  # MB

        print(f"\nMemory usage by model size:")
        for size, mem in memory_usage:
            print(f"  Size {size}x{size}: {mem:.2f} MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
