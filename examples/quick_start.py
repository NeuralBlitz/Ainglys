#!/usr/bin/env python3
"""
AI Hub - Quick Start Examples
==============================

Run any of these examples to see the capabilities in action.

Usage:
    python3 examples/quick_start.py --example meta_learning
    python3 examples/quick_start.py --example optimization
    python3 examples/quick_start.py --example granular
    python3 examples/quick_start.py --example field
    python3 examples/quick_start.py --example api
    python3 examples/quick_start.py --example cli
    python3 examples/quick_start.py --example all
"""

import argparse
import torch
import numpy as np


def example_meta_learning():
    """Example: ACML Meta-Learning"""
    print("\n" + "=" * 60)
    print("Example 1: Category-Theoretic Meta-Learning (ACML)")
    print("=" * 60)

    from acml import (
        EnrichedCategory,
        FunctorialLayer,
        CategoricalMetaLearner,
        ACMLConfig,
    )

    # Create category
    cat = EnrichedCategory("metric")
    print("✓ Created metric-enriched category")

    # Build functorial network
    layers = [
        FunctorialLayer(784, 256, cat, activation=torch.nn.ReLU()),
        FunctorialLayer(256, 128, cat, activation=torch.nn.ReLU()),
    ]
    print("✓ Built 2-layer functorial network")

    # Create meta-learner
    config = ACMLConfig(
        encoder_dims=[784, 256, 128], decoder_dims=[128, 10], inner_steps=3
    )
    model = CategoricalMetaLearner(config)
    print("✓ Created meta-learner")

    # Test forward pass
    x = torch.randn(5, 784)
    output = model(x)
    print(f"✓ Forward pass: {x.shape} → {output.shape}")

    print("\n📊 Result: Meta-learning model ready for few-shot tasks!")
    return model


def example_optimization():
    """Example: Aether-Calc Optimization"""
    print("\n" + "=" * 60)
    print("Example 2: Aether-Calculus Optimizer")
    print("=" * 60)

    from aether_calc import AetherOptimizer, AetherConfig

    # Create simple quadratic problem
    x = torch.nn.Parameter(torch.tensor([5.0, 5.0]))

    # Configure Aether optimizer
    config = AetherConfig(
        learning_rate=0.1, volatility_threshold=2.0, thermal_noise=0.01
    )
    optimizer = AetherOptimizer([x], config)
    print("✓ Created Aether optimizer with entropy feedback")

    # Optimize: minimize (x-1)^2 + (y-2)^2
    trajectory = []
    for step in range(50):
        optimizer.zero_grad()
        loss = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
        loss.backward()

        # Track trajectory
        trajectory.append(x.detach().clone().numpy())

        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step}: x={x[0]:.4f}, y={x[1]:.4f}, loss={loss.item():.4f}")

    print(f"\n📊 Result: Converged to ({x[0]:.4f}, {x[1]:.4f})")
    print(f"   Target: (1.0, 2.0)")
    print(f"   Error: {torch.norm(x - torch.tensor([1.0, 2.0])):.4f}")

    return trajectory


def example_granular():
    """Example: GAAV Granular Arithmetic"""
    print("\n" + "=" * 60)
    print("Example 3: Granular Arithmetic with Uncertainty")
    print("=" * 60)

    from gaav import Granule, GranularOperators

    # Create measurements with uncertainty
    measurement_a = Granule(
        value=10.5,
        uncertainty={"mu": 0.0, "sigma": 0.5},
        context="temperature_sensor_A",
    )

    measurement_b = Granule(
        value=5.2,
        uncertainty={"mu": 0.0, "sigma": 0.3},
        context="temperature_sensor_B",
    )

    print(
        f"✓ Measurement A: {measurement_a.value} ± {measurement_a.uncertainty['sigma']}"
    )
    print(
        f"✓ Measurement B: {measurement_b.value} ± {measurement_b.uncertainty['sigma']}"
    )

    # Perform arithmetic with uncertainty propagation
    sum_result = GranularOperators.add(measurement_a, measurement_b)
    product_result = GranularOperators.multiply(measurement_a, measurement_b)

    print(f"\n📊 Results:")
    print(f"   A + B = {sum_result.value:.2f} ± {sum_result.uncertainty['sigma']:.2f}")
    print(
        f"   A × B = {product_result.value:.2f} ± {product_result.uncertainty['sigma']:.2f}"
    )
    print(f"   Combined uncertainty tracked automatically!")

    return sum_result, product_result


def example_field():
    """Example: GAFT Field Computation"""
    print("\n" + "=" * 60)
    print("Example 4: Granular Field Theory (GAFT)")
    print("=" * 60)

    from gaft import GranularSimulation

    # Create simulation for simple arithmetic: 5 + 3
    sim = GranularSimulation(interaction_radius=2.0, binding_threshold=0.6)

    print("✓ Created granular simulation")

    # Granularize expression
    granules = sim.granularize_expression("5 + 3")
    print(f"✓ Granularized expression into {len(granules)} particles")

    # Run simulation
    print("✓ Running simulation...")
    history = sim.run(num_steps=10, dt=0.1)

    print(f"✓ Simulation complete ({len(history)} steps)")
    print(f"✓ Final granules: {len(sim.field.granules)}")

    # Check states
    states = [g.state.value for g in sim.field.granules]
    print(f"\n📊 Results:")
    print(f"   Initial expression: 5 + 3")
    print(f"   Granule states: {set(states)}")
    print(f"   Coherence: {sim.field.compute_total_coherence():.2f}")

    return sim


def example_api():
    """Example: AI Hub REST API"""
    print("\n" + "=" * 60)
    print("Example 5: AI Hub REST API")
    print("=" * 60)

    print("✓ API server can be started with:")
    print("   ai-hub api start --port 8000")
    print("\n📊 Available endpoints:")
    print("   GET  /          - API info")
    print("   GET  /status    - Check model status")
    print("   POST /train     - Train new model")
    print("   POST /predict   - Make predictions")
    print("   POST /save      - Save model")
    print("   GET  /models    - List models")
    print("\n📊 Example usage:")
    print("   curl http://localhost:8000/status")
    print("   curl -X POST http://localhost:8000/train -d '{\"epochs\":10}'")


def example_cli():
    """Example: AI Hub CLI"""
    print("\n" + "=" * 60)
    print("Example 6: AI Hub Rich CLI")
    print("=" * 60)

    print("✓ CLI commands available:")
    print("\n📊 System commands:")
    print("   ai-hub status              - Check system status")
    print("   ai-hub config show         - Display configuration")
    print("   ai-hub config get <key>    - Get config value")
    print("   ai-hub config set <k> <v>  - Set config value")
    print("\n📊 Local operations:")
    print("   ai-hub local train         - Train model locally")
    print("   ai-hub local predict       - Make predictions")
    print("   ai-hub local evaluate      - Evaluate model")
    print("\n📊 API operations:")
    print("   ai-hub api start           - Start API server")
    print("   ai-hub api stop            - Stop API server")
    print("   ai-hub api status          - Check API status")
    print("\n📊 Model management:")
    print("   ai-hub model train         - Train via API")
    print("   ai-hub model save <name>   - Save model")
    print("   ai-hub model load <name>   - Load model")
    print("   ai-hub model list          - List saved models")


def example_integration():
    """Example: Full Integration"""
    print("\n" + "=" * 60)
    print("Example 7: Full Integration Pipeline")
    print("=" * 60)

    print("✓ Combining multiple frameworks:")

    # 1. Create meta-learner
    from acml import CategoricalMetaLearner, ACMLConfig

    config = ACMLConfig(encoder_dims=[10, 20, 10], decoder_dims=[10, 5])
    meta_learner = CategoricalMetaLearner(config)
    print("  1. ✓ ACML meta-learner")

    # 2. Optimize with Aether
    from aether_calc import AetherOptimizer, AetherConfig

    optimizer = AetherOptimizer(
        meta_learner.parameters(), AetherConfig(learning_rate=0.01)
    )
    print("  2. ✓ Aether optimizer")

    # 3. Track uncertainties
    from gaav import Granule

    loss_history = Granule(
        value=1.0, uncertainty={"sigma": 0.1}, context="training_loss"
    )
    print("  3. ✓ GAAV uncertainty tracking")

    # 4. Deploy via API
    print("  4. ✓ Deploy to AI Hub API")

    print("\n📊 Result: Production-ready ML pipeline!")
    print("   - Meta-learning for rapid adaptation")
    print("   - Robust optimization with entropy feedback")
    print("   - Uncertainty quantification")
    print("   - REST API deployment")


def main():
    parser = argparse.ArgumentParser(description="AI Hub Quick Start Examples")
    parser.add_argument(
        "--example",
        choices=[
            "meta_learning",
            "optimization",
            "granular",
            "field",
            "api",
            "cli",
            "integration",
            "all",
        ],
        default="all",
        help="Which example to run",
    )

    args = parser.parse_args()

    if args.example == "all":
        print("\n" + "🚀" * 30)
        print("AI Hub - Complete Quick Start Guide")
        print("🚀" * 30)

        example_meta_learning()
        example_optimization()
        example_granular()
        example_field()
        example_api()
        example_cli()
        example_integration()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Explore notebooks/ directory")
        print("  2. Run tests: pytest tests/ -v")
        print("  3. Start API: ai-hub api start")
        print("  4. Read USE_CASES.md for more examples")
        print("=" * 60)

    else:
        examples = {
            "meta_learning": example_meta_learning,
            "optimization": example_optimization,
            "granular": example_granular,
            "field": example_field,
            "api": example_api,
            "cli": example_cli,
            "integration": example_integration,
        }

        examples[args.example]()


if __name__ == "__main__":
    main()
