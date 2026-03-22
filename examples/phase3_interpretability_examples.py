"""
Phase 3 Interpretability Examples

Demonstrates comprehensive model interpretability features:
- Saliency maps and gradient-based attribution
- SmoothGrad and Integrated Gradients
- Feature importance analysis
- Interactive interpretability dashboard
"""

import torch
import torch.nn as nn
import numpy as np
from src.ai_hub.interpretability import (
    InterpretabilityDashboard, SaliencyAnalyzer, FeatureAttributor
)


def example_1_saliency_maps():
    """Example 1: Compute saliency maps for image classification."""
    print("=" * 70)
    print("Example 1: Saliency Maps")
    print("=" * 70)
    
    # Create simple CNN
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create dummy image input (1 channel, 28x28)
    input_image = torch.randn(1, 1, 28, 28)
    
    # Initialize analyzer
    analyzer = SaliencyAnalyzer(model)
    
    # Compute different attribution methods
    print("Computing saliency map...")
    saliency = analyzer.compute_saliency(input_image)
    print(f"  Saliency shape: {saliency.shape}")
    print(f"  Saliency range: [{saliency.min():.4f}, {saliency.max():.4f}]")
    
    print("Computing SmoothGrad...")
    smoothgrad = analyzer.compute_smoothgrad(input_image, n_samples=20)
    print(f"  SmoothGrad shape: {smoothgrad.shape}")
    print(f"  SmoothGrad range: [{smoothgrad.min():.4f}, {smoothgrad.max():.4f}]")
    
    print("Computing Integrated Gradients...")
    integrated_grads = analyzer.compute_integrated_gradients(input_image, n_steps=30)
    print(f"  IG shape: {integrated_grads.shape}")
    print(f"  IG range: [{integrated_grads.min():.4f}, {integrated_grads.max():.4f}]")
    
    print("✓ Saliency analysis complete!\n")


def example_2_feature_importance():
    """Example 2: Feature importance for tabular data."""
    print("=" * 70)
    print("Example 2: Feature Importance")
    print("=" * 70)
    
    # Create simple MLP for tabular data
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Define feature names
    feature_names = [f"feature_{i}" for i in range(20)]
    
    # Create dummy data
    X_train = np.random.randn(200, 20)
    y_train = np.random.randint(0, 2, 200)
    
    # Initialize attributor
    attributor = FeatureAttributor(model, feature_names)
    
    # Compute permutation importance
    print("Computing permutation importance...")
    importance = attributor.permutation_importance(X_train, y_train)
    
    # Show top 5 most important features
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 most important features:")
    for feature, score in sorted_importance[:5]:
        print(f"  {feature}: {score:.4f}")
    
    print("✓ Feature importance analysis complete!\n")


def example_3_full_dashboard():
    """Example 3: Create comprehensive interpretability dashboard."""
    print("=" * 70)
    print("Example 3: Full Interpretability Dashboard")
    print("=" * 70)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Initialize dashboard
    feature_names = [f"pixel_{i}" for i in range(784)]
    dashboard = InterpretabilityDashboard(model, feature_names)
    
    # Create sample data
    input_sample = torch.randn(1, 784)
    X_train = np.random.randn(500, 784)
    y_train = np.random.randint(0, 10, 500)
    
    # Generate multiple explanations
    print("Generating saliency explanation...")
    saliency_exp = dashboard.explain_prediction(input_sample, method='saliency')
    print(f"  Attribution shape: {saliency_exp['attribution'].shape}")
    
    print("Generating SmoothGrad explanation...")
    smoothgrad_exp = dashboard.explain_prediction(
        input_sample, 
        method='smoothgrad',
        n_samples=30,
        noise_level=0.15
    )
    
    print("Generating Integrated Gradients explanation...")
    intgrad_exp = dashboard.explain_prediction(
        input_sample,
        method='integrated_gradients',
        n_steps=40
    )
    
    # Compute feature importance
    print("Computing feature importance...")
    importance = dashboard.compute_feature_importance(X_train, y_train)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Feature importance plot
    fig = dashboard.plot_feature_importance(importance, top_n=15)
    fig.write_html("interpretability_feature_importance.html")
    print("  ✓ Saved feature_importance.html")
    
    # Full dashboard
    dashboard_path = dashboard.create_interactive_dashboard(
        input_sample,
        X_train,
        y_train,
        save_path="interpretability_dashboard.html"
    )
    print(f"  ✓ Saved dashboard to: {dashboard_path}")
    
    # Export explanation history
    dashboard.export_explanations("explanations_history.json")
    print("  ✓ Exported explanations to explanations_history.json")
    
    print(f"\n✓ Dashboard creation complete!")
    print(f"  - {len(dashboard.explanation_history)} explanations generated")
    print(f"  - Dashboard available at: {dashboard_path}\n")


def example_4_layer_wise_analysis():
    """Example 4: Layer-wise analysis for deep models."""
    print("=" * 70)
    print("Example 4: Layer-wise Analysis")
    print("=" * 70)
    
    # Create multi-layer model
    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(100, 128)
            self.layer2 = nn.Linear(128, 64)
            self.layer3 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 5)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.relu(self.layer3(x))
            return self.output(x)
    
    model = DeepModel()
    dashboard = InterpretabilityDashboard(model)
    
    # Analyze different layers
    input_data = torch.randn(1, 100)
    
    print("Analyzing model layers...")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Model architecture:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"    - {name}: {module.in_features} -> {module.out_features}")
    
    # Generate explanation
    explanation = dashboard.explain_prediction(input_data, method='saliency')
    print(f"\n  Attribution computed successfully!")
    print(f"  Attribution shape: {explanation['attribution'].shape}")
    
    print("✓ Layer-wise analysis complete!\n")


def main():
    """Run all interpretability examples."""
    print("\n" + "=" * 70)
    print("AI HUB - Phase 3: Interpretability Examples")
    print("=" * 70 + "\n")
    
    example_1_saliency_maps()
    example_2_feature_importance()
    example_3_full_dashboard()
    example_4_layer_wise_analysis()
    
    print("=" * 70)
    print("All interpretability examples completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - interpretability_feature_importance.html")
    print("  - interpretability_dashboard.html")
    print("  - explanations_history.json")
    print("\nOpen the HTML files in a browser to view interactive visualizations.")


if __name__ == "__main__":
    main()
