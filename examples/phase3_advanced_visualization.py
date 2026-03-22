"""
Phase 3 Advanced Visualization Examples

Demonstrates all advanced visualization features:
- Training animations
- Attention visualization
- Architecture diagrams
- Model comparisons
- Feature importance
- Confusion matrices
- Embedding visualizations
- Experiment timelines
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

print("=" * 70)
print("PHASE 3: ADVANCED VISUALIZATION EXAMPLES")
print("=" * 70)

# Create output directory
output_dir = Path("outputs/phase3_viz")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXAMPLE 1: Training Animation
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 1: Training Animation")
print("=" * 70)

from ai_hub.advanced_viz import TrainingAnimator, AnimationConfig

animator = TrainingAnimator(AnimationConfig(fps=10, duration=3))

# Simulate training progression
print("\n🎬 Simulating training frames...")
for epoch in range(20):
    # Simulate improving metrics
    loss = 1.0 * np.exp(-epoch / 5)
    accuracy = 1.0 - np.exp(-epoch / 3)

    animator.add_frame(epoch=epoch, metrics={"loss": loss, "accuracy": accuracy})

    if epoch % 5 == 0:
        print(f"   Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.2%}")

# Create animation
animation_path = output_dir / "training_animation.gif"
try:
    success = animator.create_training_animation(str(animation_path))
    if success:
        print(f"\n✅ Animation saved to: {animation_path}")
    else:
        print("\n⚠️  Animation requires matplotlib (install: pip install matplotlib)")
except Exception as e:
    print(f"\n⚠️  Could not create animation: {e}")

# ============================================================================
# EXAMPLE 2: Attention Visualization
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Attention Visualization")
print("=" * 70)

from ai_hub.advanced_viz import AttentionVisualizer

viz = AttentionVisualizer()

# Create sample attention data
tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
seq_len = len(tokens)

# Single attention matrix
attention_matrix = np.random.rand(seq_len, seq_len)
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

# Make it look more realistic (diagonal-ish)
for i in range(seq_len):
    attention_matrix[i, i] += 0.3
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

print(f"\n🔍 Creating attention heatmap for {seq_len} tokens...")
heatmap = viz.visualize_attention_heatmap(tokens, attention_matrix)

if heatmap:
    print(f"✅ Attention heatmap created")
    print(f"   Tokens: {tokens}")
else:
    print("⚠️  Plotly not available (install: pip install plotly)")

# Multi-head attention
print("\n🔍 Creating multi-head attention visualization...")
num_heads = 4
attention_heads = [np.random.rand(seq_len, seq_len) for _ in range(num_heads)]
# Normalize
attention_heads = [a / a.sum(axis=1, keepdims=True) for a in attention_heads]

multi_head_viz = viz.visualize_multi_head_attention(attention_heads, tokens)

if multi_head_viz:
    print(f"✅ Multi-head attention visualization created ({num_heads} heads)")

# ============================================================================
# EXAMPLE 3: Architecture Diagrams
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Architecture Diagrams")
print("=" * 70)

from ai_hub.advanced_viz import ArchitectureDiagrammer

diagrammer = ArchitectureDiagrammer()

# Create a model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

print("\n📐 Generating architecture diagram...")
model_diagram_path = output_dir / "model_architecture"
try:
    result = diagrammer.generate_model_diagram(model, str(model_diagram_path))
    if result:
        print(f"✅ Model diagram saved to: {result}")
    else:
        print("⚠️  Graphviz not available (install: pip install graphviz)")
except Exception as e:
    print(f"⚠️  Could not generate diagram: {e}")

print("\n📐 Generating computation graph...")
comp_graph_path = output_dir / "computation_graph"
try:
    result = diagrammer.generate_computation_graph(model, (784,), str(comp_graph_path))
    if result:
        print(f"✅ Computation graph saved to: {result}")
except Exception as e:
    print(f"⚠️  Could not generate graph: {e}")

# ============================================================================
# EXAMPLE 4: Model Comparison
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Model Comparison Visualization")
print("=" * 70)

from ai_hub.advanced_viz import ModelComparisonVisualizer

comparer = ModelComparisonVisualizer()

# Add models with training curves
epochs = 50

# Model A - Baseline
comparer.add_model(
    name="Baseline",
    metrics={
        "loss": [1.0 * np.exp(-e / 10) + 0.1 for e in range(epochs)],
        "accuracy": [1 - np.exp(-e / 8) for e in range(epochs)],
    },
    color="blue",
)

# Model B - Improved
comparer.add_model(
    name="Improved",
    metrics={
        "loss": [1.0 * np.exp(-e / 7) + 0.05 for e in range(epochs)],
        "accuracy": [1 - np.exp(-e / 5) for e in range(epochs)],
    },
    color="green",
)

# Model C - Best
comparer.add_model(
    name="Best",
    metrics={
        "loss": [1.0 * np.exp(-e / 5) + 0.02 for e in range(epochs)],
        "accuracy": [1 - np.exp(-e / 4) for e in range(epochs)],
    },
    color="red",
)

print("\n📊 Creating training curves comparison...")
comparison = comparer.plot_training_curves_comparison("loss")
if comparison:
    print("✅ Training curves comparison created")

print("\n📊 Creating final metrics comparison...")
final_metrics = {
    "Baseline": {"accuracy": 0.85, "f1": 0.84, "precision": 0.86, "recall": 0.83},
    "Improved": {"accuracy": 0.90, "f1": 0.89, "precision": 0.91, "recall": 0.88},
    "Best": {"accuracy": 0.95, "f1": 0.94, "precision": 0.95, "recall": 0.93},
}

radar = comparer.plot_metrics_radar(final_metrics)
if radar:
    print("✅ Radar chart created")

bar_chart = comparer.plot_metrics_bar(final_metrics)
if bar_chart:
    print("✅ Bar chart created")

# ============================================================================
# EXAMPLE 5: Feature Importance
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 5: Feature Importance Visualization")
print("=" * 70)

from ai_hub.advanced_viz import FeatureImportanceVisualizer

importance_viz = FeatureImportanceVisualizer()

# Generate feature importance data
feature_names = [f"feature_{i}" for i in range(20)]
importance_scores = np.random.exponential(1.0, 20)
importance_scores = importance_scores / importance_scores.sum()

print("\n📊 Creating feature importance plot...")
importance_plot = importance_viz.plot_feature_importance(
    feature_names, importance_scores, top_n=15
)

if importance_plot:
    print("✅ Feature importance plot created")
    print(
        f"   Top 3 features: {np.array(feature_names)[np.argsort(importance_scores)[-3:]]}"
    )

# Permutation importance
print("\n📊 Creating permutation importance plot...")
baseline_score = 0.85
permuted_scores = {
    f"feature_{i}": np.random.normal(0.8 - importance_scores[i], 0.05, 10)
    for i in range(10)
}

perm_plot = importance_viz.plot_permutation_importance(
    feature_names[:10], baseline_score, permuted_scores
)

if perm_plot:
    print("✅ Permutation importance plot created")

# ============================================================================
# EXAMPLE 6: Confusion Matrix and ROC Curves
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 6: Confusion Matrix and ROC Curves")
print("=" * 70)

from ai_hub.advanced_viz import ConfusionMatrixVisualizer

cm_viz = ConfusionMatrixVisualizer()

# Confusion matrix
class_names = ["Cat", "Dog", "Bird", "Fish"]
cm = np.array(
    [
        [45, 3, 1, 1],  # Cat
        [2, 48, 0, 0],  # Dog
        [1, 0, 47, 2],  # Bird
        [0, 1, 2, 47],  # Fish
    ]
)

print("\n📊 Creating confusion matrix...")
cm_plot = cm_viz.plot_confusion_matrix(cm, class_names)

if cm_plot:
    print("✅ Confusion matrix created")
    print(f"   Classes: {class_names}")
    print(f"   Total samples: {cm.sum()}")

# ROC curves
print("\n📊 Creating ROC curves...")
roc_data = {
    "Model A": (
        np.array([0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),  # FPR
        np.array([0, 0.5, 0.7, 0.85, 0.92, 0.97, 1.0]),  # TPR
        0.85,  # AUC
    ),
    "Model B": (
        np.array([0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0]),
        np.array([0, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0]),
        0.92,
    ),
    "Model C": (
        np.array([0, 0.02, 0.1, 0.25, 0.45, 0.65, 1.0]),
        np.array([0, 0.65, 0.85, 0.94, 0.97, 0.99, 1.0]),
        0.96,
    ),
}

roc_plot = cm_viz.plot_roc_curves(roc_data)
if roc_plot:
    print("✅ ROC curves created")
    for name, (_, _, auc) in roc_data.items():
        print(f"   {name}: AUC = {auc:.3f}")

# ============================================================================
# EXAMPLE 7: Embedding Visualization
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 7: Embedding Visualization (t-SNE)")
print("=" * 70)

from ai_hub.advanced_viz import EmbeddingVisualizer

embedding_viz = EmbeddingVisualizer()

# Generate sample embeddings
np.random.seed(42)
n_samples = 200
n_features = 50
embeddings = np.random.randn(n_samples, n_features)

# Add some structure (clusters)
for i in range(0, n_samples, 50):
    embeddings[i : i + 25] += np.random.randn(1, n_features) * 2

# Labels
labels = np.array([i // 50 for i in range(n_samples)])

print("\n🎯 Creating t-SNE visualization...")
tsne_plot = embedding_viz.plot_tsne(embeddings, labels, perplexity=30)

if tsne_plot:
    print("✅ t-SNE visualization created")
    print(f"   Samples: {n_samples}")
    print(f"   Dimensions: {n_features} → 2")
    print(f"   Classes: {len(np.unique(labels))}")
else:
    print("⚠️  scikit-learn required (install: pip install scikit-learn)")

# ============================================================================
# EXAMPLE 8: Experiment Timeline
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 8: Experiment Timeline")
print("=" * 70)

from ai_hub.advanced_viz import ExperimentTimelineVisualizer

timeline_viz = ExperimentTimelineVisualizer()

# Create experiment data
from datetime import datetime, timedelta

base_time = datetime.now()

experiments = [
    {
        "name": "Baseline Training",
        "start_time": (base_time).isoformat(),
        "end_time": (base_time + timedelta(hours=2.5)).isoformat(),
    },
    {
        "name": "Hyperparameter Search",
        "start_time": (base_time + timedelta(hours=3)).isoformat(),
        "end_time": (base_time + timedelta(hours=8)).isoformat(),
    },
    {
        "name": "Final Training",
        "start_time": (base_time + timedelta(hours=8.5)).isoformat(),
        "end_time": (base_time + timedelta(hours=12)).isoformat(),
    },
    {
        "name": "Evaluation",
        "start_time": (base_time + timedelta(hours=12.5)).isoformat(),
        "end_time": (base_time + timedelta(hours=13.5)).isoformat(),
    },
]

print("\n📅 Creating experiment timeline...")
timeline_plot = timeline_viz.plot_experiment_timeline(experiments)

if timeline_plot:
    print("✅ Experiment timeline created")
    for exp in experiments:
        start = datetime.fromisoformat(exp["start_time"])
        end = datetime.fromisoformat(exp["end_time"])
        duration = (end - start).total_seconds() / 3600
        print(f"   {exp['name']}: {duration:.1f} hours")

# ============================================================================
# EXAMPLE 9: Hyperparameter Parallel Coordinates
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 9: Hyperparameter Parallel Coordinates")
print("=" * 70)

from ai_hub.advanced_viz import HyperparameterVisualizer

hp_viz = HyperparameterVisualizer()

# Create trial data
trials = []
for i in range(30):
    lr = np.random.choice([0.1, 0.01, 0.001, 0.0001])
    batch_size = np.random.choice([16, 32, 64, 128])
    dropout = np.random.uniform(0.1, 0.5)

    # Score correlated with hyperparameters
    score = (
        0.7
        + 0.2 * (lr == 0.001)
        + 0.05 * (batch_size == 32)
        + np.random.normal(0, 0.02)
    )
    score = np.clip(score, 0, 1)

    trials.append(
        {
            "params": {
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
            },
            "score": score,
        }
    )

print("\n📈 Creating parallel coordinates plot...")
pc_plot = hp_viz.plot_parallel_coordinates(trials, metric="score")

if pc_plot:
    print("✅ Parallel coordinates plot created")
    print(f"   Trials: {len(trials)}")
    best_trial = max(trials, key=lambda x: x["score"])
    print(f"   Best score: {best_trial['score']:.3f}")
    print(f"   Best params: {best_trial['params']}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL PHASE 3 ADVANCED VISUALIZATION EXAMPLES COMPLETED!")
print("=" * 70)
print("\nGenerated visualizations:")
print("  • Training animations")
print("  • Attention heatmaps (single & multi-head)")
print("  • Architecture diagrams")
print("  • Model comparisons (curves, radar, bar)")
print("  • Feature importance")
print("  • Confusion matrix")
print("  • ROC curves")
print("  • t-SNE embeddings")
print("  • Experiment timeline")
print("  • Hyperparameter parallel coordinates")
print(f"\nOutput directory: {output_dir}")
print("=" * 70)

# Export note
print("\n💡 To export visualizations:")
print("   from ai_hub.visualization import DashboardManager")
print("   DashboardManager.export_plot(plot_dict, 'output.html')")
print("=" * 70)
