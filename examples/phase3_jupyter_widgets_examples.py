"""
Phase 3 Jupyter Widgets Examples

Demonstrates interactive training visualization with Jupyter widgets:
- Live training metrics dashboards
- Real-time loss/accuracy plots
- GPU monitoring
- Interactive hyperparameter adjustment
- Training controls (pause/resume/stop)
"""

import torch
import torch.nn as nn
import numpy as np
import time


def example_1_live_plotter():
    """Example 1: Using the live plotter for real-time visualization."""
    print("=" * 70)
    print("Example 1: Live Training Plotter")
    print("=" * 70)
    print()
    
    try:
        from src.ai_hub.training_widgets import LivePlotter
        
        plotter = LivePlotter(max_points=500)
        
        print("Simulating training process...")
        print("(In Jupyter notebook, you would see live updating plots)")
        print()
        
        # Simulate training epochs
        for epoch in range(50):
            # Simulate metrics with some noise
            train_loss = 2.0 * np.exp(-0.05 * epoch) + np.random.randn() * 0.1
            val_loss = 2.2 * np.exp(-0.05 * epoch) + np.random.randn() * 0.12
            train_acc = 0.5 + 0.49 * (1 - np.exp(-0.05 * epoch)) + np.random.randn() * 0.02
            val_acc = 0.48 + 0.47 * (1 - np.exp(-0.05 * epoch)) + np.random.randn() * 0.025
            lr = 0.001 * (0.95 ** epoch)
            
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': lr
            }
            
            # In Jupyter: plotter.update(metrics, epoch)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}, LR={lr:.6f}")
        
        print()
        print("✓ Live plotter simulation complete!")
        print("  (In Jupyter: plotter.show() would display interactive Plotly figure)")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("  Install with: pip install ipywidgets plotly")
    
    print()


def example_2_training_dashboard():
    """Example 2: Full training dashboard with controls."""
    print("=" * 70)
    print("Example 2: Training Dashboard")
    print("=" * 70)
    print()
    
    try:
        from src.ai_hub.training_widgets import TrainingDashboard
        
        # Create dashboard
        print("Creating training dashboard...")
        dashboard = TrainingDashboard(update_interval=1.0)
        
        print("Dashboard components:")
        print("  ✓ Progress bar")
        print("  ✓ Live metric displays")
        print("  ✓ Interactive Plotly plots")
        print("  ✓ Control buttons (Pause/Resume/Stop)")
        print("  ✓ GPU monitoring (if available)")
        print()
        
        # Simulate training
        print("Simulating training loop...")
        for epoch in range(20):
            metrics = {
                'train_loss': 1.0 / (epoch + 1) + np.random.randn() * 0.1,
                'val_loss': 1.1 / (epoch + 1) + np.random.randn() * 0.1,
                'train_acc': min(0.99, epoch / 20 + np.random.randn() * 0.05),
                'val_acc': min(0.95, epoch / 20 + np.random.randn() * 0.05),
                'lr': 0.001 * (0.95 ** epoch)
            }
            
            # In Jupyter: dashboard.update_metrics(metrics, epoch, total_epochs=20)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:2d}: Train Loss={metrics['train_loss']:.4f}, "
                      f"Val Acc={metrics['val_acc']:.4f}")
            
            time.sleep(0.1)  # Simulate epoch time
        
        print()
        print("✓ Training dashboard simulation complete!")
        print("  (In Jupyter: call dashboard.display() to show widgets)")
        
        # Show history structure
        print("\nDashboard features:")
        print("  - Metrics history tracking")
        print("  - Real-time GPU utilization monitoring")
        print("  - Interactive pause/resume/stop controls")
        print("  - Export training history to JSON")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("  Install with: pip install ipywidgets plotly")
    
    print()


def example_3_hyperparameter_widget():
    """Example 3: Interactive hyperparameter adjustment."""
    print("=" * 70)
    print("Example 3: Hyperparameter Widget")
    print("=" * 70)
    print()
    
    try:
        from src.ai_hub.training_widgets import HyperparameterWidget
        
        # Define custom hyperparameter configuration
        param_config = {
            'learning_rate': {
                'type': 'float',
                'min': 1e-6,
                'max': 1e-1,
                'default': 1e-3,
                'step': 1e-6,
                'description': 'Learning Rate:'
            },
            'batch_size': {
                'type': 'int',
                'min': 8,
                'max': 256,
                'default': 32,
                'step': 8,
                'description': 'Batch Size:'
            },
            'dropout_rate': {
                'type': 'float',
                'min': 0.0,
                'max': 0.9,
                'default': 0.5,
                'step': 0.05,
                'description': 'Dropout:'
            },
            'num_layers': {
                'type': 'int',
                'min': 1,
                'max': 10,
                'default': 3,
                'step': 1,
                'description': 'Num Layers:'
            }
        }
        
        print("Creating hyperparameter widget...")
        hp_widget = HyperparameterWidget(param_config)
        
        # Show current values
        print("\nDefault hyperparameters:")
        values = hp_widget.get_values()
        for param, value in values.items():
            print(f"  {param}: {value}")
        
        # Simulate changing a value
        print("\nSimulating parameter change...")
        hp_widget.set_values({'learning_rate': 5e-4, 'batch_size': 64})
        
        new_values = hp_widget.get_values()
        print("Updated hyperparameters:")
        for param, value in new_values.items():
            print(f"  {param}: {value}")
        
        print()
        print("✓ Hyperparameter widget demo complete!")
        print("  (In Jupyter: call hp_widget.display() to show interactive sliders)")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("  Install with: pip install ipywidgets")
    
    print()


def example_4_integration_with_trainer():
    """Example 4: Integrating widgets with a custom trainer."""
    print("=" * 70)
    print("Example 4: Integration with Custom Trainer")
    print("=" * 70)
    print()
    
    # Define a simple trainer class
    class SimpleTrainer:
        """Example trainer class that works with widgets."""
        
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.current_metrics = {}
            self.epoch = 0
        
        def train_epoch(self, data_loader):
            """Train for one epoch."""
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            self.current_metrics = {
                'train_loss': total_loss / len(data_loader),
                'train_acc': correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            self.epoch += 1
            return self.current_metrics
    
    try:
        from src.ai_hub.training_widgets import TrainingDashboard
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = SimpleTrainer(model, optimizer)
        
        # Create dashboard
        dashboard = TrainingDashboard()
        
        print("Example trainer setup:")
        print("  Model: Simple MLP (10 -> 50 -> 2)")
        print("  Optimizer: Adam (lr=0.001)")
        print("  Dashboard: TrainingDashboard")
        print()
        
        print("In Jupyter notebook:")
        print("  1. dashboard.display()  # Show dashboard")
        print("  2. dashboard.start_training_monitoring(trainer)  # Auto-update")
        print("  3. Run training loop...")
        print()
        
        print("✓ Integration example complete!")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("  Install with: pip install ipywidgets plotly")
    
    print()


def example_5_export_and_resume():
    """Example 5: Exporting training history and resuming."""
    print("=" * 70)
    print("Example 5: Export and Resume Training")
    print("=" * 70)
    print()
    
    try:
        from src.ai_hub.training_widgets import TrainingDashboard
        import json
        
        # Simulate training with history
        dashboard = TrainingDashboard()
        
        print("Simulating 30 epochs of training...")
        for epoch in range(30):
            metrics = {
                'train_loss': 2.0 * np.exp(-0.05 * epoch) + np.random.randn() * 0.1,
                'val_loss': 2.1 * np.exp(-0.05 * epoch) + np.random.randn() * 0.1,
                'train_acc': min(0.98, 0.5 + 0.48 * (1 - np.exp(-0.05 * epoch))),
                'val_acc': min(0.95, 0.48 + 0.47 * (1 - np.exp(-0.05 * epoch))),
                'lr': 0.001 * (0.95 ** epoch)
            }
            dashboard.update_metrics(metrics, epoch)
        
        # Export history
        export_path = "training_history.json"
        dashboard.export_history(export_path)
        
        print(f"\n✓ Training history exported to: {export_path}")
        
        # Load and verify
        with open(export_path, 'r') as f:
            history = json.load(f)
        
        print(f"\nExported history contains:")
        for key, values in history.items():
            if isinstance(values, list):
                print(f"  {key}: {len(values)} data points")
                if len(values) > 0:
                    print(f"    Range: [{min(values):.4f}, {max(values):.4f}]")
        
        print("\n✓ History export verified!")
        print("  You can load this history to resume training visualization")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("  Install with: pip install ipywidgets plotly")
    
    print()


def main():
    """Run all Jupyter widgets examples."""
    print("\n" + "=" * 70)
    print("AI HUB - Phase 3: Jupyter Training Widgets Examples")
    print("=" * 70)
    print()
    print("Note: These examples demonstrate widget functionality.")
    print("For full interactive experience, run in Jupyter notebook:")
    print("  jupyter notebook")
    print()
    
    example_1_live_plotter()
    example_2_training_dashboard()
    example_3_hyperparameter_widget()
    example_4_integration_with_trainer()
    example_5_export_and_resume()
    
    print("=" * 70)
    print("All Jupyter widgets examples completed!")
    print("=" * 70)
    print()
    print("To use in Jupyter Notebook:")
    print("  1. from ai_hub.training_widgets import TrainingDashboard")
    print("  2. dashboard = TrainingDashboard()")
    print("  3. dashboard.display()")
    print("  4. Start training and call dashboard.update_metrics() each epoch")


if __name__ == "__main__":
    main()
