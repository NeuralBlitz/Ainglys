"""
AutoML Hyperparameter Optimization System

Provides automated hyperparameter tuning with multiple strategies:
- Grid Search
- Random Search
- Bayesian Optimization
- Population Based Training (PBT)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import warnings


@dataclass
class HyperparameterSpace:
    """Defines hyperparameter search space."""

    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical', 'log_uniform'
    bounds: Tuple[float, float]  # For continuous
    choices: Optional[List[Any]] = None  # For categorical

    def sample(self) -> Any:
        """Sample a value from the space."""
        if self.param_type == "continuous":
            return np.random.uniform(self.bounds[0], self.bounds[1])
        elif self.param_type == "log_uniform":
            log_low = np.log(self.bounds[0])
            log_high = np.log(self.bounds[1])
            return np.exp(np.random.uniform(log_low, log_high))
        elif self.param_type == "discrete":
            return np.random.randint(int(self.bounds[0]), int(self.bounds[1]) + 1)
        elif self.param_type == "categorical":
            return np.random.choice(self.choices)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    strategy: str = "bayesian"  # grid, random, bayesian, pbt
    max_iterations: int = 100
    num_workers: int = 1
    objective_metric: str = "val_loss"
    objective_mode: str = "min"  # min or max
    early_stopping_patience: int = 10
    random_seed: int = 42
    output_dir: str = "outputs/automl"


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for PyTorch models.

    Supports multiple optimization strategies:
    - Grid Search: Exhaustive search over specified parameter values
    - Random Search: Random sampling from parameter space
    - Bayesian Optimization: Smart search using Gaussian Processes
    - Population Based Training: Evolutionary optimization
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.search_space: Dict[str, HyperparameterSpace] = {}
        self.history: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_score: Optional[float] = None

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        np.random.seed(self.config.random_seed)

    def define_search_space(self, **params):
        """
        Define the hyperparameter search space.

        Example:
            optimizer.define_search_space(
                learning_rate=HyperparameterSpace(
                    name="lr", param_type="log_uniform", bounds=(1e-5, 1e-1)
                ),
                batch_size=HyperparameterSpace(
                    name="batch_size", param_type="discrete", bounds=(16, 128)
                ),
                optimizer=HyperparameterSpace(
                    name="optimizer", param_type="categorical",
                    bounds=(0, 0), choices=["adam", "sgd", "adamw"]
                )
            )
        """
        self.search_space = params

    def optimize(
        self,
        model_fn: Callable[[Dict], nn.Module],
        train_fn: Callable[[nn.Module, Dict], float],
        val_fn: Optional[Callable[[nn.Module], float]] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            model_fn: Function that creates model given hyperparameters
            train_fn: Function that trains model and returns metric
            val_fn: Optional validation function

        Returns:
            Dictionary with best hyperparameters and optimization history
        """
        print(f"🔍 Starting {self.config.strategy} hyperparameter optimization...")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Search space: {list(self.search_space.keys())}")

        if self.config.strategy == "grid":
            return self._grid_search(model_fn, train_fn, val_fn)
        elif self.config.strategy == "random":
            return self._random_search(model_fn, train_fn, val_fn)
        elif self.config.strategy == "bayesian":
            return self._bayesian_optimization(model_fn, train_fn, val_fn)
        elif self.config.strategy == "pbt":
            return self._population_based_training(model_fn, train_fn, val_fn)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _grid_search(
        self, model_fn: Callable, train_fn: Callable, val_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Exhaustive grid search."""
        # Generate all combinations (limited for continuous params)
        param_values = {}
        for name, space in self.search_space.items():
            if space.param_type in ["continuous", "log_uniform"]:
                # Sample 5 values for continuous params
                if space.param_type == "log_uniform":
                    values = np.logspace(
                        np.log10(space.bounds[0]), np.log10(space.bounds[1]), 5
                    )
                else:
                    values = np.linspace(space.bounds[0], space.bounds[1], 5)
                param_values[name] = values.tolist()
            elif space.param_type == "discrete":
                param_values[name] = list(
                    range(int(space.bounds[0]), int(space.bounds[1]) + 1)
                )
            else:  # categorical
                param_values[name] = space.choices

        # Generate all combinations
        from itertools import product

        keys = list(param_values.keys())
        values_lists = [param_values[k] for k in keys]

        total_combinations = 1
        for v in values_lists:
            total_combinations *= len(v)

        print(f"   Grid search: {total_combinations} combinations")

        best_score = (
            float("inf") if self.config.objective_mode == "min" else float("-inf")
        )
        best_params = None

        for i, combination in enumerate(product(*values_lists)):
            if i >= self.config.max_iterations:
                break

            params = dict(zip(keys, combination))
            score = self._evaluate_params(params, model_fn, train_fn, val_fn, i)

            if self._is_better(score, best_score):
                best_score = score
                best_params = params
                print(f"   ✨ New best: {score:.4f} with {params}")

        self.best_params = best_params
        self.best_score = best_score

        return self._get_results()

    def _random_search(
        self, model_fn: Callable, train_fn: Callable, val_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Random search over parameter space."""
        best_score = (
            float("inf") if self.config.objective_mode == "min" else float("-inf")
        )
        best_params = None

        for i in range(self.config.max_iterations):
            # Sample parameters
            params = {name: space.sample() for name, space in self.search_space.items()}

            score = self._evaluate_params(params, model_fn, train_fn, val_fn, i)

            if self._is_better(score, best_score):
                best_score = score
                best_params = params
                print(f"   ✨ New best: {score:.4f} with {params}")

        self.best_params = best_params
        self.best_score = best_score

        return self._get_results()

    def _bayesian_optimization(
        self, model_fn: Callable, train_fn: Callable, val_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        Bayesian optimization using Gaussian Processes.

        Uses scikit-optimize if available, otherwise falls back to random search.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical

            # Convert search space to skopt format
            dimensions = []
            param_names = []

            for name, space in self.search_space.items():
                param_names.append(name)
                if space.param_type == "continuous":
                    dimensions.append(Real(space.bounds[0], space.bounds[1]))
                elif space.param_type == "log_uniform":
                    dimensions.append(
                        Real(space.bounds[0], space.bounds[1], prior="log-uniform")
                    )
                elif space.param_type == "discrete":
                    dimensions.append(
                        Integer(int(space.bounds[0]), int(space.bounds[1]))
                    )
                else:  # categorical
                    dimensions.append(Categorical(space.choices))

            # Objective function
            def objective(x):
                params = dict(zip(param_names, x))
                score = self._evaluate_params(
                    params, model_fn, train_fn, val_fn, len(self.history)
                )
                # skopt always minimizes, so negate if we want to maximize
                return score if self.config.objective_mode == "min" else -score

            # Run optimization
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=self.config.max_iterations,
                n_initial_points=10,
                random_state=self.config.random_seed,
            )

            self.best_params = dict(zip(param_names, result.x))
            self.best_score = (
                result.fun if self.config.objective_mode == "min" else -result.fun
            )

        except ImportError:
            print("   ⚠️ scikit-optimize not available, falling back to random search")
            return self._random_search(model_fn, train_fn, val_fn)

        return self._get_results()

    def _population_based_training(
        self, model_fn: Callable, train_fn: Callable, val_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        Population Based Training (PBT) - evolutionary optimization.

        Maintains a population of models and evolves hyperparameters.
        """
        population_size = max(4, self.config.num_workers)
        generations = self.config.max_iterations // population_size

        # Initialize population
        population = []
        for i in range(population_size):
            member = {
                "params": {
                    name: space.sample() for name, space in self.search_space.items()
                },
                "score": None,
                "model": None,
            }
            population.append(member)

        best_score = (
            float("inf") if self.config.objective_mode == "min" else float("-inf")
        )
        best_params = None

        for generation in range(generations):
            print(f"   Generation {generation + 1}/{generations}")

            # Evaluate population
            for i, member in enumerate(population):
                score = self._evaluate_params(
                    member["params"],
                    model_fn,
                    train_fn,
                    val_fn,
                    generation * population_size + i,
                )
                member["score"] = score

                if self._is_better(score, best_score):
                    best_score = score
                    best_params = member["params"].copy()

            # Sort by performance
            population.sort(
                key=lambda x: x["score"], reverse=(self.config.objective_mode == "max")
            )

            # Exploit: Replace worst with best
            num_to_replace = population_size // 2
            for i in range(num_to_replace):
                worst_idx = population_size - 1 - i
                best_idx = i

                # Copy params from best to worst
                population[worst_idx]["params"] = population[best_idx]["params"].copy()

                # Explore: Perturb params
                for name, space in self.search_space.items():
                    if np.random.random() < 0.2:  # 20% mutation rate
                        population[worst_idx]["params"][name] = space.sample()

        self.best_params = best_params
        self.best_score = best_score

        return self._get_results()

    def _evaluate_params(
        self,
        params: Dict,
        model_fn: Callable,
        train_fn: Callable,
        val_fn: Optional[Callable],
        iteration: int,
    ) -> float:
        """Evaluate a set of hyperparameters."""
        print(f"   Iteration {iteration + 1}: {params}")

        try:
            # Create model
            model = model_fn(params)

            # Train
            metric = train_fn(model, params)

            # Validate if provided
            if val_fn:
                metric = val_fn(model)

            # Record
            result = {
                "iteration": iteration,
                "params": params,
                "score": metric,
                "timestamp": datetime.now().isoformat(),
            }
            self.history.append(result)

            return metric

        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            # Return bad score on error
            return (
                float("inf") if self.config.objective_mode == "min" else float("-inf")
            )

    def _is_better(self, score: float, best_score: float) -> bool:
        """Check if score is better than best_score."""
        if self.config.objective_mode == "min":
            return score < best_score
        else:
            return score > best_score

    def _get_results(self) -> Dict[str, Any]:
        """Get optimization results."""
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "history": self.history,
            "config": {
                "strategy": self.config.strategy,
                "max_iterations": self.config.max_iterations,
                "objective_metric": self.config.objective_metric,
                "objective_mode": self.config.objective_mode,
            },
        }

        # Save results
        output_file = (
            Path(self.config.output_dir)
            / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Optimization complete!")
        print(f"   Best score: {self.best_score:.4f}")
        print(f"   Best params: {self.best_params}")
        print(f"   Results saved to: {output_file}")

        return results

    def plot_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt

            iterations = [h["iteration"] for h in self.history]
            scores = [h["score"] for h in self.history]

            plt.figure(figsize=(10, 6))
            plt.plot(iterations, scores, "b-", alpha=0.7)
            plt.scatter(iterations, scores, c="blue", alpha=0.5)
            plt.xlabel("Iteration")
            plt.ylabel(self.config.objective_metric)
            plt.title(f"{self.config.strategy.upper()} Hyperparameter Optimization")
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

        except ImportError:
            print("   ⚠️ matplotlib not available for plotting")


# Convenience functions for common optimization tasks


def quick_optimize(
    model_fn: Callable,
    train_fn: Callable,
    search_space: Dict[str, HyperparameterSpace],
    max_iterations: int = 50,
    strategy: str = "bayesian",
) -> Dict[str, Any]:
    """
    Quick hyperparameter optimization with sensible defaults.

    Example:
        best = quick_optimize(
            model_fn=lambda p: create_model(p['hidden_dim']),
            train_fn=lambda m, p: train_model(m, lr=p['learning_rate']),
            search_space={
                'learning_rate': HyperparameterSpace(
                    'lr', 'log_uniform', (1e-4, 1e-2)
                ),
                'hidden_dim': HyperparameterSpace(
                    'hidden', 'discrete', (64, 512)
                )
            }
        )
    """
    config = OptimizationConfig(strategy=strategy, max_iterations=max_iterations)

    optimizer = HyperparameterOptimizer(config)
    optimizer.define_search_space(**search_space)

    return optimizer.optimize(model_fn, train_fn)


# Export
__all__ = [
    "HyperparameterOptimizer",
    "HyperparameterSpace",
    "OptimizationConfig",
    "quick_optimize",
]
