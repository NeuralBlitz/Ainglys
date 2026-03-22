"""
Model Performance Benchmarking Suite

Comprehensive benchmarking tools for ML models:
- Inference latency benchmarking
- Throughput testing
- Memory profiling
- Energy consumption tracking
- Comparative benchmarking
- Regression testing

Example:
    >>> from ai_hub.benchmarking import ModelBenchmark
    >>> benchmark = ModelBenchmark(model)
    >>> results = benchmark.run_full_suite()
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import psutil
import gc
from contextlib import contextmanager
import warnings


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    metric_name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Container for full benchmark suite results."""

    model_name: str
    total_duration_ms: float
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str) -> Optional[BenchmarkResult]:
        """Get a specific metric by name."""
        for result in self.results:
            if result.metric_name == name:
                return result
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "total_duration_ms": self.total_duration_ms,
            "results": [r.to_dict() for r in self.results],
            "system_info": self.system_info,
        }

    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SystemProfiler:
    """Profile system resources during model execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory_mb = None

    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / 1e9,
            "memory_available_gb": psutil.virtual_memory().available / 1e9,
            "python_version": torch.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]

        return info

    @contextmanager
    def profile_memory(self):
        """Context manager to profile memory usage."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024

        yield

        gc.collect()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024 - (
            self.initial_memory_mb or 0
        )

    def get_gpu_memory_mb(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None

    def get_peak_gpu_memory_mb(self) -> Optional[float]:
        """Get peak GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return None


class LatencyBenchmark:
    """Benchmark model inference latency."""

    def __init__(self, warmup_iterations: int = 10):
        self.warmup_iterations = warmup_iterations

    def benchmark(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        num_iterations: int = 100,
        device: str = "cpu",
    ) -> BenchmarkResult:
        """
        Benchmark inference latency.

        Args:
            model: Model to benchmark
            input_data: Sample input data
            num_iterations: Number of inference iterations
            device: Device to run on

        Returns:
            BenchmarkResult with latency statistics
        """
        model = model.to(device)
        model.eval()
        input_data = input_data.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(input_data)

        # Synchronize if using CUDA
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(input_data)

                if device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)

        return BenchmarkResult(
            metric_name="inference_latency",
            value=np.mean(latencies),
            unit="ms",
            metadata={
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "num_iterations": num_iterations,
            },
        )


class ThroughputBenchmark:
    """Benchmark model throughput."""

    def benchmark(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        duration_seconds: float = 10.0,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> BenchmarkResult:
        """
        Benchmark inference throughput.

        Args:
            model: Model to benchmark
            input_data: Sample input data
            duration_seconds: Duration to benchmark
            batch_size: Batch size for inference
            device: Device to run on

        Returns:
            BenchmarkResult with throughput statistics
        """
        model = model.to(device)
        model.eval()

        # Expand input to batch size if needed
        if input_data.shape[0] == 1 and batch_size > 1:
            input_data = input_data.repeat(batch_size, *([1] * (input_data.dim() - 1)))

        input_data = input_data.to(device)

        # Benchmark
        num_inferences = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                _ = model(input_data)
                num_inferences += batch_size

                if device == "cuda":
                    torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        throughput = num_inferences / elapsed

        return BenchmarkResult(
            metric_name="throughput",
            value=throughput,
            unit="inferences/sec",
            metadata={
                "total_inferences": num_inferences,
                "duration_seconds": elapsed,
                "batch_size": batch_size,
                "device": device,
            },
        )


class ModelBenchmark:
    """
    Comprehensive model benchmarking suite.

    Provides a complete set of benchmarks for ML models including
    latency, throughput, memory usage, and system profiling.

    Example:
        >>> benchmark = ModelBenchmark(model, model_name="ResNet50")
        >>> results = benchmark.run_full_suite(input_data)
        >>> results.save("benchmark_results.json")
    """

    def __init__(self, model: nn.Module, model_name: str = "model"):
        """
        Initialize benchmark suite.

        Args:
            model: Model to benchmark
            model_name: Name of the model for reporting
        """
        self.model = model
        self.model_name = model_name
        self.system_profiler = SystemProfiler()
        self.latency_benchmark = LatencyBenchmark()
        self.throughput_benchmark = ThroughputBenchmark()

    def run_full_suite(
        self,
        input_data: torch.Tensor,
        num_latency_iterations: int = 100,
        throughput_duration: float = 10.0,
        batch_sizes: List[int] = None,
    ) -> BenchmarkSuite:
        """
        Run full benchmark suite.

        Args:
            input_data: Sample input data
            num_latency_iterations: Number of iterations for latency test
            throughput_duration: Duration for throughput test
            batch_sizes: List of batch sizes to test

        Returns:
            BenchmarkSuite with all results
        """
        batch_sizes = batch_sizes or [1, 8, 16, 32]
        results = []

        print(f"Running benchmark suite for {self.model_name}...")
        start_time = time.perf_counter()

        # Get system info
        system_info = self.system_profiler.get_system_info()
        print(
            f"System: {system_info.get('cpu_count')} CPUs, "
            f"{system_info.get('memory_total_gb', 0):.1f} GB RAM"
        )

        # CPU Benchmarks
        print("\nBenchmarking on CPU...")

        with self.system_profiler.profile_memory():
            # Latency
            print("  Measuring latency...")
            latency_result = self.latency_benchmark.benchmark(
                self.model, input_data, num_latency_iterations, device="cpu"
            )
            results.append(latency_result)
            print(f"    Mean latency: {latency_result.metadata['mean_ms']:.2f} ms")

            # Memory
            cpu_memory = self.system_profiler.get_memory_usage_mb()
            results.append(
                BenchmarkResult(
                    metric_name="cpu_memory_usage", value=cpu_memory, unit="MB"
                )
            )
            print(f"    Memory usage: {cpu_memory:.2f} MB")

            # Throughput for different batch sizes
            print("  Measuring throughput...")
            for batch_size in batch_sizes:
                throughput_result = self.throughput_benchmark.benchmark(
                    self.model,
                    input_data,
                    throughput_duration,
                    batch_size,
                    device="cpu",
                )
                throughput_result.metric_name = f"throughput_batch_{batch_size}"
                results.append(throughput_result)
                print(f"    Batch {batch_size}: {throughput_result.value:.1f} inf/sec")

        # GPU Benchmarks (if available)
        if torch.cuda.is_available():
            print("\nBenchmarking on GPU...")

            with self.system_profiler.profile_memory():
                # Latency
                print("  Measuring latency...")
                latency_result = self.latency_benchmark.benchmark(
                    self.model, input_data, num_latency_iterations, device="cuda"
                )
                latency_result.metric_name = "gpu_inference_latency"
                results.append(latency_result)
                print(f"    Mean latency: {latency_result.metadata['mean_ms']:.2f} ms")

                # GPU Memory
                gpu_memory = self.system_profiler.get_peak_gpu_memory_mb()
                if gpu_memory:
                    results.append(
                        BenchmarkResult(
                            metric_name="gpu_memory_usage", value=gpu_memory, unit="MB"
                        )
                    )
                    print(f"    GPU memory: {gpu_memory:.2f} MB")

                # Throughput
                print("  Measuring throughput...")
                for batch_size in batch_sizes[:2]:  # Test fewer sizes on GPU
                    throughput_result = self.throughput_benchmark.benchmark(
                        self.model,
                        input_data,
                        throughput_duration,
                        batch_size,
                        device="cuda",
                    )
                    throughput_result.metric_name = f"gpu_throughput_batch_{batch_size}"
                    results.append(throughput_result)
                    print(
                        f"    Batch {batch_size}: {throughput_result.value:.1f} inf/sec"
                    )

        total_duration = (time.perf_counter() - start_time) * 1000

        return BenchmarkSuite(
            model_name=self.model_name,
            total_duration_ms=total_duration,
            results=results,
            system_info=system_info,
        )

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_data: torch.Tensor,
        metric: str = "inference_latency",
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple models on the same metric.

        Args:
            models: Dictionary of model names to models
            input_data: Sample input data
            metric: Metric to compare

        Returns:
            Dictionary of model names to benchmark results
        """
        results = {}

        print(f"Comparing {len(models)} models on {metric}...")

        for name, model in models.items():
            print(f"\nBenchmarking {name}...")
            benchmark = ModelBenchmark(model, name)
            suite = benchmark.run_full_suite(input_data, num_latency_iterations=50)

            result = suite.get_metric(metric)
            if result:
                results[name] = result
                print(f"  {metric}: {result.value:.2f} {result.unit}")

        # Find best model
        if results:
            best_model = min(results.items(), key=lambda x: x[1].value)
            print(
                f"\nBest model: {best_model[0]} ({best_model[1].value:.2f} {best_model[1].unit})"
            )

        return results


def example_usage():
    """Example usage of the benchmarking suite."""
    # Create test models
    model_a = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    model_b = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Create sample input
    input_data = torch.randn(1, 784)

    # Benchmark single model
    print("=" * 70)
    print("Single Model Benchmark")
    print("=" * 70)

    benchmark = ModelBenchmark(model_a, "Model_A")
    results = benchmark.run_full_suite(input_data)

    # Save results
    results.save("benchmark_model_a.json")
    print(f"\nResults saved to benchmark_model_a.json")

    # Compare models
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)

    models = {"Model_A": model_a, "Model_B": model_b}

    comparison = benchmark.compare_models(models, input_data)


if __name__ == "__main__":
    example_usage()
