import numpy as np
import torch
from typing import List, Dict, Any, Callable, Optional, Tuple
from pathlib import Path
import json


class DataPipeline:
    """Data preprocessing pipeline."""

    def __init__(self):
        self.steps: List[Tuple[str, Callable]] = []

    def add_step(self, name: str, func: Callable) -> "DataPipeline":
        """Add a preprocessing step."""
        self.steps.append((name, func))
        return self

    def transform(self, data: Any) -> Any:
        """Apply all steps."""
        result = data
        for name, func in self.steps:
            result = func(result)
        return result

    def fit_transform(self, data: Any, fit_func: Optional[Callable] = None) -> Any:
        """Fit and transform."""
        if fit_func:
            fit_func(data)
        return self.transform(data)


class Normalizer:
    """Data normalizer."""

    def __init__(
        self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
    ):
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class DataAugmentation:
    """Data augmentation for training."""

    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        noise = np.random.randn(*data.shape) * noise_level
        return data + noise

    @staticmethod
    def scale(
        data: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        scale = np.random.uniform(*scale_range)
        return data * scale

    @staticmethod
    def shuffle(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.permutation(len(data))
        return data[indices], labels[indices]

    @staticmethod
    def split(
        data: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        split_idx = int(len(data) * train_ratio)
        return (
            (data[:split_idx], labels[:split_idx]),
            (data[split_idx:], labels[split_idx:]),
        )


class DatasetLoader:
    """Simple dataset loader."""

    @staticmethod
    def load_numpy(path: str) -> np.ndarray:
        return np.load(path)

    @staticmethod
    def save_numpy(data: np.ndarray, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)

    @staticmethod
    def load_json(path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def generate_synthetic(
        n_samples: int, n_features: int, n_classes: int = 10, noise: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data."""
        from sklearn.datasets import make_classification

        data, labels = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_classes=n_classes,
            noise=noise,
            random_state=42,
        )
        return data, labels

    @staticmethod
    def to_tensors(
        data: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Convert numpy arrays to PyTorch tensors."""
        data_tensor = torch.FloatTensor(data)
        if labels is not None:
            return data_tensor, torch.LongTensor(labels)
        return (data_tensor,)
