from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class FrameworkRegistry:
    """Registry for different ML frameworks."""

    _frameworks: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, framework_class: type):
        cls._frameworks[name] = framework_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._frameworks.get(name)

    @classmethod
    def list_frameworks(cls) -> list:
        return list(cls._frameworks.keys())


class BaseWrapper(nn.Module):
    """Base wrapper for all frameworks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AccaWrapper(BaseWrapper):
    """Wrapper for acca framework."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AdagWrapper(BaseWrapper):
    """Wrapper for adag framework."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AdaptWrapper(BaseWrapper):
    """Wrapper for adapt framework."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GabbWrapper(BaseWrapper):
    """Wrapper for gabb framework."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GranmlWrapper(BaseWrapper):
    """Wrapper for granml framework."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


FrameworkRegistry.register("acca", AccaWrapper)
FrameworkRegistry.register("adag", AdagWrapper)
FrameworkRegistry.register("adapt", AdaptWrapper)
FrameworkRegistry.register("gabb", GabbWrapper)
FrameworkRegistry.register("granml", GranmlWrapper)


def get_framework(
    name: str, input_dim: int, hidden_dim: int, output_dim: int
) -> BaseWrapper:
    """Get a framework wrapper by name."""
    framework_class = FrameworkRegistry.get(name)
    if framework_class is None:
        raise ValueError(
            f"Unknown framework: {name}. Available: {FrameworkRegistry.list_frameworks()}"
        )
    return framework_class(input_dim, hidden_dim, output_dim)
