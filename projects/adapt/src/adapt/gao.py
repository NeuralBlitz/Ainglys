"""
Granular Arithmetic Operators (GAO)

Implements uncertainty-aware, context-sensitive numeric representations
that generalize floating-point semantics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Set, Optional, Any


@dataclass
class TemporalInterval:
    """Temporal validity interval."""

    start: float
    end: float

    def intersection(self, other: TemporalInterval) -> TemporalInterval:
        return TemporalInterval(max(self.start, other.start), min(self.end, other.end))


@dataclass
class GranularNumber:
    """
    Granular Number representation.

    x^gamma = (v, sigma, c, t, tau)

    Where:
    - v: Nominal value
    - sigma: Uncertainty radius
    - c: Context tag set
    - t: Semantic type
    - tau: Temporal validity interval
    """

    value: float
    uncertainty: float
    context: Set[str]
    semantic_type: str
    temporal: TemporalInterval

    def __add__(self, other: GranularNumber) -> GranularNumber:
        return GranularArithmetic.add(self, other)

    def __mul__(self, other: GranularNumber) -> GranularNumber:
        return GranularArithmetic.multiply(self, other)

    def __sub__(self, other: GranularNumber) -> GranularNumber:
        return GranularArithmetic.subtract(self, other)

    def __truediv__(self, other: GranularNumber) -> GranularNumber:
        return GranularArithmetic.divide(self, other)

    def contains(self, x: float) -> bool:
        """Check if value x is within uncertainty bounds."""
        return abs(x - self.value) <= self.uncertainty

    def uncertainty_ratio(self) -> float:
        """Return relative uncertainty."""
        if self.value == 0:
            return float("inf") if self.uncertainty > 0 else 0.0
        return self.uncertainty / abs(self.value)

    def __repr__(self) -> str:
        return f"Granular({self.value} ± {self.uncertainty}, type={self.semantic_type})"


class GranularArithmetic:
    """
    Granular Arithmetic Operations.

    Implements uncertainty-aware arithmetic with context tracking.
    """

    @staticmethod
    def add(x: GranularNumber, y: GranularNumber) -> GranularNumber:
        """
        Addition: x ⊕ y

        v = v_x + v_y
        sigma = sqrt(sigma_x^2 + sigma_y^2)
        c = c_x ∪ c_y
        t = lcm(t_x, t_y)
        tau = intersection(tau_x, tau_y)
        """
        new_value = x.value + y.value
        new_uncertainty = math.sqrt(x.uncertainty**2 + y.uncertainty**2)
        new_context = x.context | y.context
        new_temporal = x.temporal.intersection(y.temporal)

        return GranularNumber(
            value=new_value,
            uncertainty=new_uncertainty,
            context=new_context,
            semantic_type=x.semantic_type,
            temporal=new_temporal,
        )

    @staticmethod
    def multiply(x: GranularNumber, y: GranularNumber) -> GranularNumber:
        """
        Multiplication: x ⊗ y

        v = v_x * v_y
        sigma = |v_x|*sigma_y + |v_y|*sigma_x + sigma_x*sigma_y
        c = c_x ∪ c_y
        t = t_x × t_y
        tau = intersection(tau_x, tau_y)
        """
        new_value = x.value * y.value
        new_uncertainty = (
            abs(x.value) * y.uncertainty
            + abs(y.value) * x.uncertainty
            + x.uncertainty * y.uncertainty
        )
        new_context = x.context | y.context
        new_temporal = x.temporal.intersection(y.temporal)

        return GranularNumber(
            value=new_value,
            uncertainty=new_uncertainty,
            context=new_context,
            semantic_type=f"{x.semantic_type}*{y.semantic_type}",
            temporal=new_temporal,
        )

    @staticmethod
    def subtract(x: GranularNumber, y: GranularNumber) -> GranularNumber:
        """Subtraction: x ⊖ y"""
        neg_y = GranularNumber(
            value=-y.value,
            uncertainty=y.uncertainty,
            context=y.context,
            semantic_type=y.semantic_type,
            temporal=y.temporal,
        )
        return GranularArithmetic.add(x, neg_y)

    @staticmethod
    def divide(x: GranularNumber, y: GranularNumber) -> GranularNumber:
        """Division: x ⊘ y"""
        if y.value == 0:
            raise ValueError("Division by zero")

        new_value = x.value / y.value
        new_uncertainty = (
            abs(x.value) * y.uncertainty / (y.value**2)
            + x.uncertainty / abs(y.value)
            + x.uncertainty * y.uncertainty / (y.value**2)
        )

        return GranularNumber(
            value=new_value,
            uncertainty=new_uncertainty,
            context=x.context | y.context,
            semantic_type=f"{x.semantic_type}/{y.semantic_type}",
            temporal=x.temporal.intersection(y.temporal),
        )


class GranularTensor:
    """
    Tensor of granular numbers for neural network operations.
    """

    def __init__(self, data, semantic_type: str = "tensor"):
        self.data = data
        self.semantic_type = semantic_type

    def __repr__(self):
        return f"GranularTensor(shape={self.data.shape}, type={self.semantic_type})"
