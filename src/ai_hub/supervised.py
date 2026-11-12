"""Validated, stratified supervised-learning requests for the Ainglys API.

This module defines bounded in-memory datasets and an auditable holdout split.
It deliberately does not assert any scientific validity for the supplied data.
"""
from __future__ import annotations

from collections import Counter
import math
import random
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class SupervisedTrainRequest(BaseModel):
    """Small, explicitly labeled numeric classification dataset.

    Labels must be consecutive integer class IDs starting with zero, and each
    class needs at least two samples to support a stratified held-out set.
    """

    model_config = ConfigDict(extra="forbid")

    features: list[list[float]]
    labels: list[StrictInt]
    hidden_dim: Annotated[int, Field(ge=1, le=256)] = 32
    epochs: Annotated[int, Field(ge=1, le=20)] = 5
    learning_rate: Annotated[float, Field(ge=1e-6, le=0.1, allow_inf_nan=False)] = 0.001
    validation_fraction: Annotated[float, Field(ge=0.1, le=0.4, allow_inf_nan=False)] = 0.2
    seed: Annotated[int, Field(ge=0, le=2_147_483_647)] = 42

    @model_validator(mode="after")
    def validate_dataset(self):
        count = len(self.features)
        if not 16 <= count <= 512:
            raise ValueError("features must contain 16-512 samples")
        feature_dim = len(self.features[0])
        if not 1 <= feature_dim <= 128:
            raise ValueError("features must contain 1-128 columns")
        if any(len(row) != feature_dim for row in self.features):
            raise ValueError("all feature rows must have equal dimensions")
        if any(
            not math.isfinite(value) or abs(value) > 1_000_000
            for row in self.features for value in row
        ):
            raise ValueError("all features must be finite and within +/-1e6")
        if len(self.labels) != count:
            raise ValueError("labels must match the number of feature rows")
        label_counts = Counter(self.labels)
        if not 2 <= len(label_counts) <= 32:
            raise ValueError("classification requires 2-32 classes")
        if sorted(label_counts) != list(range(len(label_counts))):
            raise ValueError("labels must use contiguous class IDs from 0")
        if min(label_counts.values()) < 2:
            raise ValueError("each class must have at least two samples")
        return self


def stratified_holdout_indices(
    labels: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return disjoint train and validation indices, with every class in both.

    This operates on *indices*, never on the feature values, preventing data
    normalization/feature selection leakage in this initial API.
    """
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)

    if len(by_class) < 2 or any(len(indices) < 2 for indices in by_class.values()):
        raise ValueError("stratified holdout needs >=2 samples for each of >=2 classes")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be a non-negative 32-bit signed integer")

    rng = random.Random(seed)
    training: list[int] = []
    validation: list[int] = []
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        heldout = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        validation.extend(indices[:heldout])
        training.extend(indices[heldout:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation
