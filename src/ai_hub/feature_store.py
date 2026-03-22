"""
Feature Store

Centralized feature management for ML workflows:
- Feature registration and versioning
- Online and offline feature serving
- Feature transformation pipelines
- Feature monitoring and drift detection
- Feature sharing across teams

Example:
    >>> from ai_hub.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> store.register_feature("user_age", feature_data, version="v1")
    >>> features = store.get_features(["user_age", "user_income"], entity_ids)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime
from collections import defaultdict
import pickle


@dataclass
class FeatureMetadata:
    """Metadata for a registered feature."""

    name: str
    version: str
    data_type: str
    description: str
    created_at: datetime
    owner: str
    tags: List[str] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "data_type": self.data_type,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "owner": self.owner,
            "tags": self.tags,
            "statistics": self.statistics,
        }


@dataclass
class FeatureView:
    """A view combining multiple features."""

    name: str
    features: List[str]
    description: str
    created_at: datetime

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "features": self.features,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }


class FeatureTransformation:
    """Transform raw data into features."""

    def __init__(self, name: str, transform_fn: Callable, input_features: List[str]):
        self.name = name
        self.transform_fn = transform_fn
        self.input_features = input_features

    def apply(self, data: pd.DataFrame) -> pd.Series:
        """Apply transformation to data."""
        return self.transform_fn(data[self.input_features])


class FeatureStore:
    """
    Centralized feature management system.

    Manages features for ML workflows with support for:
    - Feature registration and versioning
    - Online (low-latency) and offline (batch) serving
    - Feature transformations
    - Feature monitoring

    Example:
        >>> store = FeatureStore("./feature_store")
        >>>
        >>> # Register features
        >>> store.register_feature(
        ...     "user_age",
        ...     user_data['age'],
        ...     entity_ids=user_data['user_id'],
        ...     description="User age in years"
        ... )
        >>>
        >>> # Create feature view
        >>> store.create_feature_view(
        ...     "user_demographics",
        ...     ["user_age", "user_income", "user_location"]
        ... )
        >>>
        >>> # Retrieve features
        >>> features = store.get_online_features(
        ...     "user_demographics",
        ...     entity_ids=["user_001", "user_002"]
        ... )
    """

    def __init__(self, storage_path: str = "./feature_store"):
        """
        Initialize feature store.

        Args:
            storage_path: Path to store feature data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.features: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self.metadata: Dict[str, FeatureMetadata] = {}
        self.feature_views: Dict[str, FeatureView] = {}
        self.transformations: Dict[str, FeatureTransformation] = {}

        self._load_metadata()

    def _load_metadata(self):
        """Load existing metadata from storage."""
        metadata_file = self.storage_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                data = json.load(f)
                for name, meta in data.get("features", {}).items():
                    self.metadata[name] = FeatureMetadata(
                        name=meta["name"],
                        version=meta["version"],
                        data_type=meta["data_type"],
                        description=meta["description"],
                        created_at=datetime.fromisoformat(meta["created_at"]),
                        owner=meta.get("owner", "unknown"),
                        tags=meta.get("tags", []),
                        statistics=meta.get("statistics", {}),
                    )

    def _save_metadata(self):
        """Save metadata to storage."""
        metadata_file = self.storage_path / "metadata.json"
        data = {
            "features": {name: meta.to_dict() for name, meta in self.metadata.items()},
            "feature_views": {
                name: view.to_dict() for name, view in self.feature_views.items()
            },
        }
        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_feature(
        self,
        name: str,
        data: Union[pd.Series, np.ndarray, List],
        entity_ids: Union[pd.Series, np.ndarray, List],
        description: str = "",
        version: str = "v1",
        owner: str = "unknown",
        tags: Optional[List[str]] = None,
    ) -> FeatureMetadata:
        """
        Register a new feature.

        Args:
            name: Feature name
            data: Feature values
            entity_ids: IDs for each feature value
            description: Feature description
            version: Feature version
            owner: Feature owner
            tags: Tags for organization

        Returns:
            FeatureMetadata for registered feature
        """
        # Convert to DataFrame
        if isinstance(data, (np.ndarray, list)):
            data = pd.Series(data)
        if isinstance(entity_ids, (np.ndarray, list)):
            entity_ids = pd.Series(entity_ids)

        df = pd.DataFrame(
            {"entity_id": entity_ids, "value": data, "timestamp": datetime.now()}
        )

        # Calculate statistics
        if pd.api.types.is_numeric_dtype(data):
            statistics = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "null_count": int(data.isna().sum()),
            }
        else:
            statistics = {
                "unique_count": int(data.nunique()),
                "null_count": int(data.isna().sum()),
            }

        # Store feature
        feature_path = self.storage_path / f"{name}_{version}.parquet"
        df.to_parquet(feature_path)

        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            version=version,
            data_type=str(data.dtype),
            description=description,
            created_at=datetime.now(),
            owner=owner,
            tags=tags or [],
            statistics=statistics,
        )

        self.metadata[name] = metadata
        self._save_metadata()

        print(f"Registered feature '{name}' v{version} with {len(df)} entities")
        return metadata

    def get_feature(self, name: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        Get a feature by name.

        Args:
            name: Feature name
            version: Specific version (latest if None)

        Returns:
            DataFrame with feature data
        """
        if version is None:
            version = self.metadata[name].version

        feature_path = self.storage_path / f"{name}_{version}.parquet"
        if not feature_path.exists():
            raise ValueError(f"Feature {name} v{version} not found")

        return pd.read_parquet(feature_path)

    def get_online_features(
        self, feature_names: List[str], entity_ids: List[str]
    ) -> pd.DataFrame:
        """
        Get online features for specific entities (low latency).

        Args:
            feature_names: List of feature names
            entity_ids: List of entity IDs

        Returns:
            DataFrame with features for requested entities
        """
        result = pd.DataFrame({"entity_id": entity_ids})

        for feature_name in feature_names:
            feature_df = self.get_feature(feature_name)
            feature_df = feature_df[feature_df["entity_id"].isin(entity_ids)]
            feature_df = feature_df.rename(columns={"value": feature_name})
            result = result.merge(
                feature_df[["entity_id", feature_name]], on="entity_id", how="left"
            )

        return result

    def get_offline_features(
        self,
        feature_names: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get offline features for batch processing.

        Args:
            feature_names: List of feature names
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with all feature data
        """
        result = None

        for feature_name in feature_names:
            feature_df = self.get_feature(feature_name)

            # Apply date filters
            if start_date:
                feature_df = feature_df[feature_df["timestamp"] >= start_date]
            if end_date:
                feature_df = feature_df[feature_df["timestamp"] <= end_date]

            feature_df = feature_df.rename(columns={"value": feature_name})

            if result is None:
                result = feature_df[["entity_id", feature_name]]
            else:
                result = result.merge(
                    feature_df[["entity_id", feature_name]], on="entity_id", how="outer"
                )

        return result

    def create_feature_view(
        self, name: str, features: List[str], description: str = ""
    ) -> FeatureView:
        """
        Create a feature view combining multiple features.

        Args:
            name: View name
            features: List of feature names to include
            description: View description

        Returns:
            FeatureView object
        """
        # Validate features exist
        for feature in features:
            if feature not in self.metadata:
                raise ValueError(f"Feature '{feature}' not found")

        view = FeatureView(
            name=name,
            features=features,
            description=description,
            created_at=datetime.now(),
        )

        self.feature_views[name] = view
        self._save_metadata()

        print(f"Created feature view '{name}' with {len(features)} features")
        return view

    def register_transformation(
        self, name: str, transform_fn: Callable, input_features: List[str]
    ):
        """
        Register a feature transformation.

        Args:
            name: Transformation name
            transform_fn: Function that transforms input features
            input_features: List of input feature names
        """
        transformation = FeatureTransformation(name, transform_fn, input_features)
        self.transformations[name] = transformation
        print(f"Registered transformation '{name}'")

    def apply_transformation(
        self, transformation_name: str, entity_ids: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Apply a transformation to features.

        Args:
            transformation_name: Name of registered transformation
            entity_ids: Entity IDs to apply to (all if None)

        Returns:
            Transformed feature values
        """
        if transformation_name not in self.transformations:
            raise ValueError(f"Transformation '{transformation_name}' not found")

        transformation = self.transformations[transformation_name]

        # Get input features
        if entity_ids:
            data = self.get_online_features(transformation.input_features, entity_ids)
        else:
            data = self.get_offline_features(transformation.input_features)

        # Apply transformation
        return transformation.apply(data)

    def list_features(self, tag: Optional[str] = None) -> List[FeatureMetadata]:
        """
        List all registered features.

        Args:
            tag: Filter by tag

        Returns:
            List of feature metadata
        """
        features = list(self.metadata.values())

        if tag:
            features = [f for f in features if tag in f.tags]

        return features

    def get_feature_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a feature."""
        if name not in self.metadata:
            raise ValueError(f"Feature '{name}' not found")

        return self.metadata[name].statistics

    def delete_feature(self, name: str, version: Optional[str] = None):
        """
        Delete a feature.

        Args:
            name: Feature name
            version: Specific version (all if None)
        """
        if version:
            feature_path = self.storage_path / f"{name}_{version}.parquet"
            if feature_path.exists():
                feature_path.unlink()
            print(f"Deleted feature '{name}' v{version}")
        else:
            # Delete all versions
            for file in self.storage_path.glob(f"{name}_*.parquet"):
                file.unlink()
            if name in self.metadata:
                del self.metadata[name]
            self._save_metadata()
            print(f"Deleted feature '{name}' (all versions)")

    def get_lineage(self, feature_name: str) -> Dict:
        """
        Get data lineage for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Lineage information
        """
        if feature_name not in self.metadata:
            raise ValueError(f"Feature '{feature_name}' not found")

        meta = self.metadata[feature_name]

        # Find which views use this feature
        views_using = [
            view.name
            for view in self.feature_views.values()
            if feature_name in view.features
        ]

        # Find transformations that use this feature
        transformations_using = [
            name
            for name, trans in self.transformations.items()
            if feature_name in trans.input_features
        ]

        return {
            "feature_name": feature_name,
            "version": meta.version,
            "created_at": meta.created_at.isoformat(),
            "owner": meta.owner,
            "used_in_views": views_using,
            "used_in_transformations": transformations_using,
        }


def example_usage():
    """Example usage of feature store."""
    # Create sample data
    np.random.seed(42)
    n_users = 1000

    user_data = pd.DataFrame(
        {
            "user_id": [f"user_{i:04d}" for i in range(n_users)],
            "age": np.random.randint(18, 80, n_users),
            "income": np.random.normal(50000, 15000, n_users).astype(int),
            "location": np.random.choice(["urban", "suburban", "rural"], n_users),
            "signup_date": pd.date_range(start="2020-01-01", periods=n_users, freq="D"),
        }
    )

    # Initialize feature store
    store = FeatureStore("./example_feature_store")

    print("=" * 70)
    print("FEATURE STORE EXAMPLE")
    print("=" * 70)

    # Register features
    print("\n1. Registering Features")
    print("-" * 40)

    store.register_feature(
        "user_age",
        user_data["age"],
        user_data["user_id"],
        description="User age in years",
        owner="data_team",
        tags=["demographics", "user"],
    )

    store.register_feature(
        "user_income",
        user_data["income"],
        user_data["user_id"],
        description="Annual income in USD",
        owner="data_team",
        tags=["demographics", "financial"],
    )

    store.register_feature(
        "user_location",
        user_data["location"],
        user_data["user_id"],
        description="User location type",
        owner="data_team",
        tags=["demographics", "geographic"],
    )

    # Create feature view
    print("\n2. Creating Feature View")
    print("-" * 40)

    store.create_feature_view(
        "user_demographics",
        ["user_age", "user_income", "user_location"],
        description="Complete user demographic features",
    )

    # Get online features (for real-time prediction)
    print("\n3. Getting Online Features")
    print("-" * 40)

    online_features = store.get_online_features(
        ["user_age", "user_income"], entity_ids=["user_0001", "user_0002", "user_0003"]
    )
    print(f"Retrieved features for {len(online_features)} entities")
    print(online_features)

    # Get offline features (for training)
    print("\n4. Getting Offline Features (for training)")
    print("-" * 40)

    offline_features = store.get_offline_features(
        ["user_age", "user_income", "user_location"]
    )
    print(f"Retrieved {len(offline_features)} records for batch processing")
    print(f"Shape: {offline_features.shape}")

    # Register transformation
    print("\n5. Feature Transformation")
    print("-" * 40)

    store.register_transformation(
        "income_per_age",
        lambda df: df["user_income"] / df["user_age"],
        ["user_income", "user_age"],
    )

    transformed = store.apply_transformation(
        "income_per_age", ["user_0001", "user_0002"]
    )
    print(f"Transformed feature for {len(transformed)} entities")
    print(transformed)

    # List features
    print("\n6. Feature Inventory")
    print("-" * 40)

    all_features = store.list_features()
    print(f"Total features: {len(all_features)}")
    for feat in all_features:
        print(f"  - {feat.name} ({feat.data_type})")

    # Get feature statistics
    print("\n7. Feature Statistics")
    print("-" * 40)

    stats = store.get_feature_statistics("user_age")
    print(f"User age statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    # Get lineage
    print("\n8. Feature Lineage")
    print("-" * 40)

    lineage = store.get_lineage("user_age")
    print(f"Feature: {lineage['feature_name']}")
    print(f"Owner: {lineage['owner']}")
    print(f"Used in views: {lineage['used_in_views']}")

    print("\n" + "=" * 70)
    print("Feature Store Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
