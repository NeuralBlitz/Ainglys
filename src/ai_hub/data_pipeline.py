"""
Data Pipeline Management

Comprehensive data pipeline tools for ML workflows:
- ETL pipeline construction
- Data validation and quality checks
- Data augmentation and preprocessing
- Streaming data support
- Pipeline versioning and lineage
- Multi-source data integration

Example:
    >>> from ai_hub.data_pipeline import DataPipeline, PipelineStage
    >>> pipeline = DataPipeline()
    >>> pipeline.add_stage(PipelineStage.extract("s3://bucket/data"))
    >>> pipeline.add_stage(PipelineStage.transform(normalize))
    >>> pipeline.add_stage(PipelineStage.load("processed_data"))
    >>> result = pipeline.execute()
"""

from typing import Dict, List, Optional, Callable, Any, Iterator, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import time
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class DataQualityReport:
    """Data quality assessment report."""

    total_records: int
    valid_records: int
    missing_values: Dict[str, int] = field(default_factory=dict)
    outliers: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    schema_violations: List[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        if self.total_records == 0:
            return 0.0

        score = self.valid_records / self.total_records

        # Penalize for issues
        total_missing = sum(self.missing_values.values())
        score -= (
            total_missing / (self.total_records * len(self.missing_values) + 1)
        ) * 0.1
        score -= (self.duplicates / self.total_records) * 0.1

        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "quality_score": self.quality_score,
            "missing_values": self.missing_values,
            "outliers": self.outliers,
            "duplicates": self.duplicates,
            "schema_violations": self.schema_violations,
        }


class DataValidator:
    """Validate data quality and schema compliance."""

    def __init__(self, schema: Optional[Dict[str, str]] = None):
        """
        Initialize validator.

        Args:
            schema: Expected schema {column: type}
        """
        self.schema = schema or {}

    def validate_dataframe(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate a pandas DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            DataQualityReport with validation results
        """
        total_records = len(df)
        missing_values = {}
        outliers = {}
        schema_violations = []

        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)

        # Check outliers (numeric columns only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(
                df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            )
            if outlier_count > 0:
                outliers[col] = int(outlier_count)

        # Check duplicates
        duplicates = df.duplicated().sum()

        # Check schema
        if self.schema:
            for col, expected_type in self.schema.items():
                if col not in df.columns:
                    schema_violations.append(f"Missing column: {col}")
                elif str(df[col].dtype) != expected_type:
                    schema_violations.append(
                        f"Type mismatch in {col}: expected {expected_type}, got {df[col].dtype}"
                    )

        # Calculate valid records
        valid_records = total_records - duplicates - sum(missing_values.values())

        return DataQualityReport(
            total_records=total_records,
            valid_records=max(0, valid_records),
            missing_values=missing_values,
            outliers=outliers,
            duplicates=int(duplicates),
            schema_violations=schema_violations,
        )

    def validate_batch(self, data: List[Dict]) -> DataQualityReport:
        """Validate a batch of records."""
        df = pd.DataFrame(data)
        return self.validate_dataframe(df)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str):
        self.name = name
        self.execution_time_ms = 0.0
        self.records_processed = 0

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute the stage on input data."""
        pass

    def run(self, data: Any) -> Any:
        """Run stage with timing."""
        start = time.time()
        result = self.execute(data)
        self.execution_time_ms = (time.time() - start) * 1000

        if hasattr(result, "__len__"):
            self.records_processed = len(result)

        return result

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "execution_time_ms": self.execution_time_ms,
            "records_processed": self.records_processed,
        }


class ExtractStage(PipelineStage):
    """Data extraction stage."""

    def __init__(self, source: Union[str, Path, Callable], source_type: str = "auto"):
        super().__init__(f"extract_{source_type}")
        self.source = source
        self.source_type = source_type

    def execute(self, data: Any = None) -> Any:
        """Extract data from source."""
        if callable(self.source):
            return self.source()

        source_str = str(self.source)

        if source_str.endswith(".csv") or self.source_type == "csv":
            return pd.read_csv(source_str)
        elif source_str.endswith(".json") or self.source_type == "json":
            with open(source_str) as f:
                return json.load(f)
        elif source_str.endswith(".parquet") or self.source_type == "parquet":
            return pd.read_parquet(source_str)
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")


class TransformStage(PipelineStage):
    """Data transformation stage."""

    def __init__(self, transform_fn: Callable, name: str = "transform"):
        super().__init__(name)
        self.transform_fn = transform_fn

    def execute(self, data: Any) -> Any:
        """Apply transformation function."""
        return self.transform_fn(data)


class LoadStage(PipelineStage):
    """Data loading stage."""

    def __init__(self, destination: Union[str, Path], dest_type: str = "auto"):
        super().__init__(f"load_{dest_type}")
        self.destination = destination
        self.dest_type = dest_type

    def execute(self, data: Any) -> Any:
        """Load data to destination."""
        dest_str = str(self.destination)

        if isinstance(data, pd.DataFrame):
            if dest_str.endswith(".csv") or self.dest_type == "csv":
                data.to_csv(dest_str, index=False)
            elif dest_str.endswith(".parquet") or self.dest_type == "parquet":
                data.to_parquet(dest_str)
            elif dest_str.endswith(".json") or self.dest_type == "json":
                data.to_json(dest_str, orient="records")
            else:
                raise ValueError(f"Unknown destination type: {self.dest_type}")
        elif isinstance(data, (dict, list)):
            with open(dest_str, "w") as f:
                json.dump(data, f, indent=2)

        return data


class DataPipeline:
    """
    Complete data pipeline for ML workflows.

    Supports ETL operations with validation, versioning, and monitoring.

    Example:
        >>> pipeline = DataPipeline(name="preprocessing")
        >>>
        >>> # Define stages
        >>> pipeline.add_stage(ExtractStage("data.csv", "csv"))
        >>> pipeline.add_stage(TransformStage(lambda df: df.dropna()))
        >>> pipeline.add_stage(LoadStage("processed.csv", "csv"))
        >>>
        >>> # Execute
        >>> result = pipeline.execute()
        >>> report = pipeline.get_quality_report()
    """

    def __init__(self, name: str = "pipeline", version: str = "1.0.0"):
        """
        Initialize pipeline.

        Args:
            name: Pipeline name
            version: Pipeline version
        """
        self.name = name
        self.version = version
        self.stages: List[PipelineStage] = []
        self.validator = DataValidator()
        self.quality_reports: List[DataQualityReport] = []
        self.execution_log: List[Dict] = []
        self._data_hash: Optional[str] = None

    def add_stage(self, stage: PipelineStage) -> "DataPipeline":
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self

    def validate(self, data: Any) -> DataQualityReport:
        """Validate data at current stage."""
        if isinstance(data, pd.DataFrame):
            report = self.validator.validate_dataframe(data)
        elif isinstance(data, list):
            report = self.validator.validate_batch(data)
        else:
            report = DataQualityReport(
                total_records=1 if data else 0, valid_records=1 if data else 0
            )

        self.quality_reports.append(report)
        return report

    def execute(
        self, initial_data: Any = None, validate_each_stage: bool = True
    ) -> Any:
        """
        Execute the complete pipeline.

        Args:
            initial_data: Initial data (if first stage doesn't extract)
            validate_each_stage: Whether to validate after each stage

        Returns:
            Final processed data
        """
        print(f"Executing pipeline: {self.name} v{self.version}")
        print(f"Stages: {len(self.stages)}")

        data = initial_data
        start_time = time.time()

        for i, stage in enumerate(self.stages):
            print(f"\nStage {i + 1}/{len(self.stages)}: {stage.name}")

            # Execute stage
            data = stage.run(data)

            print(f"  Records: {stage.records_processed}")
            print(f"  Time: {stage.execution_time_ms:.2f} ms")

            # Validate if requested
            if validate_each_stage:
                report = self.validate(data)
                print(f"  Quality Score: {report.quality_score:.2%}")

            # Log execution
            self.execution_log.append(
                {
                    "stage": stage.name,
                    "index": i,
                    "timestamp": time.time(),
                    **stage.to_dict(),
                }
            )

        total_time = (time.time() - start_time) * 1000
        print(f"\nPipeline completed in {total_time:.2f} ms")

        return data

    def execute_parallel(
        self,
        data_batches: List[Any],
        num_workers: int = None,
        use_processes: bool = False,
    ) -> List[Any]:
        """
        Execute pipeline on multiple batches in parallel.

        Args:
            data_batches: List of data batches to process
            num_workers: Number of parallel workers
            use_processes: Use processes instead of threads

        Returns:
            List of processed batches
        """
        num_workers = num_workers or mp.cpu_count()

        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        def process_batch(batch):
            # Create a copy of pipeline for this batch
            batch_pipeline = DataPipeline(f"{self.name}_batch")
            batch_pipeline.stages = self.stages
            return batch_pipeline.execute(batch)

        print(
            f"Executing pipeline on {len(data_batches)} batches with {num_workers} workers"
        )

        with Executor(max_workers=num_workers) as executor:
            results = list(executor.map(process_batch, data_batches))

        return results

    def get_quality_report(self) -> DataQualityReport:
        """Get aggregated quality report."""
        if not self.quality_reports:
            return DataQualityReport(total_records=0, valid_records=0)

        # Aggregate reports
        total_records = sum(r.total_records for r in self.quality_reports)
        valid_records = sum(r.valid_records for r in self.quality_reports)

        # Combine issues
        all_missing = {}
        all_outliers = {}
        total_duplicates = 0

        for report in self.quality_reports:
            for col, count in report.missing_values.items():
                all_missing[col] = all_missing.get(col, 0) + count
            for col, count in report.outliers.items():
                all_outliers[col] = all_outliers.get(col, 0) + count
            total_duplicates += report.duplicates

        return DataQualityReport(
            total_records=total_records,
            valid_records=valid_records,
            missing_values=all_missing,
            outliers=all_outliers,
            duplicates=total_duplicates,
        )

    def save_checkpoint(self, filepath: str):
        """Save pipeline state to checkpoint."""
        checkpoint = {
            "name": self.name,
            "version": self.version,
            "stages": [s.to_dict() for s in self.stages],
            "execution_log": self.execution_log,
            "quality_report": self.get_quality_report().to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

    @classmethod
    def from_checkpoint(cls, filepath: str) -> "DataPipeline":
        """Load pipeline from checkpoint."""
        with open(filepath) as f:
            checkpoint = json.load(f)

        pipeline = cls(checkpoint["name"], checkpoint["version"])
        pipeline.execution_log = checkpoint.get("execution_log", [])
        return pipeline

    def get_lineage(self) -> Dict:
        """Get data lineage information."""
        return {
            "pipeline_name": self.name,
            "version": self.version,
            "num_stages": len(self.stages),
            "stages": [s.name for s in self.stages],
            "execution_log": self.execution_log,
            "quality_score": self.get_quality_report().quality_score,
        }


def example_usage():
    """Example usage of data pipeline."""
    # Create sample data
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        }
    )

    # Save sample
    data.to_csv("sample_data.csv", index=False)

    # Create pipeline
    pipeline = DataPipeline("preprocessing", "1.0.0")

    # Add stages
    pipeline.add_stage(ExtractStage("sample_data.csv", "csv"))

    pipeline.add_stage(
        TransformStage(
            lambda df: df.assign(
                feature1_normalized=df["feature1"] / df["feature1"].max()
            ),
            "normalize",
        )
    )

    pipeline.add_stage(TransformStage(lambda df: df.dropna(), "clean"))

    pipeline.add_stage(LoadStage("processed_data.csv", "csv"))

    # Execute
    result = pipeline.execute()

    # Get quality report
    report = pipeline.get_quality_report()
    print(f"\nFinal Quality Score: {report.quality_score:.2%}")

    # Save checkpoint
    pipeline.save_checkpoint("pipeline_checkpoint.json")

    # Show lineage
    lineage = pipeline.get_lineage()
    print(f"\nPipeline Lineage:")
    print(f"  Stages: {lineage['stages']}")
    print(f"  Quality: {lineage['quality_score']:.2%}")


if __name__ == "__main__":
    example_usage()
