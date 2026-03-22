"""
Package Template Generator CLI

Quickly scaffold new research packages with proper structure.

Usage:
    ai-hub create-package --name "new_project" --type "meta-learning"
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional


# Template for pyproject.toml
PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
version = "0.1.0"
description = "{description}"
readme = "README.md"
requires-python = ">=3.10"
license = {{text = "MIT"}}
dependencies = [
    "numpy>=1.24.0",
    "torch>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "mypy>=0.950",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
"""

# Template for README.md
README_TEMPLATE = """# {package_name}

{description}

## Overview

This package implements {package_type} for AI/ML research.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from {package_name} import CoreModule

# Create model
model = CoreModule(input_dim=128, hidden_dim=256, output_dim=10)

# Forward pass
import torch
x = torch.randn(32, 128)
output = model(x)
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Tests

```bash
pytest tests/ -v
```

## Mathematical Foundation

Describe the mathematical framework here:

$$
L_{{expected}} = L_{{standard}} + \\alpha L_{{novel}}
$$

## Citation

```bibtex
@software{{{package_name},
  title = {{{package_name}}: {description}}},
  author = {{Author Name}},
  year = {{{year}}},
}}
```
"""

# Template for __init__.py
INIT_TEMPLATE = '''"""
{package_name} - {description}
"""

__version__ = "0.1.0"

from .core import CoreModule
from .model import {class_name}

__all__ = ["CoreModule", "{class_name}"]
'''

# Template for core.py
CORE_TEMPLATE = '''"""
Core module for {package_name}.
"""

import torch
import torch.nn as nn


class CoreModule(nn.Module):
    """
    Core neural network module.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = torch.relu(self.fc1(x))
        return self.fc2(h)
'''

# Template for model.py
MODEL_TEMPLATE = '''"""
Main {package_type} model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class {class_name}(nn.Module):
    """
    {description}
    
    Implementation of {package_type} architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {{
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }}
'''

# Template for tests
TEST_TEMPLATE = '''"""
Tests for {package_name}.
"""

import pytest
import torch
from {package_name} import CoreModule, {class_name}


class TestCoreModule:
    """Test core module."""
    
    def test_initialization(self):
        """Test model initialization."""
        module = CoreModule(10, 20, 5)
        assert module is not None
    
    def test_forward(self):
        """Test forward pass."""
        module = CoreModule(10, 20, 5)
        x = torch.randn(2, 10)
        out = module(x)
        assert out.shape == (2, 5)
    
    def test_backward(self):
        """Test backward pass."""
        module = CoreModule(10, 20, 5)
        x = torch.randn(2, 10)
        target = torch.randn(2, 5)
        
        out = module(x)
        loss = torch.nn.MSELoss()(out, target)
        loss.backward()
        
        assert module.fc1.weight.grad is not None


class Test{class_name}:
    """Test main model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = {class_name}(128, 256, 10)
        assert model is not None
    
    def test_forward_shape(self):
        """Test output shape."""
        model = {class_name}(128, 256, 10)
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4, 10)
    
    def test_get_config(self):
        """Test config retrieval."""
        model = {class_name}(128, 256, 10)
        config = model.get_config()
        assert config["input_dim"] == 128
        assert config["output_dim"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

# Template for .gitignore
GITIGNORE_TEMPLATE = """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
*.log
.DS_Store
"""


def create_package(
    name: str,
    package_type: str = "general",
    description: Optional[str] = None,
    output_dir: str = "Advanced-Research/Ideas/projects",
):
    """
    Create a new research package with full structure.

    Args:
        name: Package name (e.g., "new_project")
        package_type: Type of research (e.g., "meta-learning", "optimization")
        description: Package description
        output_dir: Where to create the package
    """
    # Normalize name
    package_name = name.lower().replace("-", "_").replace(" ", "_")
    class_name = "".join(word.capitalize() for word in package_name.split("_"))

    if description is None:
        description = f"AI/ML Research Framework - {package_name}"

    # Create directories
    base_path = Path(output_dir) / package_name
    src_path = base_path / "src" / package_name
    tests_path = base_path / "tests"

    for path in [src_path, tests_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Template variables
    template_vars = {
        "package_name": package_name,
        "class_name": class_name,
        "description": description,
        "package_type": package_type,
        "year": datetime.now().year,
    }

    # Create files
    files_to_create = [
        (base_path / "pyproject.toml", PYPROJECT_TEMPLATE),
        (base_path / "README.md", README_TEMPLATE),
        (base_path / ".gitignore", GITIGNORE_TEMPLATE),
        (src_path / "__init__.py", INIT_TEMPLATE),
        (src_path / "core.py", CORE_TEMPLATE),
        (src_path / "model.py", MODEL_TEMPLATE),
        (tests_path / "test_core.py", TEST_TEMPLATE),
    ]

    for filepath, template in files_to_create:
        content = template.format(**template_vars)
        with open(filepath, "w") as f:
            f.write(content)

    # Create empty __init__.py for tests
    (tests_path / "__init__.py").touch()

    print(f"✅ Created package '{package_name}' at {base_path}")
    print(f"\nNext steps:")
    print(f"  cd {base_path}")
    print(f"  pip install -e .")
    print(f"  pytest tests/ -v")

    return base_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate AI/ML research package template"
    )
    parser.add_argument(
        "--name", "-n", required=True, help="Package name (e.g., 'new_project')"
    )
    parser.add_argument(
        "--type",
        "-t",
        default="general",
        choices=[
            "general",
            "meta-learning",
            "optimization",
            "architecture",
            "field-theory",
        ],
        help="Type of research package",
    )
    parser.add_argument("--description", "-d", help="Package description")
    parser.add_argument(
        "--output",
        "-o",
        default="Advanced-Research/Ideas/projects",
        help="Output directory",
    )

    args = parser.parse_args()

    create_package(
        name=args.name,
        package_type=args.type,
        description=args.description,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
