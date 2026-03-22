# Contributing to AI Hub

Thank you for your interest in contributing to AI Hub! This document provides guidelines and instructions for contributing to the complete 4-phase ML platform with 48 features.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ai-hub.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/my-feature`

## Development Setup

```bash
# Install all packages
pip install -e ".[dev]"
cd projects/acca && pip install -e ".[dev]"
cd ../Advanced-Research/Ideas/projects/acml && pip install -e ".[dev]"
# ... repeat for other packages

# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/test_phase1_comprehensive.py -v
pytest tests/test_phase2_comprehensive.py -v
pytest tests/test_phase3_comprehensive.py -v

# Run linting
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Code Style

- Follow PEP 8
- Use Black for formatting (line length 88)
- Use type hints for all functions
- Write docstrings for all public functions
- Include comprehensive tests

## Testing Requirements

- All code must have tests
- Aim for >80% coverage
- Include unit tests and integration tests
- Run `pytest` before committing

### Test Organization

```
tests/
├── test_api.py                    # Core API tests
├── test_ml_engine.py              # ML engine tests
├── test_phase1_comprehensive.py   # Phase 1: Core Infrastructure
├── test_phase2_comprehensive.py   # Phase 2: Advanced Infrastructure
└── test_phase3_comprehensive.py   # Phase 3: Visualization & Analysis
```

## Pull Request Process

1. **Update documentation** - Ensure all docs are current
2. **Add tests** - All new features need tests
3. **Ensure all tests pass** - Run the full test suite
4. **Update CHANGELOG.md** - Document your changes
5. **Submit PR** - With clear description and examples

## Research Package Guidelines

When adding new research packages:

1. **Follow existing structure** - Use template generator if possible
2. **Include comprehensive tests** - Minimum 80% coverage
3. **Add docstrings and type hints** - For all public APIs
4. **Create example notebook** - If applicable
5. **Update README** - With package info and usage examples
6. **Add to COMPLETE_IMPLEMENTATION_SUMMARY.md** - Update stats

### Package Structure Template

```
my_package/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── core.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── notebooks/
│   └── example.ipynb
├── README.md
└── pyproject.toml
```

## Phase Development Guidelines

### Phase 1: Core Infrastructure
- Focus on stability and security
- All endpoints must be tested
- Follow OAuth2/JWT patterns from security.py
- Use model_registry.py as reference for versioning

### Phase 2: Advanced Infrastructure
- Include async support where applicable
- WebSocket handlers need proper connection management
- Ray integration should handle worker failures gracefully
- AutoML strategies need convergence tests

### Phase 3: Visualization & Analysis
- Plotly figures should be exportable to HTML
- Interpretability features need both image and tabular support
- Jupyter widgets must handle missing dependencies gracefully
- Dataset versioning follows git-like patterns

### Phase 4: Production
- Docker services should be stateless where possible
- Include health checks for all services
- Monitoring endpoints must follow Prometheus format
- Document all environment variables

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on constructive feedback
- Help maintain code quality

## Documentation Standards

### Docstring Format

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Detailed description explaining what the function does,
    when to use it, and any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
        
    Example:
        >>> result = my_function(42, "test")
        >>> print(result)
        True
    """
    pass
```

### README Requirements

Every module/package should have:
- Overview section
- Installation instructions
- Usage examples
- API documentation
- Test instructions

## Testing Best Practices

### Unit Tests
```python
def test_feature():
    """Test specific feature functionality."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = feature_under_test(input_data)
    
    # Assert
    assert result.expected_property == expected_value
```

### Integration Tests
```python
def test_end_to_end():
    """Test complete workflow."""
    # Setup
    client = TestClient(app)
    
    # Execute workflow
    response1 = client.post("/endpoint1", json=data1)
    response2 = client.post("/endpoint2", json=response1.json())
    
    # Verify
    assert response2.status_code == 200
```

## Questions?

- 📖 Check [COMPLETE_IMPLEMENTATION_SUMMARY.md](COMPLETE_IMPLEMENTATION_SUMMARY.md) for full platform details
- 🐛 Open an issue for bugs or feature requests
- 💬 Join our research community discussions

## Platform Statistics (Current)

- **21 Core Modules**
- **87 Research Packages**
- **170+ Tests**
- **48 Features Implemented**
- **95%+ Test Coverage**

Let's keep improving the platform together!

---

*Built with ❤️ by the Advanced Research Team*
