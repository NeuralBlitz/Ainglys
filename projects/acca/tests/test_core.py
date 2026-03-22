"""
Tests for the core module.
"""

import pytest
import torch
from acca import CoreModule

class TestCoreModule:
    def test_initialization(self):
        module = CoreModule(10, 20, 5)
        assert module is not None
    
    def test_forward(self):
        module = CoreModule(10, 20, 5)
        x = torch.randn(2, 10)
        out = module(x)
        assert out.shape == (2, 5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
