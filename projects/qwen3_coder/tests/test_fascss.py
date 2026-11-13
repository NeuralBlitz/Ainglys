"""FASCSS prototype shape/gradient regressions, not scientific validation."""
import pytest
import torch

from qwen3_coder.fascss import FASCSS, SelfConfiguringModule


def test_stacked_modules_with_different_input_hidden_dimensions():
    torch.manual_seed(11)
    model = FASCSS(input_dim=10, hidden_dim=16, output_dim=3, num_modules=3)
    output = model(torch.randn(5, 10))
    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()
    assert model.modules[0].branches[0][0].in_features == 10
    assert model.modules[1].branches[0][0].in_features == 16
    assert model.modules[2].branches[0][0].in_features == 16


@pytest.mark.parametrize("num_modules", [1, 2, 5])
def test_forward_with_variable_stack_depth(num_modules):
    model = FASCSS(input_dim=4, hidden_dim=6, output_dim=2, num_modules=num_modules)
    assert model(torch.zeros(2, 4)).shape == (2, 2)


def test_gradients_reach_first_and_last_branches():
    torch.manual_seed(12)
    model = FASCSS(input_dim=7, hidden_dim=12, output_dim=2, num_modules=3)
    prediction = model(torch.randn(6, 7))
    prediction.square().mean().backward()
    first = model.modules[0].branches[0][0].weight.grad
    last = model.modules[-1].branches[0][0].weight.grad
    assert first is not None and torch.isfinite(first).all()
    assert last is not None and torch.isfinite(last).all()


def test_adapt_accepts_nonmatching_input_and_hidden_widths():
    torch.manual_seed(13)
    model = FASCSS(input_dim=3, hidden_dim=5, output_dim=2, num_modules=3)
    x, target = torch.randn(4, 3), torch.randn(4, 2)
    model.adapt(x, target, steps=1)
    output = model(x)
    assert output.shape == target.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_empty_stack(count):
    with pytest.raises(ValueError, match="num_modules"):
        FASCSS(input_dim=3, hidden_dim=5, output_dim=2, num_modules=count)


def test_single_self_configuring_module_shapes():
    module = SelfConfiguringModule(input_dim=3, hidden_dim=8, output_dim=4)
    assert module(torch.randn(2, 3)).shape == (2, 4)
