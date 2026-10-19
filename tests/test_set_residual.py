from __future__ import annotations

import pytest
import torch

from strands_gemma_order_stable_hybrid.set_residual import (
    SetResidualConfig,
    SetResidualError,
    TiedSetResidual,
    deterministic_argmax,
)


def _model() -> TiedSetResidual:
    model = TiedSetResidual(SetResidualConfig(hidden_size=4, rank=3, mlp_hidden=5), seed=7)
    with torch.no_grad():
        model.out.copy_(torch.linspace(-0.2, 0.2, 5))
    return model


def test_disabled_mode_returns_exact_input_object() -> None:
    model = _model()
    logits = torch.tensor([[1.0, 2.0]])
    result = model(
        logits,
        torch.full((1, 4), float("nan")),
        torch.empty(1, 2, 9),
        torch.zeros(1, 2, dtype=torch.long),
        enabled=False,
    )
    assert result is logits


def test_permutation_equivariance_and_masked_invariance() -> None:
    torch.manual_seed(3)
    model = _model()
    logits = torch.randn(1, 4)
    question = torch.randn(1, 4)
    options = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, True, True, False]])
    base = model(logits, question, options, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    moved = model(logits[:, permutation], question, options[:, permutation], mask[:, permutation])
    assert torch.allclose(moved, base[:, permutation], equal_nan=True)
    changed = options.clone()
    changed[:, 3] = float("nan")
    assert torch.allclose(model(logits, question, changed, mask)[:, :3], base[:, :3])


@pytest.mark.parametrize("count", [2, 3, 7])
def test_shapes_and_tie_behavior(count: int) -> None:
    model = _model()
    output = model(
        torch.zeros(2, count),
        torch.ones(2, 4),
        torch.ones(2, count, 4),
        torch.ones(2, count, dtype=torch.bool),
    )
    assert output.shape == (2, count)
    labels = ["zeta", "alpha", *[f"x{i}" for i in range(count - 2)]]
    assert deterministic_argmax(torch.zeros(count), labels) == labels.index(min(labels))


def test_invalid_feature_width_and_too_few_options_fail() -> None:
    model = _model()
    with pytest.raises(SetResidualError, match="feature width"):
        model(
            torch.zeros(1, 2),
            torch.zeros(1, 5),
            torch.zeros(1, 2, 5),
            torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(SetResidualError, match="at least two"):
        model(
            torch.zeros(1, 2),
            torch.zeros(1, 4),
            torch.zeros(1, 2, 4),
            torch.tensor([[True, False]]),
        )
