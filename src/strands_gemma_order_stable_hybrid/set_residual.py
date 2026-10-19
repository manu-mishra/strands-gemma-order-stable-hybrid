"""Position-free tied-set residual.

ORIGIN/CHANGE NOTICE: Adapted from the completed order-equivariant experiment.
Host-specific helpers were removed, configuration is strict, and artifact I/O is
limited to the portable files consumed by this standalone package.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

TRAINABLE_NAMES = frozenset({"P", "Q", "W", "b", "out"})
FORMAT = "e2b-content-only-tied-set-residual-v1"
CONFIG_NAME = "set_residual_config.json"
WEIGHTS_NAME = "set_residual.safetensors"


class SetResidualError(RuntimeError):
    """Raised when residual inputs or artifacts violate the contract."""


@dataclass(frozen=True)
class SetResidualConfig:
    hidden_size: int
    rank: int
    mlp_hidden: int
    format: str = FORMAT
    position_inputs: bool = False
    pooling: str = "masked-mean"
    attention: str = "content-dot-product"
    output: str = "shared-zero-initialized-residual"

    def validate(self) -> None:
        if self.format != FORMAT:
            raise SetResidualError("unknown set-residual format")
        if min(self.hidden_size, self.rank, self.mlp_hidden) < 1:
            raise SetResidualError("set-residual dimensions must be positive")
        if self.position_inputs or self.pooling != "masked-mean":
            raise SetResidualError("set residual must be position-free with mean pooling")


class TiedSetResidual(nn.Module):
    """Content self-attention/DeepSets residual with option-tied transforms."""

    def __init__(self, config: SetResidualConfig, *, seed: int = 0):
        super().__init__()
        config.validate()
        self.config = config
        generator = torch.Generator(device="cpu").manual_seed(seed)
        width, rank, hidden = config.hidden_size, config.rank, config.mlp_hidden
        self.P = nn.Parameter(torch.randn(rank, width, generator=generator) / math.sqrt(width))
        self.Q = nn.Parameter(torch.randn(rank, width, generator=generator) / math.sqrt(width))
        self.W = nn.Parameter(
            torch.randn(hidden, 5 * rank, generator=generator) / math.sqrt(5 * rank)
        )
        self.b = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Parameter(torch.zeros(hidden))
        self.assert_parameter_allowlist()

    def assert_parameter_allowlist(self) -> None:
        names = set(dict(self.named_parameters()))
        if names != TRAINABLE_NAMES:
            raise SetResidualError(
                f"unexpected trainable tensors: {sorted(names ^ TRAINABLE_NAMES)}"
            )
        if any(not parameter.requires_grad for parameter in self.parameters()):
            raise SetResidualError("all and only relational tensors must be trainable")

    def _validate_inputs(
        self,
        base_logits: torch.Tensor,
        question: torch.Tensor,
        options: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if base_logits.ndim != 2 or question.ndim != 2 or options.ndim != 3:
            raise SetResidualError("expected base [B,K], question [B,D], options [B,K,D]")
        if mask.dtype != torch.bool or mask.shape != base_logits.shape:
            raise SetResidualError("mask must be bool with the same [B,K] shape as logits")
        if options.shape[:2] != base_logits.shape or question.shape[0] != base_logits.shape[0]:
            raise SetResidualError("relational batch shapes disagree")
        if (
            options.shape[2] != self.config.hidden_size
            or question.shape[1] != self.config.hidden_size
        ):
            raise SetResidualError("feature width disagrees with residual config")
        if torch.any(mask.sum(dim=1) < 2):
            raise SetResidualError("every item must contain at least two unmasked options")
        if not torch.isfinite(base_logits[mask]).all() or not torch.isfinite(options[mask]).all():
            raise SetResidualError("active logits/options must be finite")
        if not torch.isfinite(question).all():
            raise SetResidualError("question vectors must be finite")

    def delta(
        self, question: torch.Tensor, options: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute only the shared relational correction."""
        dummy = options.new_zeros(options.shape[:2])
        self._validate_inputs(dummy, question, options, mask)
        safe_options = torch.where(mask[:, :, None], options, torch.zeros_like(options))
        z = torch.tanh(safe_options @ self.P.T)
        u = torch.tanh(question @ self.Q.T)
        attention_logits = z @ z.transpose(1, 2) / math.sqrt(self.config.rank)
        attention_logits = attention_logits.masked_fill(~mask[:, None, :], -torch.inf)
        attention = torch.softmax(attention_logits.float(), dim=-1).to(z.dtype)
        context = attention @ z
        active = mask[:, :, None].to(z.dtype)
        mean = (z * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        expanded_mean = mean[:, None, :]
        joined = torch.cat(
            [z, context, z - expanded_mean, z * expanded_mean, u[:, None, :].expand_as(z)],
            dim=-1,
        )
        return torch.tanh(joined @ self.W.T + self.b) @ self.out

    def forward(
        self,
        base_logits: torch.Tensor,
        question: torch.Tensor,
        options: torch.Tensor,
        mask: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return base_logits
        self._validate_inputs(base_logits, question, options, mask)
        return (base_logits + self.delta(question, options, mask)).masked_fill(~mask, -torch.inf)


def deterministic_argmax(logits: torch.Tensor, semantic_labels: Sequence[str]) -> int:
    """Resolve exact ties by lexicographically smallest semantic identity."""
    if logits.ndim != 1 or len(logits) != len(semantic_labels) or not semantic_labels:
        raise SetResidualError("logits and semantic labels must be one non-empty row")
    if len(set(semantic_labels)) != len(semantic_labels):
        raise SetResidualError("semantic labels must be unique")
    maximum = torch.max(logits)
    tied = [index for index, value in enumerate(logits) if bool(value == maximum)]
    return min(tied, key=lambda index: semantic_labels[index])


def tensor_shapes(config: SetResidualConfig) -> dict[str, tuple[int, ...]]:
    return {
        "P": (config.rank, config.hidden_size),
        "Q": (config.rank, config.hidden_size),
        "W": (config.mlp_hidden, 5 * config.rank),
        "b": (config.mlp_hidden,),
        "out": (config.mlp_hidden,),
    }


def save_residual(directory: Path, model: TiedSetResidual) -> None:
    """Write the two-file residual artifact, refusing replacement."""
    if directory.exists():
        raise SetResidualError(f"residual output already exists: {directory}")
    directory.mkdir(parents=True)
    model.assert_parameter_allowlist()
    state = {
        name: value.detach().cpu().float().contiguous() for name, value in model.named_parameters()
    }
    save_file(state, directory / WEIGHTS_NAME, metadata={"format": FORMAT})
    (directory / CONFIG_NAME).write_text(
        json.dumps(asdict(model.config), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_residual(directory: Path, *, device: str | torch.device = "cpu") -> TiedSetResidual:
    """Load and structurally validate a residual artifact."""
    try:
        raw = json.loads((directory / CONFIG_NAME).read_text(encoding="utf-8"))
        config = SetResidualConfig(**raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SetResidualError("invalid set-residual config") from exc
    config.validate()
    try:
        tensors = load_file(directory / WEIGHTS_NAME, device=str(device))
    except (OSError, RuntimeError) as exc:
        raise SetResidualError("invalid set-residual weights") from exc
    expected = tensor_shapes(config)
    if set(tensors) != TRAINABLE_NAMES:
        raise SetResidualError("residual artifact contains unexpected tensors")
    for name, shape in expected.items():
        if tensors[name].dtype != torch.float32 or tuple(tensors[name].shape) != shape:
            raise SetResidualError(f"invalid residual tensor {name}")
    model = TiedSetResidual(config).to(device)
    model.load_state_dict(tensors, strict=True)
    model.assert_parameter_allowlist()
    return model
