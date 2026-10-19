"""Hybrid normal and compare-choices runtime composition.

ORIGIN/CHANGE NOTICE: Adapted from the completed experiment runtime. Recursive
publication validators, checkout imports, absolute paths, and import-path mutation
were replaced by installed imports and the portable manifest API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from strands_decider import schema

from .artifacts import DEFAULT_MODE, MANIFEST_NAME, MODES, ValidatedArtifact, validate_manifest
from .option_branches import OptionBranchScores, _to_order_free_answer, load_option_branch_engine
from .set_residual import TiedSetResidual, load_residual


class HybridRuntimeError(ValueError):
    """Raised when a mode or hybrid component is incompatible."""


def resolve_mode(mode: str | None) -> str:
    selected = DEFAULT_MODE if mode is None else mode
    if selected not in MODES:
        raise HybridRuntimeError(f"unsupported hybrid mode: {selected!r}")
    return selected


class HybridOptionBranchEngine:
    """Compose an unchanged normal engine with one frozen tied-set residual."""

    def __init__(self, normal_engine: Any, residual: TiedSetResidual):
        self.normal_engine = normal_engine
        self.residual = residual
        self.model = normal_engine.model

    @torch.inference_mode()
    def score_question(
        self,
        state: Any,
        question: Any,
        *,
        option_order: Sequence[int] | None = None,
        mode: str | None = None,
    ) -> OptionBranchScores:
        selected = resolve_mode(mode)
        if selected == "normal":
            return self.normal_engine.score_question(state, question, option_order=option_order)

        capture = self.normal_engine.extract_question_features(
            state, question, option_order=option_order
        )
        base_logits = capture.scores.logits
        answer_vectors = capture.answer_vectors
        option_vectors = capture.option_vectors
        width = len(capture.scores.branches)
        if (
            base_logits.ndim != 1
            or answer_vectors.ndim != 2
            or option_vectors.ndim != 2
            or len(base_logits) != width
            or len(answer_vectors) != width
            or len(option_vectors) != width
        ):
            raise HybridRuntimeError("captured normal feature shapes disagree")
        question_vector = answer_vectors.mean(dim=0)
        mask = torch.ones((1, width), dtype=torch.bool, device=base_logits.device)
        logits = self.residual(
            base_logits.unsqueeze(0),
            question_vector.unsqueeze(0),
            option_vectors.unsqueeze(0),
            mask,
            enabled=True,
        )[0]
        probabilities = torch.softmax(logits.float(), dim=0).to(logits.dtype)
        return OptionBranchScores(
            branches=capture.scores.branches,
            logits=logits,
            probabilities=probabilities,
            input_tokens=capture.scores.input_tokens,
        )

    @torch.inference_mode()
    def evaluate(self, request: Any, *, mode: str | None = None) -> Any:
        selected = resolve_mode(mode)
        if request.images:
            raise ValueError(
                "this engine has no vision tower; isolated option images are unsupported"
            )
        answers: dict[str, Any] = {}
        input_tokens = 0
        for name, question in request.questions.items():
            scores = self.score_question(request.state, question, mode=selected)
            input_tokens += int(scores.input_tokens)
            answers[name] = _to_order_free_answer(
                scores.branches,
                scores.probabilities.tolist(),
                ordinal_smoothing=self.model.config.ordinal_smoothing,
            )
        return schema.SystemOneResponse(
            model=self.normal_engine.cfg.model_name,
            answers=answers,
            usage=schema.Usage(input_tokens=input_tokens, output_tokens=len(answers)),
        )


@dataclass(frozen=True)
class HybridBundle:
    engine: HybridOptionBranchEngine
    artifact: ValidatedArtifact
    engine_kind: str = "HybridOptionBranchEngine"


NormalLoader = Callable[..., Any]
ResidualLoader = Callable[..., TiedSetResidual]


def load_hybrid(
    artifact: str | Path,
    *,
    device: str = "cuda",
    attention_implementation: str | None = "eager",
    normal_loader: NormalLoader = load_option_branch_engine,
    residual_loader: ResidualLoader = load_residual,
) -> HybridBundle:
    """Validate a relocated artifact and load both components exactly once."""
    path = Path(artifact)
    manifest_path = path / MANIFEST_NAME if path.is_dir() else path
    validated = validate_manifest(manifest_path)
    normal = normal_loader(
        str(validated.normal.path),
        device=device,
        attn_implementation=attention_implementation,
    )
    residual = residual_loader(validated.relational.path, device=device).eval()
    residual.requires_grad_(False)
    normal_width = getattr(normal.model.config, "hidden_size", None)
    if normal_width is not None and residual.config.hidden_size != int(normal_width):
        raise HybridRuntimeError("normal and relational hidden dimensions differ")
    return HybridBundle(HybridOptionBranchEngine(normal, residual), validated)
