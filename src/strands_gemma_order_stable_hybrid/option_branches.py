"""Full-recomputation isolated-option scoring engine.

ORIGIN/CHANGE NOTICE: Adapted from the accepted isolated-option retrofit. The
standalone package imports inference, modeling, prompting, and schema primitives
from the commit-pinned ``strands-decider`` dependency instead of copying them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index
from strands_decider.modeling import (
    PointerHead,
    StrandsDeciderModel,
    apply_temperature,
    gather_options,
    masked_log_softmax,
    pool_last_token,
)
from strands_decider.prompting import (
    RenderedQuestion,
    read_choice,
    read_noul,
    read_score,
    render_question,
    render_state,
    score_legend,
)
from strands_decider.schema import (
    Answer,
    ChoiceAnswer,
    Content,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    derive_confidence,
    derive_score_confidence,
)


@dataclass(frozen=True)
class OptionBranch:
    text: str
    semantic_label: str
    description: str
    kind: str
    option_span: tuple[int, int]


@dataclass(frozen=True)
class OptionBranchScores:
    branches: tuple[OptionBranch, ...]
    logits: torch.Tensor
    probabilities: torch.Tensor
    input_tokens: int


@dataclass(frozen=True)
class OptionBranchFeatures:
    scores: OptionBranchScores
    answer_vectors: torch.Tensor
    query_vectors: torch.Tensor
    option_vectors: torch.Tensor
    key_vectors: torch.Tensor


def _option_text(kind: str, label: str, description: str) -> str:
    description = " ".join(description.split())
    if kind == "score":
        return description
    return label + (f" — {description}" if description else "")


def render_option_branches(
    question: Question,
    *,
    option_order: Sequence[int] | None = None,
) -> tuple[OptionBranch, ...]:
    """Render one sibling-free suffix per semantic option."""
    rendered = render_question(question, option_order=option_order)
    if not rendered.option_spans:
        raise ValueError(f"{rendered.kind} question has no semantic options")
    option_start = rendered.option_spans[0][0]
    prefix = rendered.text[:option_start]
    suffix = rendered.text[rendered.option_spans[-1][1] :]
    branches: list[OptionBranch] = []
    for label, description in zip(rendered.slot_labels, rendered.slot_descriptions, strict=True):
        option = _option_text(rendered.kind, label, description)
        if not option:
            raise ValueError(
                "score option descriptions must be non-empty for label-free branch scoring"
            )
        start = len(prefix)
        branches.append(
            OptionBranch(
                text=prefix + option + suffix,
                semantic_label=label,
                description=description,
                kind=rendered.kind,
                option_span=(start, start + len(option)),
            )
        )
    return tuple(branches)


def _rendered_mapping(branches: tuple[OptionBranch, ...]) -> RenderedQuestion:
    if not branches:
        raise ValueError("cannot map an empty branch set")
    kinds = {branch.kind for branch in branches}
    if len(kinds) != 1:
        raise ValueError("all branches for one question must have the same kind")
    return RenderedQuestion(
        text="",
        slot_labels=tuple(branch.semantic_label for branch in branches),
        slot_descriptions=tuple(branch.description for branch in branches),
        kind=branches[0].kind,
    )


def _to_order_free_answer(
    branches: tuple[OptionBranch, ...],
    probabilities: list[float],
    *,
    ordinal_smoothing: float,
) -> Answer:
    """Apply upstream semantic readers with stable semantic tie handling."""
    rendered = _rendered_mapping(branches)
    if rendered.kind == "noul":
        return NoulAnswer(noul=round(read_noul(probabilities, rendered), 4))
    if rendered.kind == "choice":
        by_name = read_choice(probabilities, rendered)
        highest = max(by_name.values())
        best = min(name for name, probability in by_name.items() if probability == highest)
        return ChoiceAnswer(
            choice=best,
            probabilities={name: round(value, 4) for name, value in by_name.items()},
            confidence=round(derive_confidence(list(by_name.values())), 4),
        )
    if rendered.kind == "score":
        score, ordered = read_score(probabilities, rendered)
        return ScoreAnswer(
            score=round(score, 4),
            legend=score_legend(rendered),
            probabilities={level: round(value, 4) for level, value in ordered.items()},
            confidence=round(
                derive_score_confidence(
                    list(ordered.values()), ordinal_smoothing=ordinal_smoothing
                ),
                4,
            ),
        )
    raise ValueError(f"isolated option semantics are not defined for {rendered.kind!r}")


class OptionBranchEngine(SystemOneEngine):
    """Mechanically order-equivariant, uncached isolated-option engine."""

    def __init__(self, model: StrandsDeciderModel, config: EngineConfig | None = None):
        if model.config.head_type != "pointer":
            raise ValueError("isolated option scoring requires a PointerHead checkpoint")
        super().__init__(model, config)

    def _fit_branches(
        self, state_text: str, branches: tuple[OptionBranch, ...]
    ) -> tuple[list[int], list[list[int]]]:
        texts = [branch.text for branch in branches]
        encoded = self.tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        question_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        option_starts: list[int] = []
        for branch, branch_offsets in zip(branches, offsets, strict=True):
            start, end = branch.option_span
            option_tokens = [
                index
                for index, (low, high) in enumerate(branch_offsets)
                if high > low and low >= start and high <= end
            ]
            if not option_tokens:
                raise ValueError("an isolated option has no complete tokenizer token")
            option_starts.append(option_tokens[0])
        first_start = option_starts[0]
        first_prefix = question_ids[0][:first_start]
        if any(start != first_start for start in option_starts[1:]) or any(
            ids[:start] != first_prefix
            for ids, start in zip(question_ids[1:], option_starts[1:], strict=True)
        ):
            raise ValueError(
                "tokenizer does not preserve an identical logical prefix for isolated options"
            )
        max_len = self.model.config.max_length
        longest = max(len(ids) for ids in question_ids)
        if self.cfg.strict_window:
            state_ids = self.tok(state_text, add_special_tokens=True)["input_ids"]
            if len(state_ids) + longest > max_len:
                raise ValueError(
                    f"prompt of {len(state_ids) + longest} tokens exceeds the context window "
                    f"of {max_len} tokens"
                )
            self._last_offsets = list(offsets)
            return state_ids, question_ids
        reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
        shared_cut = max(0, longest - reserve)
        if shared_cut > first_start:
            raise ValueError(
                "context window cannot retain isolated options at equal logical positions"
            )
        self._last_offsets = [branch_offsets[shared_cut:] for branch_offsets in offsets]
        question_ids = [ids[shared_cut:] for ids in question_ids]
        state_budget = max(1, max_len - reserve)
        state_ids = self.tok(
            state_text,
            add_special_tokens=True,
            truncation=True,
            max_length=state_budget,
        )["input_ids"]
        return state_ids, question_ids

    def _score_question_impl(
        self,
        state: Content,
        question: Question,
        *,
        option_order: Sequence[int] | None,
        capture_features: bool,
    ) -> tuple[OptionBranchScores, OptionBranchFeatures | None]:
        branches = render_option_branches(question, option_order=option_order)
        state_ids, question_ids = self._fit_branches(render_state(state), branches)
        offsets = self._last_offsets
        head = self.model.head
        if capture_features and type(head) is not PointerHead:
            raise TypeError("feature extraction requires the model's exact PointerHead type")
        raw_scalars: list[torch.Tensor] = []
        answers: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        options: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        input_tokens = 0
        for branch, branch_ids, branch_offsets in zip(branches, question_ids, offsets, strict=True):
            input_ids, attention_mask = self._pad([state_ids + branch_ids])
            option_in_suffix = _option_token_index(branch_offsets, [branch.option_span], 0)[0]
            option_idx = torch.tensor(
                [[len(state_ids) + option_in_suffix]],
                dtype=torch.long,
                device=self.device,
            )
            hidden = self.model.encode(input_ids=input_ids, attention_mask=attention_mask)
            decide = pool_last_token(hidden, attention_mask).to(torch.float32)
            option = gather_options(hidden, option_idx).to(torch.float32)
            captured: dict[str, torch.Tensor] = {}
            handles: list[torch.utils.hooks.RemovableHandle] = []
            if capture_features:
                assert type(head) is PointerHead

                def capture_query(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                    *,
                    sink: dict[str, torch.Tensor] = captured,
                ) -> None:
                    sink["query"] = output

                def capture_key(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                    *,
                    sink: dict[str, torch.Tensor] = captured,
                ) -> None:
                    sink["key"] = output

                handles = [
                    head.q.register_forward_hook(capture_query),
                    head.k.register_forward_hook(capture_key),
                ]
            try:
                branch_logits = head(decide, option)
            finally:
                for handle in handles:
                    handle.remove()
            if branch_logits.shape != (1, 1):
                raise ValueError(
                    "PointerHead must emit one scalar for an isolated option; "
                    f"got shape {tuple(branch_logits.shape)}"
                )
            raw_scalars.append(branch_logits[0, 0])
            if capture_features:
                if set(captured) != {"query", "key"}:
                    raise RuntimeError("PointerHead projections were not captured exactly once")
                answers.append(decide[0].clone())
                options.append(option[0, 0].clone())
                queries.append(captured["query"][0].clone())
                keys.append(captured["key"][0, 0].clone())
            input_tokens += int(attention_mask.sum().item())
        logits = torch.stack(raw_scalars)
        calibrated = apply_temperature(logits.unsqueeze(0), self._temperatures([branches[0].kind]))
        probabilities = masked_log_softmax(
            calibrated,
            torch.tensor([len(branches)], dtype=torch.long, device=self.device),
        ).exp()[0]
        scores = OptionBranchScores(branches, logits, probabilities, input_tokens)
        if not capture_features:
            return scores, None
        return scores, OptionBranchFeatures(
            scores=scores,
            answer_vectors=torch.stack(answers),
            query_vectors=torch.stack(queries),
            option_vectors=torch.stack(options),
            key_vectors=torch.stack(keys),
        )

    @torch.inference_mode()
    def score_question(
        self,
        state: Content,
        question: Question,
        *,
        option_order: Sequence[int] | None = None,
    ) -> OptionBranchScores:
        scores, _ = self._score_question_impl(
            state, question, option_order=option_order, capture_features=False
        )
        return scores

    @torch.inference_mode()
    def extract_question_features(
        self,
        state: Content,
        question: Question,
        *,
        option_order: Sequence[int] | None = None,
    ) -> OptionBranchFeatures:
        _, features = self._score_question_impl(
            state, question, option_order=option_order, capture_features=True
        )
        assert features is not None
        return features

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.images:
            raise ValueError(
                "this engine has no vision tower; isolated option images are unsupported"
            )
        answers: dict[str, Answer] = {}
        input_tokens = 0
        for name, question in request.questions.items():
            scores = self.score_question(request.state, question)
            input_tokens += scores.input_tokens
            answers[name] = _to_order_free_answer(
                scores.branches,
                scores.probabilities.tolist(),
                ordinal_smoothing=self.model.config.ordinal_smoothing,
            )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=len(answers)),
        )


@torch.inference_mode()
def load_option_branch_engine(
    checkpoint: str,
    *,
    device: str = "cuda",
    attn_implementation: str | None = None,
) -> OptionBranchEngine:
    """Load a pointer checkpoint into uncached isolated-option inference."""
    model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    return OptionBranchEngine(model, EngineConfig(device=device, use_prefix_cache=False))
