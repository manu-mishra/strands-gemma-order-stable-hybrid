from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from strands_decider.infer import EngineConfig
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from torch import nn

from strands_gemma_order_stable_hybrid.option_branches import (
    OptionBranchEngine,
    render_option_branches,
)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        truncation=False,
        max_length=None,
        **_kwargs,
    ):
        single = isinstance(texts, str)
        rows = [texts] if single else list(texts)
        encoded, offsets = [], []
        for text in rows:
            ids = [ord(character) for character in text]
            spans = [(index, index + 1) for index in range(len(text))]
            if add_special_tokens:
                ids, spans = [1, *ids], [(0, 0), *spans]
            if truncation and max_length is not None:
                ids, spans = ids[:max_length], spans[:max_length]
            encoded.append(ids)
            offsets.append(spans)
        result = {"input_ids": encoded[0] if single else encoded}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets[0] if single else offsets
        return result


class LastOptionHead(nn.Module):
    def forward(self, decide, options):
        return options[..., 0]


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            head_type="pointer",
            max_length=4096,
            temperature=1.0,
            temperature_by_kind={"noul": 2.0, "score": 0.5},
            ordinal_smoothing=0.0,
        )
        self.tokenizer = CharacterTokenizer()
        self.torso = nn.Identity()
        self.head = LastOptionHead()
        self.calls = 0

    def encode(self, input_ids, attention_mask, past_key_values=None):
        assert past_key_values is None
        self.calls += 1
        return input_ids.float().unsqueeze(-1)


def _engine() -> tuple[OptionBranchEngine, StubModel]:
    model = StubModel()
    return OptionBranchEngine(model, EngineConfig(device="cpu", model_name="stub")), model


def test_rendering_is_sibling_free_and_permutation_preserves_semantics() -> None:
    question = ChoiceQuestion(
        instructions="Choose.",
        criteria={"cerulean": "ocean", "vermilion": "sunset", "chartreuse": "leaf"},
    )
    branches = render_option_branches(question)
    assert len({branch.option_span[0] for branch in branches}) == 1
    for branch in branches:
        for sibling in {"cerulean", "vermilion", "chartreuse"} - {branch.semantic_label}:
            assert sibling not in branch.text
    permuted = render_option_branches(question, option_order=[2, 0, 1])
    assert [branch.semantic_label for branch in permuted] == [
        "chartreuse",
        "cerulean",
        "vermilion",
    ]


def test_choice_recomputes_each_branch_and_ties_by_semantic_name() -> None:
    engine, model = _engine()
    question = ChoiceQuestion(instructions="Pick.", criteria={"zeta": "same", "alpha": "same"})
    response = engine.evaluate(SystemOneRequest(state="state", questions={"pick": question}))
    assert response.answers["pick"].choice == "alpha"
    assert set(response.answers["pick"].probabilities) == {"zeta", "alpha"}
    assert model.calls == 2


def test_noul_and_score_semantic_readback() -> None:
    engine, _ = _engine()
    response = engine.evaluate(
        SystemOneRequest(
            state="state",
            questions={
                "truth": NoulQuestion(
                    instructions="True?", criteria={"false": "ends-f", "true": "ends-z"}
                ),
                "rating": ScoreQuestion(
                    instructions="Rate.", criteria=["ends-a", "ends-b", "ends-c"]
                ),
            },
        )
    )
    expected_truth = torch.softmax(torch.tensor([ord("f"), ord("z")]) / 2, dim=0)[1]
    assert response.answers["truth"].noul == pytest.approx(round(expected_truth.item(), 4))
    assert response.answers["rating"].legend == {
        "0": "ends-a",
        "1": "ends-b",
        "2": "ends-c",
    }


def test_empty_score_text_fails() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        render_option_branches(ScoreQuestion(instructions="Rate.", criteria=["low", ""]))
