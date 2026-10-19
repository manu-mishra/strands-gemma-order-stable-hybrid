from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
import torch
from safetensors.torch import save_file
from strands_decider.schema import ChoiceQuestion

import strands_gemma_order_stable_hybrid.artifacts as artifacts
from strands_gemma_order_stable_hybrid.artifacts import (
    CLEAN_RESIDUAL_CONFIG,
    MANIFEST_NAME,
    ArtifactError,
    prepare_artifact,
    validate_manifest,
)
from strands_gemma_order_stable_hybrid.hybrid import HybridOptionBranchEngine, load_hybrid
from strands_gemma_order_stable_hybrid.option_branches import (
    OptionBranch,
    OptionBranchFeatures,
    OptionBranchScores,
)
from strands_gemma_order_stable_hybrid.set_residual import SetResidualConfig, TiedSetResidual


class SpyResidual(TiedSetResidual):
    def __init__(self):
        super().__init__(SetResidualConfig(hidden_size=2, rank=2, mlp_hidden=2))
        self.calls = 0
        self.question = None

    def forward(self, base_logits, question, options, mask, *, enabled=True):
        self.calls += 1
        self.question = question.clone()
        return base_logits + options.sum(dim=-1)


class FakeNormal:
    def __init__(self, hidden_size: int = 2):
        self.model = SimpleNamespace(
            config=SimpleNamespace(ordinal_smoothing=0.0, hidden_size=hidden_size)
        )
        self.cfg = SimpleNamespace(model_name="fake")
        self.sentinel = object()
        self.extract_calls = 0
        branches = (
            OptionBranch("", "left", "L", "choice", (0, 0)),
            OptionBranch("", "right", "R", "choice", (0, 0)),
        )
        scores = OptionBranchScores(branches, torch.tensor([1.0, 2.0]), torch.tensor([0.2, 0.8]), 7)
        self.capture = OptionBranchFeatures(
            scores,
            answer_vectors=torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
            query_vectors=torch.zeros(2, 2),
            option_vectors=torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
            key_vectors=torch.zeros(2, 2),
        )

    def score_question(self, state, question, *, option_order=None):
        return self.sentinel

    def extract_question_features(self, state, question, *, option_order=None):
        self.extract_calls += 1
        return self.capture


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _raw_relational_config(normal_lock_hash: str) -> dict[str, object]:
    return {
        "architecture": {
            "format": CLEAN_RESIDUAL_CONFIG["format"],
            "hidden_size": 1536,
            "mlp_hidden": 64,
            "output": "shared residual",
            "pooling": "masked symmetric mean",
            "positions": False,
            "rank": 128,
        },
        "selection": {"selected_candidate_id": artifacts.SELECTED_CANDIDATE_IDS["compare_choices"]},
        "normal_candidate": {
            "lock_sha256": normal_lock_hash,
            "head_sha256_before": artifacts.NORMAL_SOURCE_FILE_HASHES["head.safetensors"],
            "head_sha256_after": artifacts.NORMAL_SOURCE_FILE_HASHES["head.safetensors"],
        },
    }


def _synthetic_sources(tmp_path: Path, monkeypatch, *, valid_weights: bool = False):
    normal = tmp_path / "normal-source"
    relational = tmp_path / "relational-source"
    normal.mkdir()
    relational.mkdir()

    normal_hashes: dict[str, str] = {}
    for member in artifacts.NORMAL_SOURCE_FILE_HASHES:
        _write(normal / member, f"accepted:{member}\n".encode())
        normal_hashes[member] = _sha256(normal / member)
    monkeypatch.setattr(artifacts, "NORMAL_SOURCE_FILE_HASHES", normal_hashes)

    normal_lock = {
        "candidate": {
            "selected_candidate_id": artifacts.SELECTED_CANDIDATE_IDS["normal"],
            "files": normal_hashes,
        },
        "provenance": {
            "source_commit": artifacts.UPSTREAM_IDENTITIES["strands_source"]["commit"],
            "model_revision": artifacts.UPSTREAM_IDENTITIES["checkpoint"]["revision"],
            "base_model": artifacts.UPSTREAM_IDENTITIES["gemma"]["repo"],
            "base_revision": artifacts.UPSTREAM_IDENTITIES["gemma"]["revision"],
        },
    }
    (normal / artifacts.NORMAL_LOCK_NAME).write_text(json.dumps(normal_lock), encoding="utf-8")
    normal_lock_hash = _sha256(normal / artifacts.NORMAL_LOCK_NAME)

    raw_config = _raw_relational_config(normal_lock_hash)
    (relational / artifacts.RESIDUAL_CONFIG_NAME).write_text(
        json.dumps(raw_config), encoding="utf-8"
    )
    if valid_weights:
        config = SetResidualConfig(hidden_size=1536, rank=128, mlp_hidden=64)
        model = TiedSetResidual(config)
        save_file(
            {name: value.detach().float() for name, value in model.named_parameters()},
            relational / artifacts.RESIDUAL_WEIGHTS_NAME,
        )
    else:
        (relational / artifacts.RESIDUAL_WEIGHTS_NAME).write_bytes(b"accepted residual tensor")
    relational_hashes = {
        artifacts.RESIDUAL_CONFIG_NAME: _sha256(relational / artifacts.RESIDUAL_CONFIG_NAME),
        artifacts.RESIDUAL_WEIGHTS_NAME: _sha256(relational / artifacts.RESIDUAL_WEIGHTS_NAME),
    }
    monkeypatch.setattr(artifacts, "RELATIONAL_SOURCE_FILE_HASHES", relational_hashes)

    relational_lock = {
        "candidate": {
            "selected_candidate_id": artifacts.SELECTED_CANDIDATE_IDS["compare_choices"],
            "format": CLEAN_RESIDUAL_CONFIG["format"],
            "files": relational_hashes,
        },
        "normal_candidate": {
            "lock_sha256": normal_lock_hash,
            "head_sha256": normal_hashes["head.safetensors"],
        },
    }
    (relational / artifacts.RELATIONAL_LOCK_NAME).write_text(
        json.dumps(relational_lock), encoding="utf-8"
    )
    lock_hashes = {
        **artifacts.HISTORICAL_LOCK_HASHES,
        "normal": normal_lock_hash,
        "relational": _sha256(relational / artifacts.RELATIONAL_LOCK_NAME),
    }
    monkeypatch.setattr(artifacts, "HISTORICAL_LOCK_HASHES", lock_hashes)
    return normal, relational


def test_frozen_production_constants_are_exact() -> None:
    assert artifacts.UPSTREAM_IDENTITIES == {
        "strands_source": {"commit": "0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534"},
        "checkpoint": {
            "repo": "StrandsAgents/strands-decider-E2B-gemma4-v1-2610",
            "revision": "d383ecf8534b13d49b3b567a1b66f88fca6d50ba",
        },
        "gemma": {
            "repo": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        },
    }
    assert artifacts.SELECTED_CANDIDATE_IDS == {
        "normal": "wd-0.1-epochs-25-seed-43",
        "compare_choices": "rank-128-hidden-64-seed-17",
    }
    assert artifacts.HISTORICAL_LOCK_HASHES == {
        "normal": "a1bee0186af38614372ba2f6e2f058e8ebc94d5b72e0f2d18e87ac36d405cf5a",
        "relational": "a1738049034a10890c74b6777ecfd015835fd74c98af71dc1e4c6479f9145b49",
        "hybrid": "47748ebb2aca6df06dac46c4a91d075726cf96ef3d4d3c3319a6225bd8550926",
    }
    assert artifacts.NORMAL_SOURCE_FILE_HASHES["head.safetensors"] == (
        "781f0f460975316cac306ae8bb7fa11dd1471d3e8b183140ac8c0aaf1a02e915"
    )
    assert artifacts.NORMAL_SOURCE_FILE_HASHES["lora/adapter_model.safetensors"] == (
        "c7786a59354aa190082b1c3aedb7ff3311fe260f070f360c4d3518762691c551"
    )
    assert artifacts.RELATIONAL_SOURCE_FILE_HASHES["set_residual.safetensors"] == (
        "afe53a76352e389b8aba2d92599742e20770e9a374baafb401c7a283ce22fca3"
    )
    assert artifacts.CLEAN_RESIDUAL_CONFIG_SHA256 == (
        "b3d0973c5ed2480d6ddcd8d50b954f65653986e017223ea06a02de3c024b8a2b"
    )
    assert artifacts.NORMAL_SOURCE_FILE_HASHES["provenance.json"] == (
        "214928b49e4ebb43833c345882662ef5c60917c9fafd7e99a1144ee3d9c5582b"
    )


def test_canonical_portable_manifest_digest_is_exact_and_not_a_historical_lock() -> None:
    canonical = (
        json.dumps(artifacts._expected_manifest(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    independently_computed = hashlib.sha256(canonical).hexdigest()
    assert independently_computed == (
        "faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974"
    )
    assert artifacts.PORTABLE_MANIFEST_SHA256 == independently_computed
    assert artifacts._expected_manifest_sha256() == independently_computed
    assert independently_computed not in artifacts.HISTORICAL_LOCK_HASHES.values()


def test_production_dependency_pins_are_exact() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    for pin in (
        '"setuptools==84.0.0"',
        '"strands-decider @ git+https://github.com/strands-labs/strands-decider.git@0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534"',
        '"torch==2.14.1"',
        '"transformers==5.18.0"',
        '"peft==0.21.2"',
        '"accelerate==1.15.0"',
        '"pydantic==2.12.5"',
        '"safetensors==0.8.0"',
    ):
        assert pin in pyproject


def test_direct_normal_delegation_and_compare_composition() -> None:
    normal = FakeNormal()
    residual = SpyResidual()
    engine = HybridOptionBranchEngine(normal, residual)
    assert engine.score_question("state", "question") is normal.sentinel
    assert normal.extract_calls == residual.calls == 0
    compared = engine.score_question("state", "question", mode="compare_choices")
    assert residual.calls == normal.extract_calls == 1
    assert torch.equal(residual.question, torch.tensor([[2.0, 4.0]]))
    assert torch.equal(compared.logits, torch.tensor([3.0, 6.0]))
    assert torch.allclose(compared.probabilities, torch.softmax(compared.logits, dim=0))


def test_prepare_emits_exact_portable_inventory_and_manifest(tmp_path, monkeypatch) -> None:
    normal, relational = _synthetic_sources(tmp_path, monkeypatch)
    manifest = prepare_artifact(normal, relational, tmp_path / "artifact")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == artifacts._expected_manifest()
    assert set(payload) == {
        "schema_version",
        "format",
        "upstream",
        "selected_candidate_ids",
        "historical_lock_sha256",
        "source_components",
        "modes",
        "default_mode",
        "normal",
        "relational",
    }
    assert set(payload["normal"]["files"]) == set(artifacts.NORMAL_OUTPUT_MEMBERS)
    assert set(payload["relational"]["files"]) == {
        artifacts.RESIDUAL_CONFIG_NAME,
        artifacts.RESIDUAL_WEIGHTS_NAME,
    }
    assert not (manifest.parent / "normal" / artifacts.NORMAL_LOCK_NAME).exists()
    assert not (manifest.parent / "relational" / artifacts.RELATIONAL_LOCK_NAME).exists()
    assert not (manifest.parent / "normal" / "train_config.json").exists()
    clean = json.loads(
        (manifest.parent / "relational" / artifacts.RESIDUAL_CONFIG_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert clean == CLEAN_RESIDUAL_CONFIG
    assert "lock_path" not in clean


def test_prepare_rejects_changed_or_extra_source_before_copy(tmp_path, monkeypatch) -> None:
    normal, relational = _synthetic_sources(tmp_path, monkeypatch)
    (normal / "head.safetensors").write_bytes(b"changed")
    output = tmp_path / "artifact"
    with pytest.raises(ArtifactError, match="released identity"):
        prepare_artifact(normal, relational, output)
    assert not output.exists()

    _write(normal / "head.safetensors", b"accepted:head.safetensors\n")
    (relational / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ArtifactError, match="released identity"):
        prepare_artifact(normal, relational, output)
    assert not output.exists()


def test_manifest_relocates_and_rejects_self_declared_hash_drift(tmp_path, monkeypatch) -> None:
    normal, relational = _synthetic_sources(tmp_path, monkeypatch)
    manifest = prepare_artifact(normal, relational, tmp_path / "artifact")
    relocated = tmp_path / "relocated"
    shutil.copytree(manifest.parent, relocated)
    validated = validate_manifest(relocated / MANIFEST_NAME)
    assert validated.normal.path == relocated / "normal"

    changed = validated.normal.path / "head.safetensors"
    changed.write_bytes(b"self-declared replacement")
    payload = json.loads((relocated / MANIFEST_NAME).read_text(encoding="utf-8"))
    payload["normal"]["files"]["head.safetensors"] = _sha256(changed)
    (relocated / MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="released identity"):
        validate_manifest(relocated / MANIFEST_NAME)


def test_manifest_rejects_changed_pins_ids_locks_and_config(tmp_path, monkeypatch) -> None:
    normal, relational = _synthetic_sources(tmp_path, monkeypatch)
    manifest = prepare_artifact(normal, relational, tmp_path / "artifact")
    original = json.loads(manifest.read_text(encoding="utf-8"))
    mutations = (
        lambda value: value["upstream"]["gemma"].update(repo="other/model"),
        lambda value: value["selected_candidate_ids"].update(normal="other"),
        lambda value: value["historical_lock_sha256"].update(normal="0" * 64),
        lambda value: value.update(default_mode="compare_choices"),
        lambda value: value.update(extra=True),
    )
    for mutate in mutations:
        payload = json.loads(json.dumps(original))
        mutate(payload)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="released identity"):
            validate_manifest(manifest)

    manifest.write_text(json.dumps(original), encoding="utf-8")
    config_path = manifest.parent / "relational" / artifacts.RESIDUAL_CONFIG_NAME
    wrong = {**CLEAN_RESIDUAL_CONFIG, "rank": 64}
    config_path.write_text(json.dumps(wrong), encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["relational"]["files"][artifacts.RESIDUAL_CONFIG_NAME] = _sha256(config_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="released identity"):
        validate_manifest(manifest)


def test_manifest_rejects_missing_extra_symlink_and_traversal(tmp_path, monkeypatch) -> None:
    normal, relational = _synthetic_sources(tmp_path, monkeypatch)
    manifest = prepare_artifact(normal, relational, tmp_path / "artifact")
    original = manifest.read_text(encoding="utf-8")

    extra = manifest.parent / "normal" / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ArtifactError, match="inventory"):
        validate_manifest(manifest)
    extra.unlink()

    member = manifest.parent / "normal" / "README.md"
    member.unlink()
    with pytest.raises(ArtifactError, match="inventory"):
        validate_manifest(manifest)
    member.write_bytes(normal.joinpath("README.md").read_bytes())

    member.unlink()
    member.symlink_to(normal / "README.md")
    with pytest.raises(ArtifactError, match="symlink"):
        validate_manifest(manifest)
    member.unlink()
    member.write_bytes(normal.joinpath("README.md").read_bytes())

    payload = json.loads(original)
    payload["normal"]["path"] = "../normal-source"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="released identity"):
        validate_manifest(manifest)


def test_load_hybrid_loads_each_component_once(tmp_path, monkeypatch) -> None:
    normal_source, relational_source = _synthetic_sources(tmp_path, monkeypatch)
    artifact = prepare_artifact(normal_source, relational_source, tmp_path / "artifact")
    calls = {"normal": 0, "relational": 0}
    normal = FakeNormal()
    residual = TiedSetResidual(SetResidualConfig(hidden_size=2, rank=2, mlp_hidden=2))

    def normal_loader(path, **kwargs):
        calls["normal"] += 1
        assert path.endswith("/normal")
        return normal

    def residual_loader(path, **kwargs):
        calls["relational"] += 1
        assert path.name == "relational"
        return residual

    bundle = load_hybrid(
        artifact, device="cpu", normal_loader=normal_loader, residual_loader=residual_loader
    )
    assert bundle.engine.normal_engine is normal
    assert calls == {"normal": 1, "relational": 1}


def test_load_hybrid_loads_clean_config_with_default_loader(tmp_path, monkeypatch) -> None:
    normal_source, relational_source = _synthetic_sources(tmp_path, monkeypatch, valid_weights=True)
    artifact = prepare_artifact(normal_source, relational_source, tmp_path / "artifact")
    normal = FakeNormal(hidden_size=1536)
    bundle = load_hybrid(artifact, device="cpu", normal_loader=lambda *args, **kwargs: normal)
    assert bundle.engine.residual.config.hidden_size == 1536
    assert not any(parameter.requires_grad for parameter in bundle.engine.residual.parameters())


def test_manifest_rejects_noncanonical_json_bytes(tmp_path, monkeypatch) -> None:
    normal, relational = _synthetic_sources(tmp_path, monkeypatch)
    manifest = prepare_artifact(normal, relational, tmp_path / "artifact")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="canonical portable manifest"):
        validate_manifest(manifest)


@pytest.mark.skipif(
    not os.environ.get("STRANDS_GEMMA_E2B_ARTIFACT_URI"),
    reason="requires a local released artifact URI",
)
def test_released_artifact_validates_loads_and_scores_both_modes(monkeypatch) -> None:
    raw_uri = os.environ["STRANDS_GEMMA_E2B_ARTIFACT_URI"]
    parsed = urlsplit(raw_uri)
    if parsed.scheme not in ("", "file") or parsed.netloc not in ("", "localhost"):
        pytest.fail("STRANDS_GEMMA_E2B_ARTIFACT_URI must be a local path or file URI")
    artifact_path = Path(unquote(parsed.path if parsed.scheme else raw_uri))
    manifest_path = artifact_path / MANIFEST_NAME if artifact_path.is_dir() else artifact_path
    validated = validate_manifest(manifest_path)
    assert validated.portable_manifest_sha256 == artifacts.PORTABLE_MANIFEST_SHA256

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    import huggingface_hub.constants
    import transformers.utils.hub

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(transformers.utils.hub, "is_offline_mode", lambda: True)
    bundle = load_hybrid(
        manifest_path,
        device=os.environ.get("STRANDS_GEMMA_E2B_DEVICE", "cuda"),
    )
    question = ChoiceQuestion(instructions="Pick one.", criteria={"left": "L", "right": "R"})
    normal_scores = bundle.engine.score_question("state", question, mode="normal")
    compare_scores = bundle.engine.score_question("state", question, mode="compare_choices")
    assert normal_scores.logits.shape == compare_scores.logits.shape == (2,)
    assert torch.isfinite(normal_scores.logits).all()
    assert torch.isfinite(compare_scores.logits).all()
    assert torch.allclose(
        normal_scores.probabilities.sum(), normal_scores.probabilities.new_tensor(1.0)
    )
    assert torch.allclose(
        compare_scores.probabilities.sum(), compare_scores.probabilities.new_tensor(1.0)
    )
