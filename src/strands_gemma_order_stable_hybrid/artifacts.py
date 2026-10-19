"""Authenticated, relocatable artifact for the frozen Strands Gemma E2B hybrid."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "hybrid-manifest.json"
FORMAT = "strands-gemma-order-stable-hybrid-v1"
SCHEMA_VERSION = 1
MODES = ("normal", "compare_choices")
DEFAULT_MODE = "normal"
PORTABLE_MANIFEST_SHA256 = "faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974"

UPSTREAM_IDENTITIES = {
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
SELECTED_CANDIDATE_IDS = {
    "normal": "wd-0.1-epochs-25-seed-43",
    "compare_choices": "rank-128-hidden-64-seed-17",
}
HISTORICAL_LOCK_HASHES = {
    "normal": "a1bee0186af38614372ba2f6e2f058e8ebc94d5b72e0f2d18e87ac36d405cf5a",
    "relational": "a1738049034a10890c74b6777ecfd015835fd74c98af71dc1e4c6479f9145b49",
    "hybrid": "47748ebb2aca6df06dac46c4a91d075726cf96ef3d4d3c3319a6225bd8550926",
}
NORMAL_SOURCE_FILE_HASHES = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "LICENSE.md": "0a526e4695eb2b9e8cd57655446ed77c3d4a35e55165f434ee5479071ce44ca0",
    "MANIFEST.sha256": "0449ed5da191b18eadfe8fafb527357e3c28f9c7be5a98c73a8ff352abc8205e",
    "README.md": "84fd725059fb1cbdbf82a312e8ec7aa6cdf4e9dc75e4c92aa28161055a2a6f29",
    "chat_template.jinja": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    "head.safetensors": "781f0f460975316cac306ae8bb7fa11dd1471d3e8b183140ac8c0aaf1a02e915",
    "lora/adapter_config.json": "0b9bf880b630441245e20eed71afb8a243cee13d7f0b7022c9e0d77bb267b7bc",
    "lora/adapter_model.safetensors": "c7786a59354aa190082b1c3aedb7ff3311fe260f070f360c4d3518762691c551",
    "provenance.json": "214928b49e4ebb43833c345882662ef5c60917c9fafd7e99a1144ee3d9c5582b",
    "strands_decider_config.json": "acd3fe07068d53f3cc4aa77f93c2c5122e9c27235c28f2ec1b713418aeded6d9",
    "tokenizer.json": "30383f3ce90a169cbabe9eb3da7bb77988bf906aad2e59184b378f5470ec4dbd",
    "tokenizer_config.json": "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999",
    "train_config.json": "06789e5e8ac6f89885489ac82319900f477ec7e392b1a9244f3e8727c71ec636",
}
RELATIONAL_SOURCE_FILE_HASHES = {
    "set_residual.safetensors": "afe53a76352e389b8aba2d92599742e20770e9a374baafb401c7a283ce22fca3",
    "set_residual_config.json": "62098a5b5c15a5f5b624ce84cf53eba2534589ff5bdb9406076680834c5c7859",
}
NORMAL_OUTPUT_MEMBERS = (
    "LICENSE.md",
    "README.md",
    "chat_template.jinja",
    "head.safetensors",
    "lora/adapter_config.json",
    "lora/adapter_model.safetensors",
    "provenance.json",
    "strands_decider_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CLEAN_RESIDUAL_CONFIG = {
    "hidden_size": 1536,
    "rank": 128,
    "mlp_hidden": 64,
    "format": "e2b-content-only-tied-set-residual-v1",
    "position_inputs": False,
    "pooling": "masked-mean",
    "attention": "content-dot-product",
    "output": "shared-zero-initialized-residual",
}
CLEAN_RESIDUAL_CONFIG_SHA256 = "b3d0973c5ed2480d6ddcd8d50b954f65653986e017223ea06a02de3c024b8a2b"
NORMAL_LOCK_NAME = "candidate-lock.json"
RELATIONAL_LOCK_NAME = "relational-candidate-lock.json"
RESIDUAL_CONFIG_NAME = "set_residual_config.json"
RESIDUAL_WEIGHTS_NAME = "set_residual.safetensors"


class ArtifactError(ValueError):
    """Raised for malformed, unsafe, incomplete, changed, or identity-mismatched artifacts."""


@dataclass(frozen=True)
class Component:
    path: Path
    files: dict[str, str]


@dataclass(frozen=True)
class ValidatedArtifact:
    root: Path
    normal: Component
    relational: Component
    manifest: dict[str, Any]
    portable_manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_config_bytes() -> bytes:
    return (json.dumps(CLEAN_RESIDUAL_CONFIG, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _relative(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{field} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ArtifactError(f"{field} must not be absolute or contain traversal")
    if "\\" in value or candidate.as_posix() != value:
        raise ArtifactError(f"{field} must be normalized POSIX syntax")
    return candidate


def _inside(root: Path, relative: PurePosixPath, *, field: str) -> Path:
    target = root.joinpath(*relative.parts)
    try:
        target.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{field} escapes or is missing from artifact root") from exc
    return target


def _inventory(directory: Path) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise ArtifactError("component must be a real directory")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ArtifactError("component symlinks are not allowed")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            _relative(relative, field="component member")
            files[relative] = sha256_file(path)
        elif not path.is_dir():
            raise ArtifactError("component contains a non-regular member")
    if not files:
        raise ArtifactError("component must contain at least one file")
    return files


def _read_json(path: Path, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ArtifactError(f"{label} contains duplicate JSON fields")
            value[key] = item
        return value

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is unreadable or invalid JSON") from exc


def _require_inventory(actual: dict[str, str], expected: dict[str, str], *, label: str) -> None:
    if actual != expected:
        raise ArtifactError(f"{label} inventory or hash does not match the released identity")


def _normal_source_inventory() -> dict[str, str]:
    return {**NORMAL_SOURCE_FILE_HASHES, NORMAL_LOCK_NAME: HISTORICAL_LOCK_HASHES["normal"]}


def _relational_source_inventory() -> dict[str, str]:
    return {
        **RELATIONAL_SOURCE_FILE_HASHES,
        RELATIONAL_LOCK_NAME: HISTORICAL_LOCK_HASHES["relational"],
    }


def _normal_output_inventory() -> dict[str, str]:
    return {name: NORMAL_SOURCE_FILE_HASHES[name] for name in NORMAL_OUTPUT_MEMBERS}


def _relational_output_inventory() -> dict[str, str]:
    return {
        RESIDUAL_CONFIG_NAME: CLEAN_RESIDUAL_CONFIG_SHA256,
        RESIDUAL_WEIGHTS_NAME: RELATIONAL_SOURCE_FILE_HASHES[RESIDUAL_WEIGHTS_NAME],
    }


def _validate_normal_lock(path: Path) -> None:
    raw = _read_json(path, label="normal historical lock")
    try:
        valid = (
            raw["candidate"]["selected_candidate_id"] == SELECTED_CANDIDATE_IDS["normal"]
            and raw["candidate"]["files"] == NORMAL_SOURCE_FILE_HASHES
            and raw["provenance"]["source_commit"]
            == UPSTREAM_IDENTITIES["strands_source"]["commit"]
            and raw["provenance"]["model_revision"] == UPSTREAM_IDENTITIES["checkpoint"]["revision"]
            and raw["provenance"]["base_model"] == UPSTREAM_IDENTITIES["gemma"]["repo"]
            and raw["provenance"]["base_revision"] == UPSTREAM_IDENTITIES["gemma"]["revision"]
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ArtifactError("normal historical lock identity does not match the released identity")


def _validate_relational_source(path: Path) -> None:
    config = _read_json(path / RESIDUAL_CONFIG_NAME, label="relational source config")
    expected_architecture = {
        "format": CLEAN_RESIDUAL_CONFIG["format"],
        "hidden_size": 1536,
        "mlp_hidden": 64,
        "output": "shared residual",
        "pooling": "masked symmetric mean",
        "positions": False,
        "rank": 128,
    }
    try:
        config_valid = (
            config["architecture"] == expected_architecture
            and config["selection"]["selected_candidate_id"]
            == SELECTED_CANDIDATE_IDS["compare_choices"]
            and config["normal_candidate"]["lock_sha256"] == HISTORICAL_LOCK_HASHES["normal"]
            and config["normal_candidate"]["head_sha256_before"]
            == NORMAL_SOURCE_FILE_HASHES["head.safetensors"]
            and config["normal_candidate"]["head_sha256_after"]
            == NORMAL_SOURCE_FILE_HASHES["head.safetensors"]
        )
    except (KeyError, TypeError):
        config_valid = False
    if not config_valid:
        raise ArtifactError(
            "relational source architecture or selection does not match the released identity"
        )

    lock = _read_json(path / RELATIONAL_LOCK_NAME, label="relational historical lock")
    try:
        lock_valid = (
            lock["candidate"]["selected_candidate_id"] == SELECTED_CANDIDATE_IDS["compare_choices"]
            and lock["candidate"]["format"] == CLEAN_RESIDUAL_CONFIG["format"]
            and lock["candidate"]["files"] == RELATIONAL_SOURCE_FILE_HASHES
            and lock["normal_candidate"]["lock_sha256"] == HISTORICAL_LOCK_HASHES["normal"]
            and lock["normal_candidate"]["head_sha256"]
            == NORMAL_SOURCE_FILE_HASHES["head.safetensors"]
        )
    except (KeyError, TypeError):
        lock_valid = False
    if not lock_valid:
        raise ArtifactError(
            "relational historical lock identity does not match the released identity"
        )


def _component(
    root: Path,
    raw: object,
    *,
    name: str,
    expected_path: str,
    expected_files: dict[str, str],
) -> Component:
    expected_raw = {"path": expected_path, "files": expected_files}
    if raw != expected_raw:
        raise ArtifactError(f"{name} component declaration does not match the released identity")
    component_path = _inside(root, _relative(expected_path, field=f"{name}.path"), field=name)
    actual = _inventory(component_path)
    _require_inventory(actual, expected_files, label=name)
    return Component(component_path, dict(expected_files))


def _expected_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "upstream": UPSTREAM_IDENTITIES,
        "selected_candidate_ids": SELECTED_CANDIDATE_IDS,
        "historical_lock_sha256": HISTORICAL_LOCK_HASHES,
        "source_components": {
            "normal": NORMAL_SOURCE_FILE_HASHES,
            "relational": RELATIONAL_SOURCE_FILE_HASHES,
        },
        "modes": list(MODES),
        "default_mode": DEFAULT_MODE,
        "normal": {"path": "normal", "files": _normal_output_inventory()},
        "relational": {"path": "relational", "files": _relational_output_inventory()},
    }


def _canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _expected_manifest_sha256() -> str:
    return hashlib.sha256(_canonical_manifest_bytes(_expected_manifest())).hexdigest()


def _assert_production_manifest_identity() -> None:
    computed = _expected_manifest_sha256()
    if computed != PORTABLE_MANIFEST_SHA256:
        raise RuntimeError(
            "canonical portable manifest SHA-256 mismatch: "
            f"expected {PORTABLE_MANIFEST_SHA256}, computed {computed}"
        )


_assert_production_manifest_identity()


def validate_manifest(manifest_path: Path) -> ValidatedArtifact:
    """Authenticate an exact frozen artifact using only relative paths and pinned bytes."""
    if (
        manifest_path.name != MANIFEST_NAME
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise ArtifactError(f"manifest must be a real {MANIFEST_NAME} file")
    raw = _read_json(manifest_path, label="manifest")
    expected = _expected_manifest()
    canonical_bytes = _canonical_manifest_bytes(expected)
    if raw != expected or manifest_path.read_bytes() != canonical_bytes:
        raise ArtifactError(
            "canonical portable manifest identity or bytes do not match the released identity"
        )
    root = manifest_path.parent
    normal = _component(
        root,
        raw["normal"],
        name="normal",
        expected_path="normal",
        expected_files=_normal_output_inventory(),
    )
    relational = _component(
        root,
        raw["relational"],
        name="relational",
        expected_path="relational",
        expected_files=_relational_output_inventory(),
    )
    clean_config = _read_json(relational.path / RESIDUAL_CONFIG_NAME, label="portable config")
    if clean_config != CLEAN_RESIDUAL_CONFIG:
        raise ArtifactError("portable residual config is not the canonical direct config")
    portable_manifest_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    return ValidatedArtifact(root, normal, relational, raw, portable_manifest_sha256)


def _copy_members(source: Path, target: Path, members: tuple[str, ...]) -> None:
    target.mkdir(parents=True)
    for member in members:
        relative = _relative(member, field="copied member")
        source_file = source.joinpath(*relative.parts)
        target_file = target.joinpath(*relative.parts)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target_file)


def prepare_artifact(normal_source: Path, relational_source: Path, output: Path) -> Path:
    """Authenticate accepted source bytes, then create the minimal portable artifact."""
    if output.exists():
        raise ArtifactError(f"output already exists: {output}")
    normal_source = Path(normal_source)
    relational_source = Path(relational_source)
    normal_inventory = _inventory(normal_source)
    relational_inventory = _inventory(relational_source)
    _require_inventory(normal_inventory, _normal_source_inventory(), label="normal source")
    _require_inventory(
        relational_inventory, _relational_source_inventory(), label="relational source"
    )
    _validate_normal_lock(normal_source / NORMAL_LOCK_NAME)
    _validate_relational_source(relational_source)

    try:
        output.mkdir(parents=True)
        normal_target = output / "normal"
        relational_target = output / "relational"
        _copy_members(normal_source, normal_target, NORMAL_OUTPUT_MEMBERS)
        _copy_members(relational_source, relational_target, (RESIDUAL_WEIGHTS_NAME,))
        (relational_target / RESIDUAL_CONFIG_NAME).write_bytes(_clean_config_bytes())
        manifest = output / MANIFEST_NAME
        manifest.write_bytes(_canonical_manifest_bytes(_expected_manifest()))
        validate_manifest(manifest)
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
