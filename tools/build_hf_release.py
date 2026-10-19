"""Build the Hugging Face release payload.

The input must already be a portable artifact authenticated by this package. The
output combines those exact model components with a Hub model card, aggregate-only
results, architecture notes, the experiment narrative, and required notices.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from strands_gemma_order_stable_hybrid.artifacts import (
    MANIFEST_NAME,
    PORTABLE_MANIFEST_SHA256,
    validate_manifest,
)

FRONT_MATTER = """---
license: apache-2.0
base_model:
  - google/gemma-4-E2B-it
  - StrandsAgents/strands-decider-E2B-gemma4-v1-2610
base_model_relation: adapter
library_name: strands-gemma-order-stable-hybrid
pipeline_tag: text-classification
tags:
  - gemma
  - strands-decider
  - decision-model
  - multiple-choice
  - order-stable
  - set-model
---
"""

ROOT_FILES = ("LICENSE", "LICENSE-MIT", "NOTICE", "THIRD_PARTY_NOTICES.md")


class HuggingFaceReleaseError(RuntimeError):
    """Raised when a Hub release payload cannot be built exactly."""


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _hub_model_card(project_root: Path) -> str:
    card = (project_root / "HF_MODEL_CARD.md").read_text(encoding="utf-8")
    if card.startswith("---"):
        raise HuggingFaceReleaseError("source model card already contains front matter")
    return FRONT_MATTER + "\n" + card


def build_release(*, artifact: Path, project_root: Path, output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise HuggingFaceReleaseError(f"output already exists: {output}")
    validated = validate_manifest(artifact / MANIFEST_NAME if artifact.is_dir() else artifact)
    if validated.portable_manifest_sha256 != PORTABLE_MANIFEST_SHA256:
        raise HuggingFaceReleaseError(
            "portable artifact digest does not match the released identity"
        )
    project_root = project_root.resolve()
    try:
        output.mkdir(parents=True)
        shutil.copy2(validated.root / MANIFEST_NAME, output / MANIFEST_NAME)
        shutil.copytree(validated.normal.path, output / "normal")
        shutil.copytree(validated.relational.path, output / "relational")
        card = _hub_model_card(project_root)
        (output / "README.md").write_text(card, encoding="utf-8")
        for name in ROOT_FILES:
            shutil.copy2(project_root / name, output / name)
        (output / "results").mkdir()
        shutil.copy2(
            project_root / "results/hf-evidence.json",
            output / "results/hf-evidence.json",
        )
        (output / "docs").mkdir()
        shutil.copy2(
            project_root / "docs/order-sensitivity-evidence.md",
            output / "docs/order-sensitivity-evidence.md",
        )

        forbidden_names = {"candidate-lock.json", "relational-candidate-lock.json"}
        if any(path.name in forbidden_names for path in output.rglob("*")):
            raise HuggingFaceReleaseError("machine-bound historical lock entered release")
        if any(path.suffix in {".pt", ".pth", ".ckpt", ".bin"} for path in output.rglob("*")):
            raise HuggingFaceReleaseError("unsafe model format entered release")

        entries = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                entries.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
        (output / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
        revalidated = validate_manifest(output / MANIFEST_NAME)
        if revalidated.portable_manifest_sha256 != PORTABLE_MANIFEST_SHA256:
            raise HuggingFaceReleaseError("release copy changed the portable artifact")
        return output
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    output = build_release(
        artifact=args.artifact,
        project_root=args.project_root,
        output=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
