"""Assemble exact local normal and relational component directories.

This tool copies local artifacts into an ignored output directory and records
relative paths plus hashes. It never downloads or embeds weights in package code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from strands_gemma_order_stable_hybrid.artifacts import prepare_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--relational", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_artifact(args.normal, args.relational, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
