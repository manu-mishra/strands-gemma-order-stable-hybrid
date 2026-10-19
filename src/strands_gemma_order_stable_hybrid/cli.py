"""Command-line interface for validation, preparation, and loopback serving."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import MODES, prepare_artifact, validate_manifest
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strands-gemma-order-stable-hybrid")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a portable artifact")
    validate.add_argument("artifact", type=Path)

    prepare = commands.add_parser("prepare-artifact", help="copy exact local components")
    prepare.add_argument("--normal", type=Path, required=True)
    prepare.add_argument("--relational", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    server = commands.add_parser("serve", help="serve a portable artifact on loopback")
    server.add_argument("artifact", type=Path)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--device", default="cuda")
    server.add_argument("--attention-implementation", default="eager")
    server.add_argument("--default-mode", choices=MODES, default="normal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        path = args.artifact / "hybrid-manifest.json" if args.artifact.is_dir() else args.artifact
        validate_manifest(path)
        return 0
    if args.command == "prepare-artifact":
        prepare_artifact(args.normal, args.relational, args.output)
        return 0
    serve(
        args.artifact,
        host=args.host,
        port=args.port,
        device=args.device,
        attention_implementation=args.attention_implementation,
        default_mode=args.default_mode,
    )
    return 0
