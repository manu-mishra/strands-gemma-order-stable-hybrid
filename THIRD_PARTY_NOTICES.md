# Third-party notices

This document identifies third-party source and model components referenced by the Strands Gemma Order-Stable Hybrid release package. GitHub hosts the released source distribution and sanitized evidence; Hugging Face hosts the released model components. Neither location publishes bulk raw datasets or complete item-level result files. The model page includes two attributed, paraphrased ARC Easy failure examples under CC BY-SA 4.0.

## Strands Decider

- Project: [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider)
- Pinned source: commit [`0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534`](https://github.com/strands-labs/strands-decider/tree/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534)
- License: Apache License 2.0; full text in [`LICENSE`](LICENSE)

The isolated-option module subclasses and uses the Strands typed engine, renderer, schema, model wrapper, and `PointerHead` interfaces. The actual distributed Apache-2.0 adaptation paths are:

- `src/strands_gemma_order_stable_hybrid/option_branches.py` — adapts the accepted isolated-option retrofit and composes with the pinned Strands interfaces; this exact package path did not exist in the pinned upstream tree.
- `tests/test_option_branches.py` — tests the Strands-derived normal-path adaptation.

Changes: render one isolated branch per semantic option; enforce identical prefixes and equal logical option positions; perform one full batch-one encode per option without a prefix cache; expose existing pointer projections for relational features; map output by semantic identity; and test permutation, parity, context-window, and schema behavior.

The following distributed project files are separately authored or adapted from this project's completed experiment and are offered under the MIT terms in [`LICENSE-MIT`](LICENSE-MIT):

- `src/strands_gemma_order_stable_hybrid/hybrid.py`
- `src/strands_gemma_order_stable_hybrid/server.py`
- `src/strands_gemma_order_stable_hybrid/set_residual.py`
- `src/strands_gemma_order_stable_hybrid/artifacts.py`
- `tests/test_hybrid_artifacts.py`
- `tests/test_server.py`
- `tests/test_set_residual.py`

`hybrid.py`, `server.py`, and `set_residual.py` package the completed hybrid-runtime, server, and tied-residual work for installed imports and strict portable files. `artifacts.py` is the project-authored preparation and validation layer. The listed tests cover those MIT components. Importing or composing Apache-2.0 Strands interfaces does not remove the MIT offer for separately authored files, and MIT terms do not replace Apache-2.0 obligations for the Strands-derived normal-path adaptation. No evaluation benchmark module is distributed or attributed here.

### Notice inherited from the pinned Strands source

The pinned Strands source reports that `src/strands_decider/mps_kernels.py` adapts `torch_chunk_gated_delta_rule` from the Qwen3.5 model implementation in Hugging Face Transformers 5.17 (`transformers/models/qwen3_5/modeling_qwen3_5.py`).

Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

Licensed under Apache-2.0. The upstream notice states that two triangular solves were replaced by one recursive block inversion of the unit-lower-triangular system, the result was applied to both right-hand sides, and the function was restructured with renamed variables. See the full Apache-2.0 text in [`LICENSE`](LICENSE), the [pinned Strands notice](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/THIRD_PARTY_NOTICES.md), and [Hugging Face Transformers](https://github.com/huggingface/transformers).

## Strands Gemma E2B checkpoint

- Model: [`StrandsAgents/strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610)
- Pinned revision: [`d383ecf8534b13d49b3b567a1b66f88fca6d50ba`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/tree/d383ecf8534b13d49b3b567a1b66f88fca6d50ba)
- Reported checkpoint license: Apache-2.0, with separate source-data restrictions noted by the publisher

The checkpoint supplies the inherited adapter and decision-head lineage evaluated by this project. It does not contain the Gemma base weights, and its adapter metadata does not pin a base revision. The released normal component includes the inherited adapter; the complete upstream checkpoint is not republished. The separately hosted Gemma base remains subject to its publisher’s applicable terms, as does each underlying data source.

## Gemma base

- Model: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
- Pinned revision: [`3e22461f65e89153144f8adb70e3b8c2cc9845a7`](https://huggingface.co/google/gemma-4-E2B-it/tree/3e22461f65e89153144f8adb70e3b8c2cc9845a7)
- Publisher resources: [Gemma license](https://ai.google.dev/gemma/apache_2), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), and [Gemma terms](https://ai.google.dev/gemma/terms)

The Gemma base is required to reconstruct the evaluated model identity. It is not bundled in this repository. Users must obtain it directly from the publisher under the applicable terms and use the exact revision above.

## Data and evaluation material

This release does not publish training corpora, evaluation corpora, JEVBench item material, or complete prompt/answer/prediction files. It publishes two paraphrased ARC Easy examples—items `Mercury_7044555` and `NYSEDREGENTS_2015_8_32`—with source attribution, item IDs, correct answers, and the original checkpoint’s mapped probabilities. ARC is distributed under CC BY-SA 4.0; the examples are reformatted and marked as changed. Aggregate measurements and these examples do not convey rights to other underlying dataset material. Anyone reproducing an evaluation is responsible for obtaining each dataset under its own terms.

## No trademark grant

Names and links are used only to identify component origins and reproduce attribution. Neither Apache-2.0 nor these notices grant permission to use third-party trade names or trademarks beyond reasonable and customary identification.
