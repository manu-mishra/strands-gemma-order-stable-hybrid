# Option-order sensitivity across public decision models

## Summary

The original Strands Gemma E2B checkpoint is not unusual. Every publicly available decision model tested here changed a meaningful share of its answers when the same options were presented in reverse order.

| Model | Revision | Answers changed by reversal | Standard macro accuracy (original order) |
|---|---|---:|---:|
| [Laya](https://huggingface.co/convaiinnovations/laya) (English, `laya==0.3.28`) | `7b928d82` | 1,426 / 3,900 (**36.56%**) | 49.49% |
| [Strands Gemma E2B v1](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610) | `d383ecf8` | 851 / 3,900 (**21.82%**) | 73.00% |
| [Strands Decider 2B Hobson v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) (`strands-decider==0.1.0`) | `2b52a623` | 839 / 3,900 (**21.51%**) | 72.13% |
| [Strands Gemma E4B v1](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610) | `1344d443` | 563 / 3,900 (**14.44%**) | 79.26% |
| **This retrofit**, `normal` and `compare_choices` | `v0.1.0` | **0 / 3,900** (both modes) | 72.49% / 72.46% |

Larger and more accurate models flipped less often, but none reached zero. Strands Gemma E4B, the most accurate model here, still changed about one answer in seven.

## What was measured

- **Items:** the same 3,900 questions used for this repository's standard suite. There are 13 public tasks with 300 items each, selected by a fixed hash-based sample of each evaluation split.
- **Treatment:** each item was sent twice to the same model and runtime. The first request used the source option order and the second used the complete reverse order. The state, question and option texts were identical.
- **Flip definition:** returned probabilities were mapped back to the source option identities. A flip is counted when `argmax(original) != argmax(reversed)` over the same semantic options, using the method described in [`order-sensitivity-evidence.md`](order-sensitivity-evidence.md).
- **Interface:** each system used its own native typed `choice` API, with the option texts as labels. Strands Decider rejects an empty state, so for context-free items the question text was placed in the state and the instruction was "Choose the correct answer." This was done for Strands Decider and Laya.
- **Hardware:** one NVIDIA L4 (24 GB). Measured October 2026. The Gemma models used eager attention as required upstream.
- **Errors:** zero failed or invalid responses for every model listed.

The E2B row reproduces the count in [`order-sensitivity-evidence.md`](order-sensitivity-evidence.md) exactly: 851 flips with the same per-task counts.

## Per-task flips (of 300 items each)

| Task | Laya | Strands v21 | Gemma E2B | Gemma E4B | This retrofit |
|---|---:|---:|---:|---:|---:|
| ARC Challenge | 119 | 56 | 54 | 37 | 0 |
| ARC Easy | 75 | 17 | 27 | 8 | 0 |
| Banking77 | 169 | 20 | 30 | 19 | 0 |
| BoolQ | 10 | 5 | 3 | 5 | 0 |
| CommonsenseQA | 67 | 71 | 57 | 39 | 0 |
| HellaSwag | 193 | 57 | 93 | 55 | 0 |
| MASSIVE Intent | 159 | 30 | 31 | 25 | 0 |
| MMLU | 126 | 105 | 90 | 60 | 0 |
| OpenBookQA | 89 | 54 | 54 | 50 | 0 |
| PIQA | 138 | 141 | 98 | 52 | 0 |
| SciQ | 17 | 4 | 7 | 3 | 0 |
| Social IQA | 95 | 42 | 38 | 27 | 0 |
| WinoGrande | 169 | 237 | 269 | 183 | 0 |
| **Total** | **1,426** | **839** | **851** | **563** | **0** |

The pattern repeats across models:

- **WinoGrande is the worst case for every jointly encoded model.** It is a two-option task in which both options are near-identical sentences. Strands v21 changed 79% of its answers and Gemma E2B 90%. Its accuracy there (53–54%) is close to the 50% chance level, so most of its WinoGrande "decisions" follow option position rather than content.
- **PIQA, MMLU and HellaSwag are among the most sensitive tasks** for most models. PIQA is also a two-option task with long, similar solutions.
- **BoolQ and SciQ are nearly stable everywhere.** In those tasks the answer is usually clear-cut, or the options differ sharply.

## How to read these numbers

- These figures cover complete reversal only. They are a lower bound on order sensitivity, not an estimate over all permutations. The retrofit's broader permutation evidence is reported separately in the [README](../README.md).
- The figures are specific to the listed revisions, package versions, interface and runtime. Other prompts, option formats or releases may behave differently.
- A flip does not always turn a correct answer into a wrong one. Original-order and reversed-order accuracy can be similar while individual answers change. For example, Strands v21 scored 72.13% in original order and 71.82% reversed, yet changed 839 answers.
- The comparison concerns option-order stability only. On other quality measures, Strands Gemma E4B in particular is more accurate than this retrofit. This document does not claim the retrofit is a better general decision model.
- The retrofit's zero result holds for its supported execution path: serialized, one request at a time, batch-one full recomputation. See the [README](../README.md) for the unsupported caching and cross-request batching paths.
- Strands Gemma 12B, Strands Gemma 26B-A4B and Strands Qwen3.5 2B v1 were not measured under this protocol, so no flip rate is reported for them.
