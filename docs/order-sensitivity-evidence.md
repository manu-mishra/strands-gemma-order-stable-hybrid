# Evidence that choice order changed the original model's answer

## Claim

Under the controlled standard evaluation, the pinned original Strands Gemma E2B checkpoint changed its winning **semantic option** on **851 of 3,900 questions** when the same options were presented in complete reverse order.

```text
851 / 3,900 = 0.21820512820512822 = 21.8205128205%
```

The model card displays this as **21.82%**.

This is evidence for one defined treatment—original order versus complete reversal—not a claim about every possible permutation, prompt format, runtime, or model.

## What changed between the two calls?

For each source row, the harness issued two separate requests to the same endpoint and model:

1. the options in source order;
2. the identical options in complete reverse order.

The state, question text, model name, endpoint, and option contents were held fixed. Option text was whitespace-normalized once before either request. An audit found no duplicate normalized option strings, so constructing the option mapping did not collapse choices.

Reversal necessarily changes option numbers, token positions, and neighboring option text in the original jointly encoded prompt. Those are the positional effects the experiment was designed to measure.

## Why this is a semantic flip rather than an index mistake

The original API returned probabilities keyed by semantic option text. The harness looked up each returned probability using the source option identities and stored both `orig` and `rev` vectors in the original source-option order.

The statistic was therefore:

```python
argmax(original_probabilities) != argmax(reversed_probabilities)
```

Both sides refer to the same semantic option indices. The reversed vector must not be reversed a second time. Doing so would produce a very different and incorrect count.

## Exact per-task counts

Every task contributed 300 rows.

| Task | Semantic flips | Rows | Rate |
|---|---:|---:|---:|
| ARC Challenge | 54 | 300 | 18.00% |
| ARC Easy | 27 | 300 | 9.00% |
| Banking77 | 30 | 300 | 10.00% |
| BoolQ | 3 | 300 | 1.00% |
| CommonsenseQA | 57 | 300 | 19.00% |
| HellaSwag | 93 | 300 | 31.00% |
| MASSIVE Intent | 31 | 300 | 10.33% |
| MMLU | 90 | 300 | 30.00% |
| OpenBookQA | 54 | 300 | 18.00% |
| PIQA | 98 | 300 | 32.67% |
| SciQ | 7 | 300 | 2.33% |
| Social IQA | 38 | 300 | 12.67% |
| WinoGrande | 269 | 300 | 89.67% |
| **Total** | **851** | **3,900** | **21.8205%** |

The variation by task is itself useful evidence. The effect was almost absent on BoolQ and SciQ, but very large on WinoGrande. This is not the pattern expected from accidentally comparing physical slot numbers alone.

## Hash-bound evidence chain

| Evidence | SHA-256 |
|---|---|
| Standard source, 3,900 rows | `0065f310cd369c9197478bbf68b59b7b9ebe6bcbd6dc699433ec8126f3da2d61` |
| Original/reversed baseline predictions | `48a9908778fd7e2661f83ce18edb0e2a2c4a794845aef44d8267afb42bee8155` |
| Baseline generation code | `62ac76f86b32dfb2132b8ff32d119ddd0eb7715f6dc4d98f9d2ab153699c6612` |
| Frozen prospective requirements | `f77270c9b41bcda06b55080a7365f1f22e96bd43974e1aa41aef102cfac0d79d` |
| Model/source/checkpoint authority | `c28621d7c86cb394359194b02e02c65932187a952e7d6a0ac593044f4c85cbeb` |
| Accepted normal report | `3fbbbd1f264c6fb296fa74caa4c36ea3b4e40a405e54824861e6280cea0234e2` |
| Terminal hybrid report | `0889747869f2f07d157370072a578db6aaa6f948acad0f317470fb176102ca30` |

The authority pins:

- checkpoint `StrandsAgents/strands-decider-E2B-gemma4-v1-2610` at revision `d383ecf8534b13d49b3b567a1b66f88fca6d50ba`;
- Gemma base `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`;
- Strands source at commit `0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534`.

The raw baseline file is not published because it contains all item-level predictions and labels. The released [`hf-evidence.json`](../results/hf-evidence.json) contains the exact aggregate sufficient statistics, plus the two attributed and paraphrased failure examples shown on the model page, without publishing the full rows.

## Independent recomputation algorithm

Given the hash-matched baseline JSONL:

```python
import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("strands-gemma-e2b.jsonl").read_text().splitlines()
    if line.strip()
]

argmax = lambda values: max(range(len(values)), key=values.__getitem__)
flips = sum(argmax(row["orig"]) != argmax(row["rev"]) for row in rows)

assert len(rows) == 3900
assert flips == 851
assert flips / len(rows) == 0.21820512820512822
```

A separate alignment check matched all 3,900 `(task, item_id, label)` triples between source and baseline files and verified that source option count matched both stored probability-vector widths.

## Evidence limitations

- The original baseline is an API-level result. Returned probabilities were rounded to four decimals and renormalized by the harness; this is not raw-logit evidence.
- The two arms were separate sequential API calls, so ordinary repeated-execution numerical variation is not independently estimated by this artifact.
- The raw baseline JSONL does not self-record checkpoint revision, source hash, or protocol. Those identities are supplied by the later authority and final-report bindings.
- The historical serving checkout was declared at the pinned source commit, but the transient served directory was not itself hashed in the raw baseline artifact. The later authority validates a checkout and critical files at that declared commit.
- This original-model statistic covers complete reversal only. The retrofit's broader permutation evidence is a separate result.

These qualifications narrow the claim but do not change the observed artifact: the fixed baseline prediction file contains 851 semantic winner changes after reversal.

## Related research

Choice-order sensitivity is established beyond this checkpoint:

1. [Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions](https://arxiv.org/abs/2308.11483) reports substantial performance variation after answer options are reordered and connects the effect to positional bias.
2. [On Large Language Models' Selection Bias in Multi-Choice Questions](https://arxiv.org/abs/2309.03882) studies option-position and option-label bias and proposes a debiasing method.
3. [Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models](https://aclanthology.org/2024.findings-acl.333/) examines both order and token sensitivity in ordered-choice selection.
4. [When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards](https://aclanthology.org/2024.acl-long.744/) shows that seemingly minor benchmark changes, including choice order, can alter model rankings.
5. [Order-Independence Without Fine Tuning](https://papers.neurips.cc/paper_files/paper/2024/file/85529bc995777a74072ef63c05bedd30-Paper-Conference.pdf) introduces Set-Based Prompting as an architectural approach to removing order information from designated subsequences.

Our claim is narrower than this literature: we measured one pinned decision checkpoint, then converted its scoring path and documented the backend boundary under which zero flips were observed.
