# Choice order should be incidental

**Strands Gemma E2B — Order-Stable Hybrid** addresses a measured failure in the released decision checkpoint: on 3,900 controlled questions, reversing otherwise identical choices changed the selected answer **21.82% of the time**. This release replaces the ordered-choice scoring path with isolated option scoring and an optional relational mode while keeping the Gemma torso and inherited Strands adapter frozen. The zero-flip result is deliberately narrow: it was demonstrated for serialized, batch-one, eager, full-recomputation inference; shared-prefix caching and cross-request batching are not supported.

## Why choice order matters

Choices are usually a set, not a sequence. Sorting them alphabetically, by price, or by an internal identifier should not change what any choice means. A decision should follow the question and the content of each option—not where an interface happened to place it.

The original checkpoint evaluated every choice inside one ordered prompt. Reordering the list therefore changed token positions and the context surrounding each option. We tested whether those representational changes were large enough to change the final answer.

## How large was the measured problem?

The **21.82%** figure came from **851 semantic flips in 3,900 controlled questions** when choice order was completely reversed while text and available answers remained fixed. The exact rate is `0.21820512820512822`; 21.82% is its two-decimal display. That experiment set the retrofit target: preserve the selected semantic answer when only order changes, within a clearly stated execution boundary.

## Is choice-order sensitivity a known problem?

Yes. This checkpoint-specific result is consistent with a broader research record:

- [Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions](https://arxiv.org/abs/2308.11483) measures substantial performance changes after answer options are reordered.
- [On Large Language Models’ Selection Bias in Multi-Choice Questions](https://arxiv.org/abs/2309.03882) studies biases associated with option positions and option labels.
- [Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models](https://aclanthology.org/2024.findings-acl.333/) separates sensitivity to ordering from sensitivity to option tokens.
- [When Benchmarks are Targets](https://aclanthology.org/2024.acl-long.744/) shows that choice-order and answer-selection changes can alter benchmark rankings.
- [Order-Independence Without Fine Tuning](https://papers.neurips.cc/paper_files/paper/2024/file/85529bc995777a74072ef63c05bedd30-Paper-Conference.pdf) introduces Set-Based Prompting as a prior architectural approach to removing order information from selected subsequences.

These papers establish the general phenomenon and prior approaches. They do not prove our 851-flip number; that number comes from this release’s hash-bound controlled baseline.

## What did the original failures look like?

These are verified failures from the original controlled checkpoint. The questions are paraphrased from AI2 ARC Easy; option text is shortened only for presentation. Probabilities are mapped into the same semantic option order in both columns.

### Failure 1: reversing the list changed one wrong answer into another

**ARC Easy item `Mercury_7044555`** asks which choice is *not* a weather instrument.

| Semantic option | Original-order probability | Reversed-order probability |
|---|---:|---:|
| barometer | 0.2306 | **0.4299** |
| thermometer | **0.3894** | 0.3695 |
| anemometer | 0.3470 | 0.1830 |
| millimeter — correct | 0.0329 | 0.0176 |

- Original presentation: barometer, thermometer, anemometer, millimeter → **thermometer**
- Reversed presentation: millimeter, anemometer, thermometer, barometer → **barometer**

The correct answer did not move semantically, but the model’s selected meaning changed from thermometer to barometer.

### Failure 2: reversing the list changed a correct answer into a wrong answer

**ARC Easy item `NYSEDREGENTS_2015_8_32`** asks which device directly converts chemical energy to electrical energy.

| Semantic option | Original-order probability | Reversed-order probability |
|---|---:|---:|
| solar-powered calculator | 0.1224 | **0.5615** |
| wood-burning stove | 0.1229 | 0.0967 |
| battery-powered flashlight — correct | **0.5589** | 0.2873 |
| wind-powered sailboat | 0.1958 | 0.0545 |

- Original presentation: calculator, stove, flashlight, sailboat → **battery-powered flashlight (correct)**
- Reversed presentation: sailboat, flashlight, stove, calculator → **solar-powered calculator (wrong)**

Nothing about the science question or available answers changed. Only the list order changed.

**Source and attribution:** AI2 Reasoning Challenge, ARC Easy, items `Mercury_7044555` and `NYSEDREGENTS_2015_8_32`, dataset revision `210d026faf`. ARC is described by Clark et al. in [Think you have Solved Question Answering? Try ARC](https://arxiv.org/abs/1803.05457) and distributed as [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) under CC BY-SA 4.0. The examples above are paraphrased and reformatted; item IDs are retained for traceability.

## What changed for the caller?

The caller sends **one complete request containing all options, once**. The user does not manually submit options. Internally, the runtime renders and scores one full prompt per option.

```text
one request with all options
            │
            ├── full prompt for option A ─┐
            ├── full prompt for option B ─┼── score each option ── choose mode ── answer
            └── full prompt for option C ─┘
```

Each branch contains the same state and question, one complete option, no siblings, and no generated A/B/C label. Options start at the same logical token position and each receives a complete batch-one forward pass. A fitted replacement `PointerHead` emits one raw score (logit) per option; softmax follows only after all logits exist. Exact ties use semantic identity, not display position.

## A measured example from the released model

This project-authored example was run against the exact `v0.1.0` Hugging Face snapshot. It is not a benchmark item.

**Question:** Which provider best balances reliability and cost?

| Presentation | Choice order |
|---|---|
| Original | Provider A ($100 per month, 99.9% uptime), Provider B ($80 per month, 99.5% uptime), Provider C ($90 per month, 99.9% uptime) |
| Reversed | Provider C ($90 per month, 99.9% uptime), Provider B ($80 per month, 99.5% uptime), Provider A ($100 per month, 99.9% uptime) |

Both modes returned the same semantic answer and probabilities after reversal:

| Mode | Original result | Reversed result | Semantic probabilities A / B / C |
|---|---|---|---:|
| `normal` | Provider B | Provider B | 0.2890 / 0.4049 / 0.3061 |
| `compare_choices` | Provider B | Provider B | 0.2889 / 0.4059 / 0.3051 |

The caller submitted the complete option set each time. The runtime—not the user—handled isolated branch construction and semantic remapping.

## Why are there two modes?

Independent scoring removes display position, but some decisions are relational: “meets policy” can be judged independently; “second-lowest cost” cannot.

| Mode | Use it when |
|---|---|
| `normal` (default) | Choices can be judged independently—for classification, routing, or semantic-label selection. This is the locked retrofit’s default behavior. |
| `compare_choices` | The answer depends on highest, lowest, second, closest, unique, dominant, or other relationships, and the relational residual is explicitly wanted. |

`compare_choices` begins with unchanged `normal` logits and branch features, then adds a position-free set residual: a content-and-relationship correction without slot or first/last-choice features.

## What did the retrofit achieve?

| Measurement | Original controlled E2B | `normal` | `compare_choices` |
|---|---:|---:|---:|
| Standard accuracy, 3,900 questions | **73.00%** (2,847/3,900) | 72.49% (2,827/3,900) | 72.46% (2,826/3,900) |
| Typed accuracy, 1,200 questions | 62.08% (745/1,200) | **62.83%** (754/1,200) | **62.83%** (754/1,200) |
| JEVBench, frozen prospective protocol | 178/231 | **181/231** | **181/231** |
| Answer changes after reversal | **21.82%** | **0/3,900** | **0/3,900** |
| Valid output schemas | — | 100% | 5,331/5,331 |

Against the original controlled checkpoint, standard accuracy fell about half a point: 0.51 for `normal` and 0.54 for `compare_choices`. Typed accuracy rose 0.75 points, JEVBench gained three correct items, and no reversal flips appeared on the supported path.

Permutation checks passed **100,028/100,028**: exhaustive for 821 items with at most five choices, and deterministic balanced/cyclic plans for 1,179 wider items (widths 4–8). Maximum mapped-logit error was `4.76837158203125e-07`; disabling the relational residual preserved `normal` logits byte-for-byte.

## Does comparing choices actually help?

On a locked test of 2,000 synthetic comparison problems across six task families and four to eight choices, `normal` scored **33.55%** (671/2,000; raw NLL 1.5853), while `compare_choices` scored **37.10%** (742/2,000; raw NLL 1.5389).

That is a paired gain of **+3.55 percentage points**: the relational correction recovered matched-test answers isolated scoring missed. Its paired-bootstrap 95% confidence interval was **[+2.20, +4.90] points**, excluding zero.

This gain is specific to the relational test: `compare_choices` was slightly lower on standard accuracy and identical on typed and JEVBench.

## How do I load and call it?

Install and load the released model with this minimal complete pattern:

```bash
pip install "strands-gemma-order-stable-hybrid @ git+https://github.com/manu-mishra/strands-gemma-order-stable-hybrid.git"
```

```python
from huggingface_hub import snapshot_download
from strands_gemma_order_stable_hybrid import load_hybrid
from strands_decider.schema import SystemOneRequest

path = snapshot_download(
    "manu-mishra/strands-gemma-order-stable-hybrid",
    revision="v0.1.0",
)
hybrid = load_hybrid(path, device="cuda", attention_implementation="eager")

request = SystemOneRequest.model_validate({
    "state": "Select the plan that meets the stated policy.",
    "questions": {"plan": {
        "type": "choice",
        "instructions": "Which plan should be selected?",
        "criteria": {
            "Bronze": "Complete description of Bronze",
            "Silver": "Complete description of Silver",
            "Gold": "Complete description of Gold",
        },
    }},
})
result = hybrid.engine.evaluate(request, mode="normal")
# Use mode="compare_choices" when the decision is relational.
```

The supported contract is exact: **serialized one-request-at-a-time, batch-one, eager attention, full recomputation**. It means one complete forward pass per option branch, no shared-prefix key/value cache, and no batching of unrelated requests.

## What does this not promise?

This is not a claim of universal invariance across numerical backends. Changes to batching, caching, precision, kernels, device, or attention implementation require new conformance evidence. Exact ties are deterministic by semantic identity, but near ties may remain sensitive to numerical changes.

Probabilities are raw `T=1` outputs; no calibration claim is made. Images are unsupported. The loopback server is an evaluation/reference boundary, not production infrastructure. Evaluation covers finite named datasets and does not establish safety, fairness, factuality, robustness, or suitability for high-impact decisions.

## Technical appendix

<details>
<summary><strong>Architecture, equations, and training</strong></summary>

For option feature `xᵢ`, question summary `q`, normal logit `bᵢ`, and rank `r`:

```text
zᵢ = tanh(Pxᵢ)
u = tanh(Qq)
aᵢⱼ = softmax(zᵢ · zⱼ / √r)
cᵢ = Σⱼ aᵢⱼ zⱼ
m = meanⱼ(zⱼ)
δᵢ = outᵀ tanh(W[zᵢ, cᵢ, zᵢ − m, zᵢ ⊙ m, u] + bias)
final_logitᵢ = bᵢ + δᵢ
```

Projections and readouts are tied across options; mean pooling and content attention are symmetric. The residual has no position IDs, slot embeddings, or first/last-choice features. Hidden width is 1,536, rank 128, and MLP width 64.

Conversion had two bounded stages. First, the Gemma torso and inherited Strands adapter were frozen while isolated-option features were extracted and a replacement pointer head was fitted. Second, the complete `normal` component was locked and only five relational tensors—`P`, `Q`, `W`, `bias`, and `out`—were trained using train/development roles. The relational candidate was locked before opening the 2,000-row test; neither component was retuned after frozen final evaluations.

</details>

<details>
<summary><strong>Rejected backends and measured performance</strong></summary>

A shared-prefix BF16 cache failed: 8 of 14 required cases exceeded the frozen raw-logit tolerance, with maximum difference `0.0716`. FP32 diagnostics were much closer but did not validate the tested BF16 path.

Arbitrary BF16 cross-request batching also failed. Solo permutations had zero flips in 186 checks; pairing requests with unrelated short padding companions caused 26 semantic flips in 744 placements, with maximum aligned-logit change `0.1034`. The reference server therefore serializes complete requests.

On an NVIDIA L4 with eager attention, two warmups, and 20 measurements per cell:

| Options | `normal` | `compare_choices` |
|---:|---:|---:|
| 2 | 166.8 ms | 167.9 ms |
| 4 | 330.1 ms | 330.1 ms |
| 8 | 653.6 ms | 659.5 ms |
| 16 | 1,320.2 ms | 1,321.6 ms |

Peak allocated GPU memory was approximately 10.0 GB in both modes. Latency grows with option count because every option is fully recomputed.

</details>

<details>
<summary><strong>Hashes, evidence, and licensing</strong></summary>

Model identities: Gemma base `google/gemma-4-E2B-it` at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`; Strands checkpoint `StrandsAgents/strands-decider-E2B-gemma4-v1-2610` at `d383ecf8534b13d49b3b567a1b66f88fca6d50ba`; Strands source `strands-labs/strands-decider` at `0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534`.

Artifact SHA-256 values: portable manifest `faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974`; replacement head `781f0f460975316cac306ae8bb7fa11dd1471d3e8b183140ac8c0aaf1a02e915`; relational weights `afe53a76352e389b8aba2d92599742e20770e9a374baafb401c7a283ce22fca3`.

The sanitized aggregate [evidence file](results/hf-evidence.json) is schema version 2. It records the exact **851/3,900** reversal numerator, all 13 per-task flip counts, treatment definition, semantic-alignment rule, original and retrofit accuracy counts, and protocol/source/report hashes. It also records the two attributed, paraphrased failure examples and the project-authored success sample shown above; the complete baseline rows remain omitted. The companion [evidence note](docs/order-sensitivity-evidence.md) documents the recomputation algorithm, provenance limitations, and related primary research.

Evidence SHA-256 values: frozen requirements `f77270c9b41bcda06b55080a7365f1f22e96bd43974e1aa41aef102cfac0d79d`; authority `c28621d7c86cb394359194b02e02c65932187a952e7d6a0ac593044f4c85cbeb`; accepted normal report `3fbbbd1f264c6fb296fa74caa4c36ea3b4e40a405e54824861e6280cea0234e2`; terminal hybrid report `0889747869f2f07d157370072a578db6aaa6f948acad0f317470fb176102ca30`.

Runtime source and tests are in the [GitHub repository](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid). This card contains the public architecture, evaluation summary, execution boundary, and usage contract. Full item-level prompt, label, and response files are omitted; only the two attributed and paraphrased failure examples shown above are included.

Strands-derived source is Apache-2.0. Separately authored portions retain an MIT notice; see `LICENSE`, `LICENSE-MIT`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.

</details>
