# Model card: Strands Gemma Order-Stable Hybrid

## Summary

This is a two-mode retrofit of a pinned Strands Gemma E2B decision checkpoint, distributed as a deterministic portable artifact prepared from authenticated historical component inputs. `normal` independently scores isolated option branches and is the default. `compare_choices` adds a small tied relational residual over the same base logits and projected option features.

The demonstrated order-stability contract is **serialized, one-request-at-a-time, batch-one full recomputation with eager attention**. Shared-prefix BF16 caching and arbitrary BF16 cross-request batching are explicitly unsupported because they failed frozen conformance tests.

The GitHub release contains the complete source, runtime, documentation, tests, tooling, notices, and sanitized evidence. The released model components are hosted in the Hugging Face model repository. Bulk raw datasets and complete item-level result files are not published; two attributed, paraphrased ARC Easy failures appear on the model page.

## Model details

| Field | Value |
|---|---|
| Model family | Typed conditional-set decision scorer |
| Modes | `normal` (default), `compare_choices` |
| Scientific disposition | `HYBRID PASS` on frozen gates |
| Artifact form | Deterministic portable directory with canonical `hybrid-manifest.json` |
| Normal candidate | `wd-0.1-epochs-25-seed-43` |
| Relational candidate | `rank-128-hidden-64-seed-17` |
| Historical hybrid descriptor SHA-256 | `47748ebb2aca6df06dac46c4a91d075726cf96ef3d4d3c3319a6225bd8550926` |
| Portable manifest SHA-256 | `faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974` |
| Historical normal lock SHA-256 | `a1bee0186af38614372ba2f6e2f058e8ebc94d5b72e0f2d18e87ac36d405cf5a` |
| Historical relational lock SHA-256 | `a1738049034a10890c74b6777ecfd015835fd74c98af71dc1e4c6479f9145b49` |
| Transformers support | `==5.18.0` |
| Normal head SHA-256 | `781f0f460975316cac306ae8bb7fa11dd1471d3e8b183140ac8c0aaf1a02e915` |
| Inherited adapter SHA-256 | `c7786a59354aa190082b1c3aedb7ff3311fe260f070f360c4d3518762691c551` |
| Input kinds | Strands `choice`, `noul`, and `score` questions |
| Images | Unsupported |
| Probability status | Raw selected-head `T=1`; no calibration claim |

### Immutable upstream identities

- Strands checkpoint: [`StrandsAgents/strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/tree/d383ecf8534b13d49b3b567a1b66f88fca6d50ba), revision `d383ecf8534b13d49b3b567a1b66f88fca6d50ba`
- Strands source: [`strands-labs/strands-decider`](https://github.com/strands-labs/strands-decider/tree/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534), commit `0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534`
- Gemma base: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it/tree/3e22461f65e89153144f8adb70e3b8c2cc9845a7), revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`

The adapter metadata does not identify an immutable Gemma base revision, and the Strands checkpoint does not bundle the base weights. The explicit base pin is required for identity.

### Artifact trust chain and runtime identity

The historical descriptor digest and portable manifest digest identify different byte sets. Preparation authenticates complete historical source inventories and the normal/relational lock files, validates their candidate and upstream fields, copies only allowlisted members, canonicalizes the residual configuration, and emits deterministic relative-path JSON. Historical lock files are then omitted; their digests and source-member hashes remain bound in the portable manifest. Runtime validation checks that exact manifest and every admitted portable member. The HTTP field `portable_manifest_sha256` is therefore the portable manifest SHA-256 `faa52ee…`, not historical descriptor SHA-256 `47748ebb…`.

`/health` exposes status, engine kind, modes, default mode, `portable_manifest_sha256`, both selected candidate IDs, historical lock hashes, and pinned upstream identities. Inference exposes the validated typed response plus mode, `portable_manifest_sha256`, the selected candidate ID for that mode, historical lock hashes, and latency. The same identities are recorded in [`results/results.json`](results/results.json).

## Architecture

For each active semantic option, the normal engine renders an isolated branch with a byte/token-identical common prefix and an equal logical option-start position. The branch contains no sibling option and no generated positional label. One complete batch-one forward pass produces an answer representation and option representation. A project-fitted replacement head from the normal component's `head.safetensors`, loaded through the pinned Strands `PointerHead` implementation, maps them to a scalar logit. The Gemma torso and inherited Strands adapter stay frozen; these selected replacement weights are not the checkpoint's inherited/original decision-head weights.

`compare_choices` computes a question summary from branch answer vectors and sends option vectors through tied projections, symmetric masked pooling, and content attention. A shared multilayer readout emits a residual for each option. It has no slot or position input. The residual is added to the normal logit before softmax.

When relational scoring is disabled, the runtime directly returns the original base-logit object before relational validation or parameter access. Only residual tensors were trained; the torso, inherited adapter, and normal head remained frozen. Full equations and control flow are in [`docs/architecture.md`](docs/architecture.md).

## Intended use

### Supported

- Research evaluation of typed decisions whose supplied options should not depend on serialization order.
- `normal` for independent candidate scoring against shared state and question text.
- `compare_choices` when contrasts among the supplied alternatives are part of the intended decision.
- Local, serialized use of the released model components downloaded from Hugging Face.
- Aggregate benchmarking under the documented eager/full-recomputation backend.

### Out of scope

- Bundling the separately hosted Gemma base in this source distribution.
- Public-network serving, multiworker deployment, or untrusted multi-tenant use.
- Shared-prefix caching or arbitrary cross-request batching.
- Images, generative chat, open-ended completion, or tasks outside the typed decision schema.
- Safety-critical, medical, legal, financial, employment, or other high-impact decisions without independent validation and human oversight.
- Treating output probabilities as calibrated confidence.

## Evaluation

### Main suites

| Evaluation | `normal` | `compare_choices` |
|---|---:|---:|
| Standard macro accuracy, 13 tasks / 3,900 rows | 0.7248717948717949 | 0.7246153846153847 |
| Typed macro accuracy, 4 tasks / 1,200 rows | 0.6283333333333333 | 0.6283333333333333 |
| JEVBench, 231 rows | 181/231 | 181/231 |
| Standard mapped reversal flips | 0/3,900 | 0/3,900 |
| Valid output schemas | accepted with 0 errors | 5,331/5,331 |

The `compare_choices` standard result is `-0.0002564102564102` relative to `normal`; typed and JEVBench aggregates are unchanged. These results do not support a general quality-superiority claim for relational scoring.

### Locked relational test

- Rows: 2,000 across six tasks
- Independent accuracy: `0.33550000190734863`
- Relational accuracy: `0.3709999918937683`
- Paired gain: `0.0355` (+3.55 percentage points)
- Paired percentile-bootstrap 95% CI: `[0.022, 0.049]` ([+2.20, +4.90] points)
- Bootstrap: 20,000 replicates, seed 20261017
- Independent raw NLL: `1.585288166999817`
- Relational raw NLL: `1.538940668106079`
- Six-task macro accuracy: `0.33548668026924133` independent; `0.37105219066143036` relational
- Permutation checks: 100,028/100,028 passed
- Exhaustive group: 821 items with at most five options
- Deterministic balanced/cyclic group: 1,179 items with more than five options
- Tested widths: 4, 5, 6, 7, and 8
- Maximum mapped-logit absolute error: `4.76837158203125e-07`
- Disabled residual: base logits unchanged byte-for-byte

Selection was prospective. Candidate families, criteria, seeds, and thresholds were fixed before the locked relational test was opened. Evaluation did not resume selection or retuning after test access. Normal selection compared 18 candidates using disjoint 1,500-row train and 600-row development probes. Relational selection compared nine candidates using only the 12,000-row train and 2,000-row development roles from a 19,000-row source; the 3,000-row calibration role was post-lock descriptive, and the 2,000-row test payload was not deserialized before candidate lock.

The aggregate record supplies task-level numerators/denominators for both modes, typed-by-kind counts, the paired 2×2 relational table, relational task counts, width-stratified permutation totals, all 13 gate statuses, and SHA-256 bindings for frozen protocols, selection records, admitted reports, sources, and the independent terminal audit. These sanitized sufficient statistics permit arithmetic and lineage checks without publishing complete item-level result files. Two attributed, paraphrased ARC Easy failures are included separately to make the observed problem concrete.

### Recorded performance

The accepted benchmark used an NVIDIA L4, CUDA, eager attention, Python 3.12.3, Torch 2.14.1+cu130, eight threads, two warmups, and 20 measurements per cell. Mean latency in milliseconds:

| Options | `normal` | `compare_choices` |
|---:|---:|---:|
| 2 | 166.8028116 | 167.8993888 |
| 4 | 330.06025785 | 330.1039311 |
| 8 | 653.64340445 | 659.54261005 |
| 16 | 1320.2192771 | 1321.58537075 |

Peak GPU allocation was 10,002,322,944 bytes for `normal` and 10,002,384,384 bytes for `compare_choices`; both reserved 10,305,404,928 bytes. The paired artifacts record the same complete environment, but do not cryptographically establish that they ran on one physical host.

## Known limitations and failure evidence

### Prefix caching

Shared-prefix BF16 caching is unsupported. With frozen tolerances of 0.03 raw-logit difference, 0.02 probability difference, and `1e-6` permutation difference, 8 of 14 required cases failed raw-logit parity. The largest delta was `0.07161378860473633`; the initial two-option failure was `0.06663298606872559`. Probability and permutation gates passing did not override the raw-logit failure. Measured cache speedups from 0.970x to 7.102x were rejected.

### Cross-request batching

Arbitrary BF16 cross-request/padding batching is unsupported. Solo permutations produced 0 flips in 186 checks, but batch placements produced 26 flips in 744 checks. Maximum aligned-logit and probability deltas were `0.10336923599243164` and `0.01935756206512451`. The demonstrated contract therefore requires batch one and request serialization.

### General limitations

- Order stability was tested for option permutations, not paraphrases, prompt-template changes, unrelated context edits, or alternate truncation policies.
- Different hardware, dtypes, quantization, compilers, attention kernels, or serving runtimes require new evidence.
- Accuracy results are finite-sample aggregate measurements and can conceal subgroup or item-level failures.
- No fairness, safety, privacy, security, multilingual, adversarial, or domain-specific certification is provided.
- Raw `T=1` probabilities are not asserted to be calibrated.
- Exact ties are deterministic by semantic identity; near-ties can still be numerically fragile.
- Latency scales with option count because every branch is fully recomputed.

## Training and data

The relational residual was selected and trained with the Gemma torso, inherited adapter, and normal pointer head frozen. This release does not publish training or evaluation corpora. The inherited adapter and replacement-head data lineage includes restricted or unresolved sources, so successful evaluation does not grant rights to redistribute those underlying datasets.

No bulk prompts, answer keys, item-level prediction files, or raw benchmark responses are included in the released aggregate record. The two ARC Easy failures shown on the model page are attributed, paraphrased examples with exact mapped probabilities.

## Ethical considerations

An order-stable interface removes one incidental source of variation under a narrow execution contract; it does not make the underlying model unbiased, factual, safe, or suitable for high-impact decisions. Relational scoring can amplify contrasts among options supplied by a caller, including misleading or harmful options. Deployers remain responsible for input design, domain validation, monitoring, access control, and human oversight.

## Availability and release status

The model components are released at [`manu-mishra/strands-gemma-order-stable-hybrid`](https://huggingface.co/manu-mishra/strands-gemma-order-stable-hybrid) under tag `v0.1.0`. The GitHub release hosts the source, runtime, documentation, tests, tooling, notices, and sanitized evidence. Bulk raw datasets and complete item-level result files are omitted; two attributed paraphrased ARC Easy failures are included on the model page.

The wheel is deliberately runtime-only: runtime modules, entry point, metadata, and license/notice files, with no weights. The source distribution is the complete released source package and additionally carries the README, this model card, blog, architecture document, evidence files, preparation tool, tests, and `uv.lock`. The Hugging Face release carries the normal and relational model components. Neither package publishes bulk raw datasets or complete item-level result files; only the two attributed paraphrased ARC Easy failures are included.

Scientific status: all frozen hybrid gates passed. Release status: source, documentation, aggregate evidence, and model components are released. Source-code licensing does not grant rights to underlying model or dataset artifacts beyond their applicable terms.

## Licensing and attribution

Strands-derived source is covered by Apache-2.0; see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Separately authored project portions are covered by [`LICENSE-MIT`](LICENSE-MIT). Referenced Gemma materials have their own source and acquisition terms: [Gemma license](https://ai.google.dev/gemma/apache_2), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), and [Gemma terms](https://ai.google.dev/gemma/terms).
