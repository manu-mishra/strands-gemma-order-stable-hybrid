# Strands Gemma Order-Stable Hybrid

**This retrofit is order-stable only under serialized, one-request-at-a-time, batch-one full recomputation.** Each option is encoded independently at the same logical position. Shared-prefix caching and arbitrary cross-request batching are not supported: both failed frozen numerical-conformance gates.

This released source bundle documents a two-mode retrofit of the released Strands Gemma E2B decision checkpoint. `normal` is the default independent scorer and preserves the locked normal component. `compare_choices` adds a tied, position-free relational residual when the supplied choices need to be compared as a set. The result is a conditional-set decision model, not a claim of universal order invariance or a general quality improvement.

> GitHub hosts the released source, runtime, documentation, tests, tooling, and sanitized evidence. Hugging Face hosts the released model components. Bulk raw datasets and complete item-level result files are omitted; the model page includes two attributed, paraphrased ARC Easy failure examples.

## Evidence at a glance

| Measure | `normal` | `compare_choices` |
|---|---:|---:|
| 13-task standard macro accuracy (3,900 rows) | 72.487179% | 72.461538% |
| Typed macro accuracy (1,200 rows) | 62.833333% | 62.833333% |
| JEVBench | 181/231 (78.354978%) | 181/231 (78.354978%) |
| Standard reversal flips | 0/3,900 | 0/3,900 |
| Schema errors | 0 | 0/5,331 |

On the separately locked 2,000-row relational test, `compare_choices` improved accuracy from 33.55% to 37.10%: **+3.55 percentage points**, with a paired percentile-bootstrap 95% confidence interval of **[+2.20, +4.90] points** (20,000 replicates; seed 20261017). All **100,028/100,028** mapped permutation checks passed; the maximum aligned-logit error was `4.76837158203125e-07`. This relational result does not imply that `compare_choices` improves every benchmark: on the standard suite it was 0.025641 point below `normal`, while typed and JEVBench results were unchanged.

The machine-readable, aggregate-only record is [`results/results.json`](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/blob/v0.1.0/results/results.json). The exact 851-flip recomputation, per-task counts, semantic-alignment method, evidence hashes, and primary literature are documented in [`docs/order-sensitivity-evidence.md`](docs/order-sensitivity-evidence.md).

## Released model snapshot

The model payload is released on Hugging Face:

- Repository: [`manu-mishra/strands-gemma-order-stable-hybrid`](https://huggingface.co/manu-mishra/strands-gemma-order-stable-hybrid)
- Release tag: `v0.1.0`
- Portable manifest: `faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974`

Download the release tag rather than floating `main`:

```python
from huggingface_hub import snapshot_download
from strands_gemma_order_stable_hybrid import load_hybrid

path = snapshot_download(
    "manu-mishra/strands-gemma-order-stable-hybrid",
    revision="v0.1.0",
)
hybrid = load_hybrid(path, device="cuda", attention_implementation="eager")
```

The tagged snapshot has been checksum-verified and loaded offline in both modes.

## Pinned upstream identities

Reproduction must use immutable identities, not floating branches:

- Strands checkpoint: [`StrandsAgents/strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/tree/d383ecf8534b13d49b3b567a1b66f88fca6d50ba), revision `d383ecf8534b13d49b3b567a1b66f88fca6d50ba`
- Strands source: [`strands-labs/strands-decider`](https://github.com/strands-labs/strands-decider/tree/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534), commit `0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534`
- Gemma base: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it/tree/3e22461f65e89153144f8adb70e3b8c2cc9845a7), revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`

The Strands checkpoint does not contain the Gemma base weights, and its adapter configuration does not pin the base revision. The separate Gemma pin above is therefore part of the identity.

## Artifact identity and trust chain

Two SHA-256 values identify different stages and must not be interchanged:

- Historical hybrid descriptor: `47748ebb2aca6df06dac46c4a91d075726cf96ef3d4d3c3319a6225bd8550926`.
- Deterministic portable `hybrid-manifest.json`: `faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974`.

During preparation, the tool authenticates the complete historical normal and relational source inventories and their locks, validates selected-candidate and upstream identities, copies only allowlisted members, canonicalizes the residual configuration, and emits sorted compact JSON with relative paths. The historical source lock files are preparation inputs and are omitted from the portable output; their digests and source-member hashes remain embedded in the portable manifest as provenance. At runtime, validation checks the exact portable manifest structure and every admitted member, and `portable_manifest_sha256` in HTTP responses is the portable manifest digest (`faa52ee…`), not the historical descriptor digest (`47748ebb…`). This provides a hash-bound preparation lineage without claiming that runtime reopens the omitted historical files.

## Modes

### `normal` — default

`normal` renders one branch per semantic option, performs one complete batch-one torso encoding per option with no key/value cache, and softmaxes the option logits. Its project-fitted replacement pointer head is stored in the normal component's `head.safetensors` and loaded through the pinned Strands `PointerHead` implementation. The Gemma torso and inherited Strands adapter remain frozen; the selected normal head is not the checkpoint's inherited/original decision-head weights. This mode is appropriate when options should be judged independently against the shared state and question.

When `normal` is selected, the hybrid wrapper delegates directly to the locked normal engine. It does not extract relational features or read, cast, clone, or add residual parameters. This disabled-arm boundary was verified byte-for-byte at the logit output.

### `compare_choices` — opt in

`compare_choices` first obtains the same base logits and per-option pointer features, then applies a small tied set residual. Shared transforms, masked symmetric pooling, and content attention provide no slot or position input. Permuting options therefore permutes output rows under the supported backend contract.

Use this mode when the decision depends on relations among the supplied alternatives. It is not the default and should not be described as a blanket upgrade over `normal`.

See [`docs/architecture.md`](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/blob/v0.1.0/docs/architecture.md) for equations, control flow, and the exact backend boundary.

## Install and load the released model

Install the runtime from the GitHub release tag, then download the model components from the matching Hugging Face release tag:

```bash
pip install "strands-gemma-order-stable-hybrid @ git+https://github.com/manu-mishra/strands-gemma-order-stable-hybrid.git@v0.1.0"

MODEL_PATH="$(
python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    "manu-mishra/strands-gemma-order-stable-hybrid",
    revision="v0.1.0",
))
PY
)"

strands-gemma-order-stable-hybrid validate "$MODEL_PATH"
```

The portable manifest uses relative paths and SHA-256 hashes, so the downloaded directory can move between machines without retaining checkout-specific paths. Package metadata supports Python 3.10 or later and Transformers `==5.18.0` (the 5.18.x line), together with the bounded PyTorch, PEFT, Accelerate, Pydantic, safetensors, and Hugging Face Hub dependencies. The accepted measurements used Python 3.12.3, Torch 2.14.1+cu130, CUDA on an NVIDIA L4, eager attention, and eight CPU threads.

Start the serialized loopback reference server:

```bash
strands-gemma-order-stable-hybrid serve \
  "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda \
  --attention-implementation eager
```

The server deliberately binds only to `127.0.0.1`, loads the model once, handles requests serially, and accepts bodies up to 16 MiB. It supplies no authentication, TLS, multiworker, or public-network contract.

## Build the Hugging Face release payload

For deterministic release construction from exact normal and relational component directories, prepare and validate the portable artifact, then emit the Hub upload folder:

```bash
uv sync --extra dev

uv run strands-gemma-order-stable-hybrid prepare-artifact \
  --normal "<normal-component>" \
  --relational "<relational-component>" \
  --output .artifacts/hybrid-v1

uv run strands-gemma-order-stable-hybrid validate .artifacts/hybrid-v1

uv run python tools/build_hf_release.py \
  --artifact .artifacts/hybrid-v1 \
  --output .artifacts/hf-release

(cd .artifacts/hf-release && sha256sum -c SHA256SUMS)
```

The build tool emits the exact Hugging Face release payload: the two authenticated model components, portable manifest, Hub model card, sanitized evidence, notices, and checksum inventory. Architecture and narrative are embedded in the Hub model card. Historical machine-bound locks, bulk raw datasets, complete item-level result files, and unsafe weight formats are omitted; the two attributed paraphrased examples are explicitly included.

## HTTP API

Health and identity information:

```bash
curl http://127.0.0.1:8000/health
```

Score a standard Strands `SystemOneRequest` in the default mode:

```bash
curl -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  --data @request.json
```

Opt into relational comparison by adding `"mode": "compare_choices"` to the top-level request object, or by using `?mode=compare_choices`. If body and query specify different modes, the server rejects the request. Valid modes are exactly `normal` and `compare_choices`. Invalid payloads return 422; unknown routes return 404. `/health` returns status, engine kind, modes, default mode, `portable_manifest_sha256`, both selected candidate IDs, all historical lock hashes, and pinned upstream identities. Inference returns the validated Strands response plus mode, `portable_manifest_sha256`, the selected candidate ID for that mode, historical lock hashes, and latency.

Requests may contain Strands `choice`, `noul`, and `score` questions. Images are rejected. Exact ties are resolved by semantic identity rather than encounter order.

## Metrics and performance

The final quality gates used 3,900 standard examples across 13 tasks, 1,200 typed examples across four tasks, 231 JEVBench examples, and a distinct 2,000-row relational test. Raw selected-head probabilities use `T=1`; no probability-calibration claim is made.

On the recorded L4 eager/full-recomputation run, mean latency for `normal` versus `compare_choices` was 166.803/167.899 ms at 2 options, 330.060/330.104 ms at 4, 653.643/659.543 ms at 8, and 1320.219/1321.585 ms at 16. Each cell used 20 measurements after two warmups. Peak allocated GPU memory was 10,002,322,944 bytes for `normal` and 10,002,384,384 bytes for `compare_choices`. The artifacts share the complete recorded environment, but same-host identity was procedural rather than cryptographically proven because the evidence schema did not record a host identifier.

## Limitations

- **Narrow execution contract.** The demonstrated order stability applies to serialized, one-request-at-a-time, batch-one full recomputation with eager attention. Other devices, dtypes, kernels, attention implementations, or serving stacks require new conformance evidence.
- **No prefix cache.** Shared-prefix BF16 caching exceeded the frozen raw-logit tolerance in 8 of 14 required cases; maximum delta was `0.07161378860473633`. Measured speedups were rejected.
- **No arbitrary cross-request batching.** BF16 padding companions caused 26 semantic flips across 744 placements, with maximum aligned-logit delta `0.10336923599243164`.
- **No universal invariance claim.** Passing option-permutation checks does not establish invariance to paraphrase, prompt-template changes, context truncation, numerical backend changes, or unrelated request colocation.
- **Task coverage is limited.** Accuracy evidence is aggregate evidence on the named suites, not a safety, fairness, robustness, or domain-readiness certification.
- **No calibrated probabilities.** Brier score and raw `T=1` NLL are measurements, not a calibration guarantee.
- **Loopback reference server only.** The HTTP boundary is not hardened production infrastructure.

## Release locations

The scientific terminal disposition is `HYBRID PASS` for the frozen gates. GitHub release tag `v0.1.0` hosts the complete source distribution, runtime, documentation, tests, tooling, notices, and sanitized evidence. Hugging Face release tag `v0.1.0` hosts the normal and relational model components. The separately hosted Gemma base remains pinned to its upstream revision. Training data, bulk raw evaluation data, and complete item-level result files are not published; two attributed paraphrased ARC Easy failures appear on the model page.

## Distribution scope

The wheel is deliberately runtime-only: it contains the runtime package, entry point, metadata, and license/notice files, but no weights. The source distribution (sdist) is the complete released source package: it additionally contains this README, `blog.md`, `MODEL_CARD.md`, architecture and evidence files, preparation tooling, tests, and `uv.lock`. Model components are released on Hugging Face. Neither package includes bulk raw datasets or complete item-level result files; the model page and evidence file include only the two attributed paraphrased ARC Easy failures.

## Documentation

- [Experiment narrative](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/blob/v0.1.0/blog.md)
- [Model card](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/blob/v0.1.0/MODEL_CARD.md)
- [Architecture and backend boundary](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/blob/v0.1.0/docs/architecture.md)
- [Aggregate results](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/blob/v0.1.0/results/results.json)
- [Apache-2.0 license](LICENSE)
- [MIT license for project-authored portions](LICENSE-MIT)
- [Notice](NOTICE)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Licensing

Strands-derived portions are distributed under Apache-2.0; see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Separately authored project portions are available under the MIT terms in [`LICENSE-MIT`](LICENSE-MIT). Model and data terms are component-specific; source-code licenses do not grant rights to model components or datasets beyond their applicable terms. Gemma references: [license](https://ai.google.dev/gemma/apache_2), [model card](https://ai.google.dev/gemma/docs/core/model_card_4), and [terms](https://ai.google.dev/gemma/terms).
