# Architecture and backend boundary

## Scope

The hybrid wraps a pinned Strands Gemma E2B checkpoint with two typed decision modes:

- `normal`: independent isolated-branch scoring; default.
- `compare_choices`: the same base scoring plus a tied relational residual over the active option set.

The demonstrated order-stability contract is serialized, one-request-at-a-time, batch-one full recomputation with eager attention. The architecture is permutation-equivariant with respect to option rows; backend conformance is a separate empirical requirement.

## Symbols

For a question with active semantic options indexed by `i = 1, …, K`:

- `s`: rendered shared state and question prefix
- `o_i`: full text of semantic option `i`
- `r_i = render(s, o_i)`: isolated branch
- `H_i`: final hidden states from Gemma for branch `i`
- `d_i`: pooled final-token answer vector for branch `i`
- `x_i`: hidden vector gathered at the equal logical option-start position
- `q(·)`, `k(·)`: selected replacement pointer-head projections
- `b_i`: normal base logit
- `M`: active-option mask
- `r`: residual projection rank
- `ℓ_i`: final logit
- `p_i`: probability over active semantic options

## Rendering invariants

The branch renderer first obtains the Strands prompt structure, then extracts a common prefix and final suffix. It constructs one branch per semantic option with the following invariants:

1. Every branch has identical bytes and tokens before its option.
2. Every option begins at the same logical token position.
3. A branch contains its own complete option and no sibling option.
4. No numeric, alphabetic, or encounter-order label is generated.
5. Score levels keep semantic identities for readback, but only descriptions enter the branch.
6. If context fitting is necessary, complete options and equal starts are preserved; strict-window mode raises instead of silently violating the contract.

Duplicate display strings may remain distinct because mapping is by unique semantic identity, not text equality or row position.

## `normal` control flow

For each question, serially:

1. Validate the typed question and reject unsupported images.
2. Render `K` isolated branches and verify equal prefix and option-start invariants.
3. For each branch `i`, create one complete state-plus-branch row.
4. Call the Gemma encoder once with batch size one and no `past_key_values`.
5. Extract `d_i` and `x_i` from the same forward pass.
6. Apply the frozen project-fitted replacement pointer readout from the normal component's `head.safetensors`, loaded through the pinned Strands `PointerHead` implementation, to obtain `b_i`. These selected head weights are not the checkpoint's inherited/original decision-head weights; the Gemma torso and inherited Strands adapter stay frozen.
7. Stack logits in semantic-option order, apply the existing question-kind temperature, and softmax over active options.
8. Convert probabilities into the Strands `choice`, `noul`, or `score` response schema.

The base equations are

```text
H_i = Gemma(r_i)
d_i = pool_last_token(H_i)
x_i = gather_at_option_start(H_i)
b_i = <q(d_i), k(x_i)> / sqrt(pointer_dim)
p_i = softmax_i(b / T)
```

The final evidence reports raw selected-head `T = 1` probabilities and makes no calibration claim.

For `choice`, all semantic probabilities are returned. For `noul`, the readback is the probability of true. For `score`, the readback is the expected ordinal level. Exact argmax ties are resolved by a canonical semantic-identity rule, never by encounter order.

### Exact normal bypass

At the hybrid boundary, `resolve_mode(None)` selects `normal`. In this mode the wrapper delegates directly to the locked normal engine. It does not request relational features and does not read, validate, cast, clone, or add residual parameters. This is stronger than a zero-valued residual: it preserves the disabled path’s base logits byte-for-byte.

## `compare_choices` control flow

The relational path begins with the normal feature extraction:

1. Run the same isolated branch passes and obtain base logits `b_i`, answer vectors `d_i`, and option vectors `x_i`.
2. Compute the permutation-invariant question summary

```text
q̄ = (1 / |M|) Σ_{i∈M} d_i
```

3. Apply the tied set residual to active option rows.
4. Add each residual to the corresponding base logit.
5. Mask inactive rows and softmax over active options.
6. Use the same semantic readback as `normal`.

The residual has no slot embedding, row number, display position, or order-dependent feature.

## Relational equations

Let `P` and `Q` be learned projections into rank `r`:

```text
z_i = tanh(P x_i)
u   = tanh(Q q̄)
```

Content attention is computed only over active options:

```text
e_ij = (z_i · z_j) / sqrt(r)
a_ij = exp(e_ij) / Σ_{k∈M} exp(e_ik),  j ∈ M
c_i  = Σ_{j∈M} a_ij z_j
```

A symmetric set mean is

```text
m = (1 / |M|) Σ_{j∈M} z_j
```

Each option receives a shared relational feature vector:

```text
h_i = concat(z_i, c_i, z_i - m, z_i ⊙ m, u)
δ_i = w_outᵀ tanh(W h_i + b)
ℓ_i = b_i + δ_i
p_i = softmax_i(ℓ over M)
```

`P`, `Q`, `W`, `b`, and `w_out` are the only trainable tensors. The transforms are shared across options. `w_out` is initialized to exact zero, so the initial enabled residual reproduces the base logits. Padding is excluded from attention and mean pooling and is masked to negative infinity before softmax. Every item must have at least two finite active options.

## Why the residual is permutation-equivariant

Let `Π` be a permutation matrix over active option rows. The row-wise shared projections satisfy `P(ΠX) = ΠP(X)`. The active mean is invariant, and dot-product attention without positional terms transforms as

```text
A(ΠZ) = Π A(Z) Πᵀ.
```

Therefore contextual rows transform as `C(ΠZ) = ΠC(Z)`. Every term in `h_i` is either a permuted row-wise quantity or a broadcast invariant summary. The shared output map then gives

```text
δ(ΠX) = Πδ(X)
ℓ(ΠX) = Πℓ(X).
```

Mapping the permuted logits back by semantic identity should recover the original vector up to floating-point tolerance. The locked test observed 100,028 passes in 100,028 checks, with maximum mapped-logit absolute error `4.76837158203125e-07`.

This proof concerns the set-residual computation given equivalent input features. It does not prove that every model backend or batching scheme generates equivalent branch features.

## Artifact and load boundary

The runtime consumes a deterministic portable artifact, not the historical descriptor file. Two identities apply to different stages:

- Historical hybrid descriptor SHA-256: `47748ebb2aca6df06dac46c4a91d075726cf96ef3d4d3c3319a6225bd8550926`.
- Portable `hybrid-manifest.json` SHA-256: `faa52ee416545296fabafa4c0754be98216817f4efeda3e3fb64b4ea7f481974`.

Preparation first authenticates byte-exact historical normal and relational source inventories, including historical lock files, and validates the selected-candidate and upstream identity fields inside those locks. It then copies only allowlisted members, emits a canonical residual configuration, and writes sorted compact manifest JSON with relative paths. Historical lock files are omitted from the portable directory, but their digests and source-member hashes are embedded in the manifest.

Runtime validation requires exact manifest structure, rejects unsafe paths and links, hashes every admitted portable member, verifies the canonical residual configuration, and hashes the manifest itself. It does not reopen the omitted historical lock files. The portable manifest's embedded provenance is the one-way link to the preparation inputs. After validation, the loader:

1. Loads the normal candidate exactly once.
2. Loads the relational residual exactly once in evaluation mode with gradients disabled.
3. Verifies feature-dimension compatibility.
4. Exposes exactly `normal` and `compare_choices`, with `normal` as default.

Unknown mode names fail closed. The selected component identities are `wd-0.1-epochs-25-seed-43` and `rank-128-hidden-64-seed-17`. The model components are released on Hugging Face and the runtime loads them from the locally downloaded snapshot.

## HTTP control flow

The reference server uses Python’s single-threaded `HTTPServer` and is forced to loopback `127.0.0.1`.

```text
GET /health
  → status, engine kind, mode list, default mode, portable_manifest_sha256
  → both selected candidate IDs, historical lock hashes, pinned upstream identities

POST /v1/systemone[?mode=...]
  → enforce body size ≤ 16 MiB
  → parse JSON object
  → reconcile optional body/query mode
  → validate SystemOneRequest
  → serialized engine.evaluate(request, mode)
  → validate SystemOneResponse
  → append mode, portable_manifest_sha256, selected candidate ID
  → append historical lock hashes and latency
  → JSON response
```

In both responses, `portable_manifest_sha256` is precisely the deterministic portable-manifest digest `faa52ee…`. Health also exposes both selected candidate IDs, all historical lock hashes, and the pinned upstream identities. Inference exposes the mode-specific selected candidate ID and historical lock hashes.

Multiple query-mode values, conflicting body/query modes, and invalid schemas return 422. Unknown routes return 404. The server has no authentication, TLS, multiworker, public bind, or cross-request batching contract.

## Accepted backend boundary

### Supported

- One request at a time
- One branch per model invocation
- Batch size one
- Full recomputation for every option
- No key/value prefix cache
- Eager attention in the recorded CUDA/L4 evaluation
- Semantic remapping after option permutations

### Rejected: shared-prefix BF16 caching

Frozen tolerances were 0.03 for raw logits, 0.02 for probabilities, and `1e-6` for mapped permutation error. Eight of 14 required cache cases failed raw-logit parity; maximum delta was `0.07161378860473633`. Although probability and permutation checks passed and some speedups were large, the raw-logit gate was mandatory. Cache execution is unsupported.

FP32 cache comparisons were much closer (approximately `2.4e-06`), but that diagnostic does not clear the BF16 runtime. Reduced-precision mitigations also failed at least one required case.

### Rejected: arbitrary BF16 cross-request batching

Independent solo permutations produced 0 flips in 186 checks. Adding arbitrary padding companions produced 26 semantic flips in 744 placements, maximum aligned-logit delta `0.10336923599243164`, and maximum probability delta `0.01935756206512451`. Consequently, unrelated requests must not be colocated in one batch under the documented contract.

## Extension checklist

Any proposed backend expansion should be treated as a new conformance target. At minimum:

1. Freeze acceptance thresholds before examining final cases.
2. Compare aligned raw logits, probabilities, and semantic decisions.
3. Include near-ties, variable option counts, duplicate display text, long contexts, and padding companions.
4. Test all option placements and map outputs by semantic identity.
5. Record exact source, checkpoint, base, dtype, device, kernel, attention, compiler, and serving identities.
6. Reject the optimization if any mandatory gate fails; do not substitute end-task accuracy for numerical conformance.

## Audit and distribution boundary

Package metadata supports Python 3.10 or later and pins Transformers `==5.18.0`. The wheel is runtime-only: runtime modules, entry point, metadata, and license/notice files, with no weights. The source distribution is the complete released source package and additionally includes the README, model card, blog, this architecture document, aggregate results, preparation tool, tests, and `uv.lock`. Model components are released separately on Hugging Face. Raw datasets and item-level evaluation material are not included in either package.

The sanitized [`results/results.json`](../results/results.json) records task-level normal/compare counts, typed-by-kind counts, paired relational outcomes, relational task counts, per-width conformance totals, selection split/candidate counts, all 13 gate statuses, and SHA-256 bindings for frozen protocols, selected evidence, sources, and the terminal audit. These provide recomputable aggregate arithmetic and traceability without publishing rows, prompts, labels, or responses.

## Source boundaries

The Apache-2.0 Strands-derived normal-path adaptation is distributed as:

- `src/strands_gemma_order_stable_hybrid/option_branches.py`
- `tests/test_option_branches.py`

The first file adapts the accepted isolated-option retrofit and composes with the pinned Strands engine, rendering, schema, model, and `PointerHead` interfaces. The test covers that adaptation. The pinned upstream commit did not contain this exact package path.

Separately authored project files offered under MIT terms are:

- `src/strands_gemma_order_stable_hybrid/hybrid.py`
- `src/strands_gemma_order_stable_hybrid/server.py`
- `src/strands_gemma_order_stable_hybrid/set_residual.py`
- `src/strands_gemma_order_stable_hybrid/artifacts.py`
- `tests/test_hybrid_artifacts.py`
- `tests/test_server.py`
- `tests/test_set_residual.py`

These files adapt or package the completed project experiment and portable-artifact boundary. Their MIT terms do not replace Apache-2.0 obligations for imported or composed Strands material. There is no distributed evaluation benchmark path claimed here. See [`../NOTICE`](../NOTICE), [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), [`../LICENSE`](../LICENSE), and [`../LICENSE-MIT`](../LICENSE-MIT) for the licensing split.
