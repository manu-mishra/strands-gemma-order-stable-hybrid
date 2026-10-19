# The flashlight that became a solar calculator

“Which device directly converts chemical energy to electrical energy?”

The options were a solar-powered calculator, a wood-burning stove, a battery-powered flashlight, and a wind-powered sailboat. In that order, the checkpoint chose the battery-powered flashlight—the correct answer—with probability 0.5589.

Then we reversed the same four options. The checkpoint chose the solar-powered calculator, now with probability 0.5615.

| Semantic option | Original order | Reversed order |
|---|---:|---:|
| solar-powered calculator | 0.1224 | **0.5615** |
| wood-burning stove | 0.1229 | 0.0967 |
| battery-powered flashlight — correct | **0.5589** | 0.2873 |
| wind-powered sailboat | 0.1958 | 0.0545 |

These are semantic probabilities: both columns map scores back to the same option meanings, regardless of where each option appeared. This was not an index-mapping error. The selected meaning changed even though the question and available answers did not.

The example paraphrases and reformats ARC Easy item `NYSEDREGENTS_2015_8_32` at dataset revision `210d026faf`. ARC is the AI2 Reasoning Challenge introduced by Clark et al. in [*Think you have Solved Question Answering? Try ARC*](https://arxiv.org/abs/1803.05457); the dataset is distributed under CC BY-SA 4.0.

That flashlight was not an isolated anecdote. Complete reversal changed the winning semantic option on **851 of 3,900 questions: 21.82%**. The [evidence note](docs/order-sensitivity-evidence.md) defines the treatment and semantic alignment, reports all 13 task counts, and states the limits of the measurement.

The result led to a practical research question: **Can we retrofit a released checkpoint so option order stops driving decisions, without retraining the Gemma torso or inherited adapter and without giving up useful quality?**

The answer was a two-mode scorer. Its default mode isolates options. Its second mode restores controlled comparison between options without adding display position. Under the tested execution path, both modes produced zero winner changes on all 3,900 original-versus-reversed cases. Standard accuracy fell about half a percentage point against the original checkpoint, while typed and JEVBench results improved. On a separate relational test, comparison added 3.55 points over isolated scoring.

## An ordered prompt makes order part of the problem

The original scorer encoded all choices together as one token sequence. That design feels natural because language models consume sequences. It also makes list order a model input.

Move an option and three things change at once: its token position, its label or number, and the neighboring option text that shapes its contextual representation. The model no longer sees “the same option in another display slot.” It sees a different sequence and computes different hidden states.

Prior work shows that this problem extends beyond one checkpoint. Studies have measured [large accuracy changes after reordering multiple-choice options](https://arxiv.org/abs/2308.11483), separated [position and option-label selection bias](https://arxiv.org/abs/2309.03882), and analyzed [order sensitivity alongside token sensitivity](https://aclanthology.org/2024.findings-acl.333/). Small benchmark choices can even [change model rankings](https://aclanthology.org/2024.acl-long.744/). Set-Based Prompting offers a related architectural direction by [removing order information from designated prompt subsequences](https://papers.neurips.cc/paper_files/paper/2024/file/85529bc995777a74072ef63c05bedd30-Paper-Conference.pdf).

Those papers establish the broader failure mode. The 851 flips here come from this checkpoint, this complete-reversal treatment, and this evaluation.

## Give each option the same view

The retrofit’s `normal` mode replaces one joint prompt with isolated option branches. The caller still sends **one request containing every option, once**. The runtime—not the caller—builds and scores the branches.

For a question with four choices, the runtime creates four full prompts. Each prompt contains the same state and question, followed by exactly one complete option. No branch contains a sibling option. No branch receives an A/B/C label or a generated row number. Every option begins at the same logical token position.

The frozen Gemma torso and inherited adapter process each branch separately. The runtime extracts branch features and applies a fitted replacement pointer head to produce one logit per option. It then stacks the logits in semantic-option order and applies softmax across the active choices. Exact ties use semantic identity rather than encounter order.

This arrangement removes display position and sibling context from the base score. A solar calculator gets the same branch whether a UI displays it first or last. Reordering only changes presentation; semantic mapping restores the scores to their option identities.

Isolation has a clear cost. With *K* options, the supported path runs *K* complete batch-one forward passes. It does not reuse a shared prefix. That cost became important later.

## Isolation cannot answer every kind of question

Independent scores work well when each option can stand on its own: classify an intent, route a request, or select the plan that meets a policy.

Some questions are relational by definition. “Which plan has the second-lowest cost?” cannot be answered from one plan in isolation. Neither can “Which value is unique?” or “Which provider is closest to the group mean?” A scorer must compare options to solve those tasks.

The `compare_choices` mode starts with the unchanged isolated logits and features. It then applies a small, position-free residual over the active option set. Shared projections encode each option. Content attention lets options compare with one another. A symmetric mean summarizes the set, and a shared readout produces one correction per option.

The same transforms operate on every row. The residual receives no slot embedding, row number, first-choice flag, or last-choice flag. If the input rows are permuted, the residual rows permute with them; semantic remapping puts the results back in place. This is the set-based part of the hybrid.

The two modes therefore express different assumptions. `normal` asks, “How well does this option fit by itself?” `compare_choices` also asks, “How does this option relate to the alternatives?”

## Train the smallest parts, then keep the final test unread

The work used two training stages.

In stage one, the Gemma torso and inherited adapter stayed frozen. We extracted isolated-option features and fitted the replacement pointer head. Eighteen head configurations used disjoint 1,500-row training and 600-row development probes.

In stage two, the complete `normal` scorer stayed fixed. Only five residual tensors learned from relational tasks: two projections, one hidden transform, its bias, and the output readout. Nine residual configurations used 12,000 training rows and 2,000 development rows from a 19,000-row source.

The important firewall was simple: choose the architecture and weights before reading the 2,000-row final relational test. The process did not deserialize that test payload during selection and did not resume tuning after opening it. A separate 3,000-row calibration split served only post-selection description.

This separation matters because repeatedly adapting to a final score turns that score into development feedback. Keeping the final rows unread made the paired comparison prospective rather than an account of whichever settings happened to look best afterward.

## Stability cost about half a standard-suite point

We used three complementary evaluations. The **standard suite** contains 3,900 multiple-choice questions from 13 tasks. The **typed suite** contains 1,200 held-out decisions spanning choice, yes/no, and ordered-score outputs across emotion, intent, sarcasm, and hate-severity tasks. [JEVBench](https://github.com/fstandhartinger/JevBench) is a public 231-item benchmark for typed decision models that measures decision accuracy alongside schema and operational behavior.

The three scorers produced the following exact outcomes. The original column refers to the controlled released checkpoint; the other columns use isolated branches.

| Measurement | Original | `normal` | `compare_choices` |
|---|---:|---:|---:|
| Standard, 3,900 questions | **73.00%** (2,847) | 72.49% (2,827) | 72.46% (2,826) |
| Typed, 1,200 questions | 62.08% (745) | **62.83%** (754) | **62.83%** (754) |
| JEVBench | 178/231 (77.06%) | **181/231** (78.35%) | **181/231** (78.35%) |
| Semantic flips after complete reversal | 851/3,900 | **0/3,900** | **0/3,900** |

Against the original checkpoint, `normal` gave up 0.51 percentage point on the standard suite and `compare_choices` gave up 0.54 point. That is the honest tradeoff behind “about half a point.” Both modes gained 0.75 point on the typed suite and three correct items on JEVBench.

Comparison did not improve the broad aggregate. It answered one fewer standard question correctly than `normal`, a difference of -0.025641 percentage point, and matched `normal` on typed and JEVBench. The full arithmetic appears in [`results/results.json`](results/results.json).

The mapped permutation checks went beyond complete reversal for the relational scorer. All **100,028 of 100,028** checks passed across option widths four through eight. The largest mapped-logit difference was `4.76837158203125e-07`. Disabling the residual preserved base logits byte for byte.

## Relational questions showed why the second mode exists

The separate final test contained 2,000 comparison problems across six task families and four to eight choices. Isolated `normal` scoring answered **671/2,000**, or **33.55%**, correctly. `compare_choices` answered **742/2,000**, or **37.10%**, correctly. Raw negative log-likelihood moved from 1.5853 to 1.5389.

The paired outcomes explain the **+3.55-point** gain. Both modes got 612 rows right. `normal` alone got 59 right; `compare_choices` alone got 130 right. The residual therefore recovered a net 71 answers on the same rows.

A paired percentile bootstrap resampled those 2,000 matched outcomes 20,000 times. Its 95% interval ran from **+2.20 to +4.90 points**. Put concretely, those endpoints correspond to about 44 to 98 additional correct answers per 2,000 on this population. The interval excludes zero, but it describes these six relational task families—not every decision problem.

## The fast paths failed

Full recomputation looked wasteful because every branch shares a long prefix. A BF16 prefix cache produced observed performance ratios from 0.970×—about 3% slower—to 7.102× faster. It also changed the numbers too much.

Before evaluating it, we set tolerances of 0.03 for raw logits, 0.02 for probabilities, and `1e-6` for mapped permutation error. The cache failed raw-logit parity in **8 of 14** required cases. Its maximum raw-logit difference reached **0.0716138**. FP32 diagnostics stayed near `2.4e-06`, but that did not validate the BF16 path the service needed.

Cross-request batching produced a second setback. Requests evaluated alone had zero semantic flips in 186 permutation checks. Put those same requests into BF16 batches with unrelated padding companions, and **26 of 744 placements flipped**. The maximum aligned-logit change reached **0.1033692**; the maximum probability change reached 0.0193576.

The equations had not changed. The numerical route through padding, kernels, and reduced-precision operations had.

That result supplies the systems lesson: **mathematical symmetry is not a serving guarantee**. A position-free scorer can still lose semantic stability when an optimization changes branch features. Validate aligned logits, probabilities, and decisions through the whole serving stack; do not infer runtime behavior from architecture alone.

## Choose a mode—and honor its execution contract

Use `normal` by default for classification, routing, semantic-label selection, and policies that each option can satisfy independently. Use `compare_choices` when words such as highest, lowest, second, closest, unique, or dominant define the decision.

For either mode, the caller submits all options in one typed request. The supported runtime executes one request at a time, invokes one option branch per model call, uses batch size one and eager attention, fully recomputes each branch, and maps outputs by semantic identity. It does not use a key/value prefix cache or batch unrelated requests together.

That boundary trades throughput for measured behavior. New devices, dtypes, quantization, kernels, attention implementations, caches, or batching strategies need their own conformance checks.

The implementation and tests are in the [v0.1.0 source release](https://github.com/manu-mishra/strands-gemma-order-stable-hybrid/tree/v0.1.0). Model components are available at the Hugging Face [`v0.1.0` tag](https://huggingface.co/manu-mishra/strands-gemma-order-stable-hybrid/tree/v0.1.0).

## What this retrofit established

The project started with a concrete defect: reversing four science answers turned a battery flashlight into a solar calculator, and the same kind of semantic flip appeared 851 times in 3,900 questions.

Isolated branches removed ordered-list position from base scoring while preserving the frozen Gemma torso and inherited adapter. A position-free residual restored useful cross-option reasoning. Together, the modes reached zero complete-reversal flips on the standard suite under the supported path, held broad quality close to the original checkpoint, and improved a prospective relational test by 3.55 points.

The setbacks mattered as much as the positive result. Prefix caching and cross-request batching showed that a sound set computation can still drift in a real numerical stack. The dependable unit is not just the equation; it is the equation plus the execution contract.

The evidence remains bounded. The original 21.82% result covers complete reversal, not every permutation. The evaluations cover named standard, typed, JEVBench, and relational populations. Raw temperature-one probabilities are not presented as calibrated. Images are unsupported, and the loopback server is not production infrastructure. Within those limits, the retrofit shows a practical way to separate semantic choice from display order without retraining the model’s core.