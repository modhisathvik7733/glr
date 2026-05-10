# Plan: a curriculum-first research program for grounded concept learning

## Context

Through three iterations of this conversation, the question has evolved:

1. *"What architecture should I build?"* — first cut. Architecture-centric. Wrong.
2. *"What unified primitive does grounding + reasoning + fluency?"* — better, but still architecture-centric. Hand-waved the foundational problems.
3. **"What learning process creates semantic concepts efficiently?"** — the right question.

The deepest insight, in your words: current LLMs do *language first → concepts emerge accidentally*. You want *concepts first → language emerges naturally*. **That inversion, not any specific layer or block, is the contribution.**

This plan is built around the curriculum. The architecture is in service of the curriculum, not the other way around. The architecture chapter exists only to specify what the curriculum is training; if a simpler architecture lets the curriculum work, use it.

The single goal stated by you: **strong foundations so the model can expand to all domains.** That is the design constraint. Every stage of the curriculum exists to make foundations stronger before expanding.

---

## Why the curriculum is the contribution (the diagnosis)

Random internet pretraining works for current LLMs because they're allowed to be enormous and they don't really need clean concepts — they need surface coverage. For a small, grounded, data-efficient reasoner, this strategy is actively harmful:

| What internet data gives | What it costs |
|---|---|
| Surface fluency | Conceptual incoherence |
| Wide knowledge | Contradictions baked in |
| Pattern coverage | Shortcut features that block abstraction |
| Stylistic variety | Concept entanglement |

Children don't learn this way. They learn objects, then actions, then relations, then causality, then composition, then abstractions, then language *as labels for already-grounded concepts*. The order is not aesthetic — it is **structural**. Every later layer of competence depends on an earlier one being stable. If you teach language before concepts, language *becomes* the concepts (which is what every LLM has done), and the resulting concept system inherits the noise of language.

So the curriculum is not "nice to have." For an architecture that depends on stable concept attractors, it is **load-bearing**. Get the order wrong and the slots collapse, the binding fails, the energy landscape never stabilizes — the architectural problems we discussed in the last iteration are *caused by* feeding language too early. Get the order right and most of those problems prevent themselves.

---

## The curriculum (six stages, with concrete content and gates)

Each stage has: a goal, a dataset, an objective, and a **gate** — a quantitative criterion that must be met before moving to the next stage. Skipping a gate is how research dies.

### Stage A — Concept Physics

**Goal**: stable, disentangled, reusable concept attractors for *objects*, *attributes*, *spatial relations*. No actions yet, no time. Just "what is in the world right now."

**Datasets**:
- **CLEVR** ([CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)) — procedural images of objects with controllable color/shape/size/material/position. Ground truth concept inventory is exact.
- **dSprites / 3D Shapes** — even simpler factor-of-variation datasets. Use these first if CLEVR is too rich.
- A custom procedural generator gives you infinite samples and lets you hold out specific concept combinations for compositional tests.

**Objective**: masked-region prediction in concept space (JEPA-style) + VICReg slot regularization + Sinkhorn-routed slot assignment + slot-consistency under augmentation.

**Architecture used**: minimal — slot-attention encoder + TPR-style binding, no language head. ~30–80M params.

**Gate (must pass to continue)**:
- Linear probe on slot contents recovers ground-truth factors with ≥80% accuracy.
- Mutual-info-gap (MIG) across slots > 0.4.
- Compositional held-out probe accuracy within 5pt of training.
- Causal slot-swap: swap a slot's content between two scenes, ≥70% of swaps produce predicted output change.

**If the gate fails, stop.** The whole approach is in question. Don't paper over with bigger models or more data — diagnose.

### Stage B — Concept Dynamics

**Goal**: actions, transitions, causality. Concepts learned in A now have temporal extent. *Push → object moves. Drop → object falls.*

**Datasets**:
- **Something-Something-V2** — short videos labeled with action templates ("pushing X off Y").
- **CATER** — synthetic videos of object interactions; designed for compositional action understanding.
- **Procedural physics rollouts** (MuJoCo, PyBullet, or 2D box2d worlds) — completely free, controllable, infinite. Probably your most important data source here.
- **MiniGrid** / **BabyAI** — small grid worlds with rule-based dynamics; cheap and great for causality.

**Objective**: predict next-state concept register from current-state concept register and action. Add a *causal intervention loss*: artificially perturb one slot, check whether the predicted future state changes coherently.

**Architecture additions**: a temporal recurrence on the concept register (the Resonate sublayer from earlier drafts becomes a forward dynamics model). Still no language.

**Gate**:
- Next-state prediction accuracy on held-out actions > 70% (concept-level, not pixel-level).
- Counterfactual prediction works: "if X had been pushed instead of Y" produces a sensible prediction not seen in training.
- Held-out action × object combinations show compositional generalization.

### Stage C — Compositional & Relational Reasoning

**Goal**: combine concepts into multi-step structures. *"The cube that is left of the sphere that is bigger than the cylinder."* Multi-hop, recursive, relational.

**Datasets**:
- **CLEVRER** (CLEVR + reasoning questions over videos)
- **GQA** (compositional questions over real images, but with structured ground-truth programs)
- **ConceptARC** — abstract reasoning puzzles, each requiring a different abstract concept.
- **SCAN** and **COGS** — pure compositional generalization benchmarks for sequence-to-sequence models. Cheap to train on, well-understood as diagnostics.
- Synthetic reasoning traces generated programmatically: chains of (entity, relation, entity) with target conclusions.

**Objective**: structured prediction — output a small program / parse / chain of concept operations that explains the input. Train the *iterative* Resonate loop: multiple steps of concept update, supervised with intermediate concept states from the synthetic ground truth.

**Gate**:
- SCAN/COGS held-out compositional split: ≥80% (current SOTA territory for object-centric models).
- ConceptARC: meaningful improvement over an architecture-matched non-curriculum baseline.
- Iterative reasoning depth helps: at fixed parameter count, more Resonate iterations should beat a single iteration on multi-hop tasks. If not, the iterative mechanism isn't doing anything and needs a redesign.

### Stage D — Concept-to-Language Grounding

**Goal**: now, and only now, introduce language. Language is taught as **labels for already-grounded concepts**, not as the substrate of thought.

**Datasets**:
- Captioned versions of A/B/C data (CLEVR captions, Something-Something natural-language descriptions, MiniGrid/BabyAI natural-language instructions).
- Programmatically generated paraphrases (vary syntax while preserving semantics) — this teaches *semantic invariance under linguistic variation*, which is the property that lets the model later survive contact with messy real text.

**Objective**: bidirectional. (a) Predict text from concepts (captioning). (b) Predict concepts from text (parsing). Both losses share the cross-modal projection. The Emit sublayer joins the model here.

**Architecture additions**: surface register S (token autoregressive transformer, optionally warm-started from open-model embeddings), Emit sublayer between C and S.

**Gate**:
- Captioning quality on held-out scenes meets a baseline LM trained on the same captions (fluency floor).
- Concept parsing accuracy from text matches concept extraction from the corresponding scene (cross-modal consistency).
- *Critical test*: paraphrased input → same concept register. If the slot contents change with surface phrasing, language has corrupted concepts and the curriculum has failed.

### Stage E — Controlled Compositional Language

**Goal**: longer, structured language tasks — multi-sentence reasoning, instruction-following, simple dialogue. Still synthetic / curated, no internet.

**Datasets**:
- TinyStories (already curriculum-friendly).
- Procedurally generated multi-step instructions tied to MiniGrid/BabyAI rollouts.
- **Distilled reasoning traces from a teacher**: as you suggested, use a strong teacher *not* to imitate its outputs but to generate structured curricular content — "give me 1000 paraphrases of this concept", "give me 100 chains of reasoning over these objects", "produce 50 dialogues using only these concepts." This is teacher-as-curriculum-generator, not teacher-as-imitation-target. Big distinction.
- Phi-style synthetic textbooks (controlled difficulty progression).

**Objective**: language modeling on rich-but-controlled data, with concept register continuing to track meaning across turns.

**Gate**:
- Instruction-following on held-out instructions matches a same-size LM baseline trained only on Stage E data.
- Reasoning tasks (GSM8K-easy, basic logic) outperform same-size baseline by margin attributable to grounded concepts.
- Concept register remains stable across multi-turn interactions (slot identity persists across dialogue).

### Stage F — Real-World Knowledge as Enrichment

**Goal**: now, finally, expose the model to natural messy text. Books, Wikipedia, code, conversations. Concepts are already stable; language is already grounded; this stage is about *broadening* the concept inventory and the linguistic register, not building it from scratch.

**Datasets**: filtered/quality-weighted subsets of standard corpora — RedPajama-quality-filtered, OpenWebMath, The Stack (code), etc. Filtering matters more than quantity.

**Objective**: continued language modeling + a *concept-stability regularizer* — penalize drift in slot organization relative to a frozen Stage-E checkpoint. This is what protects against catastrophic forgetting of the foundations as the model absorbs noisy data.

**Gate**: natural-language benchmarks (HellaSwag, ARC, MMLU subsets) within striking distance of same-size open models *while* preserving Stage A–C concept probe accuracy. If concept probes degrade by more than ~10pt, the regularization isn't strong enough and you've started corrupting foundations again — the failure mode you correctly named.

---

---

## Data sources and synthesis strategy (per stage)

Saying "diverse synthetic data" repeatedly is meaningless without specifying *where it comes from, how it's generated, and how diversity is actually enforced.* This section makes each stage's data concrete: source, size, generation method, diversity mechanism, filter.

### Stage A — Concept Physics

Use the **richer object-centric benchmark family** that has emerged since the original CLEVR / dSprites — these specifically target compositional generalization, complex textures, and multi-part objects, which is what we want concepts to capture.

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| dSprites | shape, scale, orientation, position | ~700K images / 300MB | Direct download (DeepMind, public) |
| 3D Shapes | shape, color, scale, orientations, floor/wall color | ~480K images / 1GB | Direct download (DeepMind, public) |
| **CLEVRTex** | objects with **homogeneous textures** — forces concepts beyond solid color | ~50K images | [GitHub: CLEVRTex](https://github.com/karazijal/clevrtex) |
| **Super-CLEVR** | broader object inventory, harder VQA | ~30K images + 60K questions | [Super-CLEVR](https://github.com/Lizw14/Super-CLEVR) |
| **MOVi-C** (Kubric) | **realistic textured objects + photorealistic rendering** | unlimited | Generate via [Kubric](https://github.com/google-research/kubric) |
| **MultiShapeNet (MSN)** | complex realistic furniture objects | ~1M scenes | Public release |
| **PTR** | objects composed of **multi-colored parts and textures** — tests part-whole concepts | ~70K images | [PTR dataset](http://ptr.csail.mit.edu/) |
| CLEVR + held-out compositions | held-out tuples, OOD attributes | unlimited | Generate via the CLEVR Blender pipeline; exclude specific (color, shape, size) tuples |

**Diversity enforcement (Stage A)**:
- **Mixed-batch training**: every batch samples from all five sources. Without mixing, the model overfits to one ontology (CLEVR-style 3D shapes have rendering biases that don't transfer to 2D, and vice versa).
- **Factor-aware augmentation** (for slot-consistency loss): for dSprites/3D Shapes, perturb one ground-truth factor at a time. The slot-consistency loss then asks: did only the corresponding slot change? This is much stronger supervision than generic crop/jitter.
- **Held-out splits, two kinds**: (a) compositional held-out — specific (color, shape, size) tuples never seen during training; (b) attribute held-out — a specific shape (e.g., star) never seen, tests slot-content novelty handling.

### Stage B — Concept Dynamics

Big upgrade here. The 2024–2026 physics-reasoning benchmarks (Physion++, IntPhys, CausalVQA) are dramatically better grounded than Something-Something for *causal* reasoning. **Genesis** (2025) is a universal physics engine that lets you generate massive, diverse physics rollouts on consumer hardware.

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| **Physion / Physion++** | **intuitive physics**: rolling, sliding, falling, colliding, deforming, **with latent physical properties to infer** | 1.2K + augmented examples | [physion-benchmark.github.io](https://physion-benchmark.github.io/) |
| **IntPhys** | **possible vs. impossible events** — direct probe of physics understanding | ~15K trials | [IntPhys benchmark](https://arxiv.org/abs/1803.07616) |
| **CausalVQA** (Meta, 2025) | **counterfactual / hypothetical / anticipation / planning** physics Q&A on real video | 1,586 items + 779 video segments | [CausalVQA repo](https://github.com/facebookresearch/CausalVQA) |
| **VideoPhy-2** (ICLR 2026) | action-centric physical commonsense | benchmark | [VideoPhy](https://github.com/Hritikbansal/videophy) |
| Something-Something-V2 | object-action templates ("pushing X off Y") | 220K videos / ~20GB | Public download (free with registration) |
| CATER | compositional action sequences in synthetic 3D | 5.5K videos | Public download |
| **Genesis physics engine** (2025) | causality, momentum, gravity at **massive parallelism** on consumer GPUs | unlimited | [Genesis: universal physics engine](https://genesis-embodied-ai.github.io/) — replaces hand-rolled MuJoCo |
| **Habitat 3.0** | photorealistic indoor environments + **social interaction** (avatars, furniture) | unlimited | [aihabitat.org](https://aihabitat.org/) |
| **ProcTHOR** | procedurally generated 3D embodied environments at scale | 10K+ houses procedurally | [ProcTHOR](https://procthor.allenai.org/) |
| MiniGrid / BabyAI rollouts | discrete causality, multi-step actions, planning | unlimited | Library output, generate at runtime |
| DM Control suite rollouts | continuous control + state | unlimited | DeepMind Control Suite library |

**Synthesis specifics for procedural physics** (the hardest to do well):
- 5–10 primitive shapes, varying mass and elasticity
- Action repertoire: push, drop, throw, stack, knock-over, contain
- ~100K random initial configurations
- 30 frames per rollout at 10fps, paired with structured ground truth: (initial state, action, final state, intermediate state at frames 5/10/15/20/25)
- **Crucial diversity axis**: vary physics constants (gravity, friction, damping) *independently* from rendering style. Generate the same physics under photorealistic and toon rendering. The model is forced to decouple physics from appearance.

### Stage C — Compositional & Relational Reasoning

ARC-AGI-2 (2026) is much harder than ARC-AGI-1 — average compositional depth went from 1.3 to 2.7 steps, exactly isolating compositional generalization as the failure mode. **Compositional-ARC** specifically tests systematic generalization. Both should be in your evaluation suite (training is harder — ARC tasks are few-shot by construction; you'd train on similar synthetic compositions and evaluate on ARC).

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| **ARC-AGI-1 + ARC-AGI-2** | abstract spatial reasoning under increasing compositional depth | ~1000 tasks | [arcprize.org](https://arcprize.org/arc-agi) — primarily an evaluation set; use as the gold-standard probe |
| **Compositional-ARC** (2026) | **systematic generalization** to unseen compositions of geometric transformations | benchmark with controlled splits | [arXiv 2504.01445](https://arxiv.org/abs/2504.01445) |
| CLEVRER | causal video reasoning Q&A | 20K videos + 300K questions | Public download |
| GQA | compositional VQA with structured programs | 22M questions / 113K images | Public download |
| ConceptARC | abstract visual reasoning | ~700 tasks | Public (small but high-value diagnostics) |
| SCAN | compositional sequence-to-sequence | unlimited | Generate with reference grammar |
| COGS | semantic parsing, held-out compositions | 24K train + held-out splits | Public |
| Synthetic relation chains | (entity, relation, entity) → conclusions | unlimited | Knowledge-graph-style generator (you write) |
| **Synthetic ARC-style training data** | priors needed to crack ARC at test time | unlimited | Generate ARC-style transformations programmatically; current SOTA on Compositional-ARC is a 5.7M-param meta-learning model, suggesting small models can succeed here |

**Synthesis for relation chains**: build a synthetic knowledge graph with ~50 entity types and ~30 relation types. Sample random connected subgraphs of size 3–8. Generate questions of the form "what is the relationship between X and Z (given X→Y, Y→Z)?" Hold out specific (entity_type × relation_type) combinations for compositional evaluation. ~500 lines of generator code; produces unlimited examples.

### Stage D — Concept-to-Language Grounding

DOCCI (Google, 2024) is exceptional for this stage: 15K images with **136-word average dense human descriptions** that explicitly cover spatial relations, counting, text rendering, world knowledge — i.e., the things concepts are supposed to capture. Apple's DataComp-12M is the cleaner replacement for LAION-COCO.

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| **DOCCI** (Google, 2024) | **dense, human-grounded descriptions** with spatial relations, counting, contrast | 15K images, ~136 words/caption | [google.github.io/docci](https://google.github.io/docci/) — high-value-per-example |
| **DataComp-12M** (Apple) | curated multimodal pairs from a 12.8B-pair pool with **proven-quality CLIP filtering** | 12M | [HF: apple/DataComp-12M](https://huggingface.co/datasets/apple/DataComp-12M) |
| CLEVR templated captions | scene → natural-language description | 700K captions | CLEVR templating pipeline |
| Something-Something-V2 labels | action template → caption | 220K | Already paired |
| BabyAI instructions | action → English instruction | unlimited | Library output |
| Programmatic paraphrases | semantic invariance under linguistic variation | ~5–10× the caption set | Teacher LM (Llama 3.1 8B / Qwen 2.5 7B for paraphrasing) |
| **GroundCap** (2025) | visually-grounded image captioning with explicit grounding annotations | recent benchmark | [arXiv 2502.13898](https://arxiv.org/abs/2502.13898) |

**Paraphrase generation specifics** (this is what enforces semantic invariance, the property that lets concepts later survive Stage F's messy text):
- For each Stage A/B/C caption, prompt the teacher: *"rewrite this sentence preserving meaning, varying syntax. Produce 5 versions."*
- Filter teacher output: discard paraphrases whose Absorb output diverges from the original beyond a similarity threshold (this is the "filter through your concept register" mechanism — see also the teacher-curriculum filter below).
- Train Emit to produce any of the paraphrases given the same concept register; train Absorb to produce identical concept registers from any paraphrase. This is the bidirectional invariance constraint.

**Diversity strategy for Stage D**: at least 3 syntactic templates per scene (active/passive/relative-clause), 2 register variations (formal/colloquial), and 1–2 length variations (short/long).

### Stage E — Compositional Language

The 2025–2026 wave of distilled reasoning datasets (OpenR1-Math, NuminaMath, Cosmopedia, FinePhrase) is exactly what this stage needs. **Use OpenR1-Math-220k for reasoning grounding** — its R1-distilled traces are the strongest open dataset for teaching multi-step reasoning. **Cosmopedia / FinePhrase** are the modern Phi-style textbook generators.

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| TinyStories | simple narrative composition (entry-level) | 2.7M stories / ~5GB | Public download |
| **Cosmopedia v2** (HuggingFace) | textbook-quality synthetic content across 200+ topics | ~25B tokens | [HF: HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) — direct successor to Phi's textbook approach |
| **FinePhrase** (HuggingFace, 2026) | **state-of-the-art synthetic web rewrites** (FAQ/Math/Table/Tutorial formats) generated from FineWeb-Edu | sizable | [HF FinePhrase](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) (recent release) — beats Cosmopedia and Nemotron-HQ on benchmarks |
| **OpenR1-Math-220k** | **R1-distilled multi-step math reasoning traces** | 220K problems, 800K traces | [HF: open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — biggest reasoning-grounding upgrade |
| **NuminaMath 1.5** | broad math problem source for reasoning | sizable | [HF: AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) |
| **OpenThoughts** | distilled reasoning traces from R1 | 1.4M+ examples | [arXiv 2503.19633](https://arxiv.org/html/2503.19633v1) |
| BabyAI multi-step instructions | grounded multi-step language | unlimited | Library output paired with environment state |
| Symbolic reasoning chains | logical composition (no code yet) | unlimited | Templated generator (you write) |

**Teacher-as-curriculum-generator specifics** (your earlier point, made concrete):
- **Teacher choice**: Llama 3.1 8B Instruct or Qwen 2.5 7B Instruct. Both free, fluent enough, runnable on a consumer GPU.
- **Syllabus**: pre-define ~200 concept topics drawn from Stage A–C inventory (e.g., "containment," "agency," "transitive action," "causality with delay," etc.).
- **Per-topic prompts**: teacher generates 50 explanations at varying difficulty, 100 worked examples, 50 paraphrases of each example, 50 incorrect-answer-with-explanation pairs.
- **Filter through your concept register** (this is the part I previously didn't specify):
  - Your model's Absorb runs on each teacher-generated example, producing a concept register `C_teacher`.
  - Acceptance test: (a) the energy `E(C_teacher)` settles below threshold within iteration cap; (b) `C_teacher` parses to a concept inventory consistent with the topic prompt (cosine similarity > 0.7 to the prompt's concept seed).
  - Examples failing either test are discarded.
  - This is the gate that prevents teacher noise from corrupting your concept inventory — *your* model decides what counts as curricular-quality, not the teacher.
- **Budget**: ~5–10M tokens of accepted teacher content. Teacher inference ~$50–100.

### Stage F — Real-world Knowledge

Major upgrade. RedPajama is now superseded by the FineWeb / DCLM / Nemotron-CC family — these are 2024–2026 quality-filtered datasets explicitly designed for small-model training. **Nemotron-CC** is currently the most ambitious: 6.3T tokens with 1.9T tokens of synthetic rephrasings, designed precisely to avoid the "throw away 90% of data" problem that plagued FineWeb-Edu and DCLM.

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| **Nemotron-CC** (NVIDIA, 2024) | **highest-quality general web** with synthetic rephrasings, optimized for both short and long token horizons | 6.3T tokens (1.9T synthetic) | [HF: nvidia/Nemotron-CC](https://huggingface.co/datasets/nvidia/Nemotron-CC) |
| **FineWeb-Edu** | educational web subset, very strong for small models | 1.3T tokens | [HF: HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **DCLM-baseline** | DataComp-LM filtered baseline | ~3.8T tokens | [HF: mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) |
| **OpenWebMath** | math in natural language | ~14B tokens | [HF: open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) |
| Wikipedia (filtered) | encyclopedic knowledge | ~5B tokens | Public, filter by page quality |
| **The Stack v2** (filtered, weighted equally) | code as one modality among many | ~10B tokens at this stage | [HF: bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) — license-filtered |
| StackExchange Q&A | dialogue + reasoning | ~5B tokens | Public |

**Three-stage filtering pipeline** (this is the part that determines whether Stage F preserves or corrupts foundations):
1. **Phi-style quality classifier**: train a small classifier (~20M params) on existing labeled high/low-quality web text. Apply to candidates; reject bottom 50%.
2. **Concept-parseability filter**: run candidates through your model's Absorb. Documents whose concept register fails to settle (high terminal energy after iteration cap) are flagged. **Keep ~10% of flagged documents** — these are out-of-distribution and the model needs to learn to handle them — but discard the rest.
3. **Mix-balance enforcement**: at every batch, sample roughly equally across modalities (math, encyclopedia, code, dialogue, general web). The natural distribution of internet text is 80%+ general web; deliberately re-weight to balance. This is the structural defense against any single modality (especially code, given your Creor goal) dominating Stage F's concept formation.

### Stage G — Code as first-class modality

The Stack v2 (BigCode, 2024) is the current SOTA — 600+ languages, license-clean. **SwallowCode (2026)** is the most exciting new entrant: 16.1B tokens of *refined Python* through a four-stage pipeline (syntax validation → pylint filtering → two-stage LLM rewriting). It demonstrably improved HumanEval pass@1 by **+17.0** over Stack-Edu in continual pretraining at 50B tokens. **Stack-Edu** is the educational subset.

| Source | Teaches | Size | How to obtain |
|---|---|---|---|
| **The Stack v2** (curated, license-filtered) | broad code distribution across 600+ languages | filtered subset of ~3T tokens | [HF: bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| **SwallowCode** (2026) | **refined, style-conformant, self-contained Python** with proven HumanEval lift | ~16.1B tokens | [SwallowCode](https://huggingface.co/datasets/tokyotech-llm/swallow-code) — biggest 2026 quality jump |
| **Stack-Edu** | educational code subset, best for foundation | filtered subset | [Part of SmolLM training set](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| **OpenCodeReasoning** / **KodCode** | distilled code reasoning traces | sizable | Recent open-source releases |
| **Qwen2.5-Coder synthetic** | code synthesized by Qwen2.5-Coder for self-training | sizable | Generation pattern from Qwen3 paper |
| Real Creor user telemetry (with consent) | what users actually want | ongoing | Your product loop |
| Synthetic codebases | architectural concepts | unlimited | Templated generator with concept seeds (you write) |
| Code review + commit pairs | refactor reasoning | scrape from GitHub Archive, filter by review-comment density | GitHub Archive |
| Synthetic refactor pairs | explicit before/after examples | unlimited | Apply controlled mutations to working code, then correct |

**Synthetic codebase generation specifics**:
- **Seed**: pick an architectural pattern (microservices, event-driven, MVC, repository pattern, hexagonal, etc.)
- **Generate**: parameterized project generator (e.g., a templated FastAPI / Flask / Express app builder) produces a working multi-file project with that pattern.
- **Inject**: controlled bugs or sub-optimal patterns (n+1 queries, missing-validation, tight coupling, etc.).
- **Correct**: generate the fixed version.
- **Output**: (broken_project, fix_diff, explanation) tuple. Train Emit to produce the diff; train Resonate to derive the fix from the architectural concepts.

**Creor telemetry use** (with explicit user consent, opt-in):
- Accepted completions: positive examples.
- Rejected completions / completions edited within 30 seconds: negative or implicit-correction examples.
- Explicit thumbs up/down: highest-weight signal.
- Format as preference pairs for DPO-style fine-tuning of Emit. Concept register stays frozen during DPO so foundation isn't disturbed.

---

## Stage-to-stage transitions: what gets frozen and unfrozen

The curriculum is implemented as a sequence of training runs where specific blocks are frozen, unfrozen, or initialized. Pinning this down so it's not ambiguous:

| Stage | Initialize from | Frozen | Trained | New modules added |
|---|---|---|---|---|
| A | random | none | slot encoder + TPR + Absorb | concept register `C` |
| B | Stage A checkpoint | slot encoder | + temporal Resonate, dynamics head | recurrence over `C` |
| C | Stage B checkpoint | slot encoder | + iterative Resonate, energy head | energy gate |
| D | Stage C checkpoint | slot encoder, Resonate (low LR) | + Emit, surface register `S` (warm-started from Llama-3 embeddings) | language head |
| E | Stage D checkpoint | none | full model fine-tune at low LR | none |
| F | Stage E checkpoint | none, but with concept-stability regularizer at full strength | full model + stability regularizer | none |
| G | Stage F checkpoint | concept register's content distribution (via stability regularizer at *max* strength) | + AST-aware Absorb path, code-Emit specialization | tree-sitter AST embedder |
| H | Stage G checkpoint | core model | + project-level concept memory module | persistent project graph |

**Rules**:
- Always continue training from the previous stage's checkpoint, never from scratch.
- "Frozen" means parameters don't update; gradients do flow through (so downstream losses can still propagate).
- LR schedule: each stage starts at half the previous stage's peak LR. This prevents disturbance to already-converged blocks.
- After every stage, run the cumulative gate evaluation suite: all gates from previous stages must continue to pass. If a previous stage's gate regresses by more than 5pt, stop, debug, and increase the freeze/regularization strength.

---

## Faithfulness and concept-stability losses, made implementable

I wrote the loss formulas but not how to actually compute them. Concrete implementations:

### Faithfulness loss (Stages D, E, F)

```
L_faithful = α₁ · L_align + α₂ · L_reconstruct
```

- **`L_align` (contrastive alignment)**: take the input's concept register `C_in` (computed by Absorb on input alone) and the output's concept register `C_out` (computed by Absorb on the generated text). Apply InfoNCE: `C_in` and `C_out` should match more closely than `C_in` matches `C_out` from a randomly sampled different example in the batch. Standard contrastive setup, ~30 lines of PyTorch.
- **`L_reconstruct`**: pass `C_out` through a small auxiliary projection to predict the input tokens. Cross-entropy loss against the actual input. Forces the output's concept content to retain enough information to recover the input.

Suggested initial weights: α₁ = 0.3, α₂ = 0.1. Tune via ablation on a held-out faithfulness benchmark (e.g., generate output for 1000 inputs, check that input is recoverable from output's concept register at >70% accuracy).

### Concept-stability regularizer (Stage F, max strength in Stage G)

Three terms, implemented as:

- **`λ₁ · D_KL`**: Stage E's slot-content distribution is captured as a moving average of concept-register vectors during the last 1% of Stage E training. Stored as a frozen reference distribution. KL between current Stage F slot distribution and this reference. Implementation: model the reference as a mixture of K Gaussians per slot.
- **`λ₂ · routing drift`**: store Stage E's mean attention pattern (slot-routing) on a fixed reference set of inputs. During Stage F, run those reference inputs and penalize Frobenius-norm divergence between current routing and stored reference.
- **`λ₃ · frozen probe anchor`**: maintain a frozen Stage A–C probe head and a frozen reference test set (~10K examples drawn from CLEVR/dSprites/CATER). Every 5K Stage F steps, run the probes on the reference set; the probe accuracy is added as a loss term (penalize drops below Stage E's accuracy).

The frozen reference test set is the **concept-foundation regression suite**. It is the load-bearing artifact that makes "foundation preserved" verifiable rather than wishful.

---

## What expands "to all domains" mean concretely

You said: *"i want to make sure that the foundations are strong so model can expand better to all domains."* This curriculum is structured to make that expansion possible because:

- **Stages A–C give you a domain-agnostic concept substrate.** Objects, relations, causality, composition are the building blocks of *every* domain — physics, code, mathematics, social reasoning, planning. A model that has them stably can be specialized to a new domain by adding domain-specific concept inventory, not by retraining the substrate.
- **Stage D's invariance property is the key transfer mechanism.** If concepts survive paraphrase, they will (with high probability) survive translation into other domains' vocabulary, because vocabulary is just a different "paraphrase" of an underlying concept.
- **Stages E–F build the language register without overwriting foundations.** The concept-stability regularizer in Stage F is the architectural commitment to *not letting downstream training corrupt upstream learning*. This is what makes the foundation actually load-bearing.

The opposite — what current LLMs do — is to optimize directly for downstream surface performance, hoping the foundations show up. They sometimes do, accidentally, in models large enough to brute-force it. You don't have that budget. The curriculum is what substitutes capacity with structure.

---

---

## Architecture spec (instrumental — designed to make the curriculum trainable)

This is the minimum architecture the curriculum needs. It is *not* the contribution — the curriculum is. But the curriculum can't run on a vanilla transformer, so this section pins down what the model looks like.

### State

| Register | Shape | Purpose |
|---|---|---|
| Concept register `C` | `K × d_role × d_filler` (e.g. 64 × 16 × 32) | Slot-bound TPR concepts |
| Surface register `S` | `T × d_model` (T up to 2048) | Tokens, video patches, action embeddings |
| Energy scalar `E(C)` | `R` | Used to halt iterative reasoning |

`d = d_role · d_filler` is the effective slot width. Roles are a small learnable basis; fillers carry the bulk of concept content.

### One repeated block ("BCS layer"), three sublayers

1. **Absorb**: `C ← cross_attn(C, S)` — pull surface evidence into slots.
2. **Resonate**: `C ← self_attn_TPR(C)` — slots interact, bind, compose. This is the reasoning step. Iterated 1–N times at inference, controlled by `E`.
3. **Emit**: `S ← cross_attn(S, C) + self_attn(S)` — concepts modulate surface; surface still has its own self-attention so token-level structure is preserved (this is the fluency safeguard).

### Where each curriculum stage uses what

| Stage | Absorb | Resonate | Emit | S register |
|---|---|---|---|---|
| A — Concept Physics | image patches → slots | 1 iter | none | image-only |
| B — Dynamics | video frames → slots | recurrent over time | none | video-only |
| C — Reasoning | scenes → slots | iterative, energy-gated | none | scene-only |
| D — Language Grounding | text + scene → slots | 1 iter | text from concepts | text + image |
| E — Compositional Language | text → slots | iterative | text from concepts | text |
| F — Real-world | text → slots | iterative | text from concepts | text |

The same parameters are used across stages; new stages add modalities or unfreeze sublayers, they don't introduce new modules. This is what "in service of the curriculum" means concretely.

### Fluency commitment (your original concern, restated)

- Surface register `S` keeps full self-attention from Stage D onward. It is, internally, a small autoregressive LM.
- Warm-start `S` from open-model embeddings (Llama-3 tokenizer + first-layer embeddings) at Stage D.
- The Emit sublayer adds concept conditioning *via cross-attention*; it does not replace token-level self-attention.
- **Structural guarantee**: in the worst case where concepts contribute nothing useful, the model degrades to a same-size LM. Fluency cannot drop below baseline.
- Concept register being useless is detectable (probe accuracy drops). So you'll know if Emit has stopped using concepts.

---

## Training strategy — the full how, not just the what

The curriculum says *what* to train at each stage. This section says *how* to train it: optimization recipe, hyperparameters per stage, within-stage data ordering, multi-task loss balancing, replay-based anti-forgetting, RLVR / GRPO for reasoning, distillation strategy, and stability practices. Most of these are not invented — they are drawn from the 2025–2026 training playbooks (SmolLM3, DeepSeek-R1, GRPO, FOREVER replay, EDCO curriculum) and adapted to this curriculum.

### Default optimization recipe (applies to all stages unless overridden)

| Knob | Value | Source / why |
|---|---|---|
| Optimizer | AdamW, β₁ = 0.9, β₂ = 0.95, ε = 1e-8, weight decay = 0.1 | [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) defaults; β₂ = 0.95 stabler than 0.999 for small models |
| LR schedule | linear warmup → cosine decay | Standard, robustly best for small models |
| Warmup steps | ~10% of total stage steps, capped at 2000 | SmolLM3 finding |
| Gradient clipping | global norm 1.0 | Mandatory for slot-attention stability |
| Precision | bfloat16 with fp32 master weights | bfloat16 has wider dynamic range than fp16, no loss scaling needed |
| Activation memory | gradient checkpointing on Resonate iterations | Pays K× compute for K× memory savings on iterative loops |
| Attention kernel | FlashAttention-2 / 3 | Mandatory; 2–3× throughput |
| Compile | `torch.compile(mode='reduce-overhead')` on the BCS layer | Real speedup on PyTorch ≥ 2.2 |

### Per-stage hyperparameters

These are starting points. Sweep on a 10× smaller proxy model first.

| Stage | Peak LR | Warmup | Total tokens | Batch tokens | Replay weight | Notes |
|---|---|---|---|---|---|---|
| A | 5e-4 | 2000 | 5–10B equiv | 256K | n/a (first stage) | Slot encoder + TPR + VICReg |
| B | 3e-4 | 1500 | 10–20B | 512K | 10% from A | Add temporal Resonate |
| C | 3e-4 | 1500 | 10–20B | 512K | 10% from A+B mix | Iterative Resonate + energy head |
| D | 2e-4 | 2000 | 20B | 1M | 10% from B+C | Emit warm-start: lower LR to protect Resonate |
| E | 1.5e-4 | 1500 | 30B | 1M | 15% from C+D | Multi-task balanced |
| F | 1e-4 | 2000 | 100B | 1.5M | **20%** from C–E + concept-stability regularizer | Lowest foundation-stage LR; max anti-forgetting |
| G | 5e-5 | 2000 | 50B | 1M | 25% from C–F | Code specialization, regularizer at MAX |
| H (RL) | 1e-5 (policy) | 500 | ~5B equiv | 256K | n/a (online) | GRPO/RLVR, KL penalty against Stage G |

**Why LR halves across stages**: each stage starts at half the previous stage's peak LR. This prevents disturbance to converged blocks and is what staged training literature consistently supports. The smol-training-playbook's specific finding that LR ~5e-4 with 2000 step warmup is robust for small models is the calibration point.

### Within-stage curriculum (the *inner* curriculum)

Curriculum learning at the *example* level reduces training steps by 18–45% to reach baseline performance ([arXiv 2506.11300](https://arxiv.org/abs/2506.11300)). Within each stage, order training data by composite difficulty:

**Difficulty signals (composite score)**:
- **Compression ratio** (gzip on text) — proxy for information density
- **Lexical diversity (MTLD)** — vocabulary richness
- **Readability (Flesch)** — surface complexity
- **Per-sample loss** at start of stage — model-perceived difficulty
- **Inference entropy** (EDCO 2026 method, [OpenReview EDCO](https://openreview.net/forum?id=Oboo6f5dQl)) — predictive uncertainty proxy

**Schedule**: easy → hard with 10–15% periodic re-injection of easy examples (anti-forgetting at the within-stage level). Concretely: sort by composite score, divide into deciles, and at each step sample from a mixture: 70% from the current target decile, 20% from earlier deciles, 10% from later deciles.

**Where curriculum learning fails**: post-training of large LLMs ([OpenReview](https://openreview.net/forum?id=sHn5rq6L0O)). For our small-model regime curriculum gains are large, but be aware that this advantage shrinks at scale — relevant only if you eventually scale up.

### Multi-task loss balancing

Stages D, E, F, G have multiple simultaneous losses (faithfulness, VICReg, concept-stability, masked prediction, CE on tokens). Balancing is non-trivial.

**Three-step protocol**:
1. **Static weights at start**: ablate on smaller proxy runs. Initial guess for Stage E: λ_CE = 1.0, λ_faithful = 0.3, λ_VICReg = 0.1, λ_stability = 0.5 (Stage F+).
2. **Auxiliary loss warmup**: turn aux losses on gradually over the first 5% of stage steps. Lets the main task establish a baseline before constraints fire. Without this, the model often collapses to satisfying the regularizer at the cost of the task.
3. **GradNorm or PCGrad** if losses fight (you'll see oscillating eval metrics): project conflicting gradients. Implementation in PyTorch is ~50 lines.

### Replay strategy — the load-bearing anti-forgetting mechanism

This is the practical mechanism behind the "foundation preserved" promise. Without it, Stage F's natural-text exposure will erase Stage A–C concepts.

**Three layers, stackable:**

1. **Persistent replay buffer**: at the end of each stage, store ~5% of the stage's data as anchor examples. Maintain across all subsequent stages.
2. **Mixed-batch composition** (table column above): 80–90% current-stage data, 10–25% replay from previous stages. The replay-weight column above scales up at later stages because the foundation gets more vulnerable as more new data arrives.
3. **Smart replay scheduling** — two recent improvements over uniform replay:
   - **FOREVER** ([arXiv 2601.03938](https://arxiv.org/html/2601.03938)): align replay frequency with optimizer update magnitude (model time), not training step (clock time). Implementation: track the moving average of update norm; trigger an extra replay batch when current update norm > 2× moving average (model is in a high-update regime, more likely to forget).
   - **SURE** ([OpenReview](https://openreview.net/pdf?id=IgZWU75BLL)): surprise-driven prioritized replay — samples where the current model is surprised (high loss) get higher replay priority. Cheap to compute, large effect.

**Combined with**: stage-transition freezes (already specified) + concept-stability regularizer (Stage F+) = three-layered defense against catastrophic forgetting. This is the most comprehensive defense the literature currently offers; doing all three is what the foundation-preservation goal requires.

### Reinforcement learning post-training (Stage H, after Stage G's SFT)

Modern reasoning models all rely on RL post-training. Skipping it leaves a large reasoning-quality gap. The 2025–2026 standard is **GRPO** ([Cameron Wolfe on GRPO](https://cameronrwolfe.substack.com/p/grpo)), which DeepSeek-R1 used:

**GRPO recipe for our model**:
- For each prompt, sample G = 8 candidate completions from the current policy.
- Compute reward via **verifiable check (RLVR)** — no human preference labels needed:
  - **For code**: does it run? pass tests? satisfy the energy-gate convergence check (reasoning settled)?
  - **For math**: final answer matches ground truth?
  - **For architectural reasoning** (Stage H Creor specialization): proposed plan satisfies structural invariants (no orphan dependencies, no broken interface contracts, no unreachable code)?
- Compute group-relative advantage: `A_i = (r_i − mean(r)) / std(r)`. **No critic needed.**
- Policy update with KL penalty against pre-RL checkpoint (β = 0.04 typical).

**Critical adaptation for this architecture**: the reward function should *include the energy-gate convergence signal*. Reasoning that produces a correct answer but never settled energy should be penalized — it indicates the model bypassed the concept register and got the answer "by accident" through token-level shortcuts. Otherwise RLVR will reward shortcut reasoning that breaks foundation discipline.

**Why GRPO over PPO**: PPO needs a critic (~doubles memory and compute). GRPO's group-relative advantage works as well or better empirically and fits a single A100. ([Sebastian Raschka summary](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training).)

**Why not just DPO**: DPO (and SimPO/KTO) are good for alignment but weaker for reasoning specifically — they learn from static preference pairs and can't explore. The 2026 standard stack is: SFT → DPO/KTO (alignment) → GRPO/RLVR (reasoning).

### Distillation strategy (where the teacher actually goes)

Two distinct uses of a teacher (Llama 3.1 8B / Qwen 2.5 7B / Qwen3-coder for Stage G):

1. **Curriculum generation** (Stage E, F, G) — teacher generates synthetic curricular content; we filter via concept register (recipe in the data section). Teacher acts as content factory.
2. **Hidden-state distillation for fluency** (Stage D, optional) — match teacher's last-layer hidden states on non-reasoning tokens (KaVa-style, [arXiv 2510.02312](https://arxiv.org/html/2510.02312v1)). Imports baseline fluency cheaply.

**What you should NOT do with the teacher**: token-level KL distillation on reasoning steps. Your model reasons in concept space, not token space — KL between teacher tokens and student tokens during reasoning is incoherent and will pull reasoning back into token-space, defeating the entire architecture.

### Stability practices (the engineering monster you correctly flagged)

You called training instability the "engineering monster hidden underneath everything." Specific mitigations:

| Failure mode | Defense | Cost |
|---|---|---|
| Loss spikes (sudden NaN/inf) | If loss > 3× EMA, skip batch, drop LR 5% for 100 steps | ~free |
| Slot routing collapse | Temperature-scaled softmax (τ = 0.5 initially), clamp routing logits to ±10 | ~free |
| Slot deadness (some slots never activate) | Add a usage-balance loss penalizing low-entropy routing distribution | small grad-cost |
| Resonate iteration explosion | Spectral normalization on the recurrent block weights | small forward-cost |
| Concept oscillation (`C_t` flipping between attractors) | Detect via cosine sim between consecutive `C_t`; if oscillating, halt early | ~free |
| VICReg covariance term going negative | Clamp covariance term to 0 from below; never let it push gradients in the wrong direction | ~free |
| Memory blow-up on iterative Resonate | Activation checkpointing + gradient accumulation on iteration count | trades compute for memory |
| Mixed-precision underflow on small gradients | Scale concept-register loss by 8× (since slot-attention has small grads) | ~free |

These are debugged-once-then-forgotten engineering details. Bake them into the training loop from day 1; don't discover them during a 5-day Stage A run.

### Eval cadence and early-stop discipline

- **Every 5K steps for short stages (A, B, C), every 20K for long stages (E, F, G)**: run stage-specific gate metrics + the cumulative foundation regression suite.
- **Early stop trigger**: if any earlier-stage gate regresses by > 5pt without recovery for 50K steps, **halt training and debug**. Don't keep training a model that is undoing earlier stages.
- **Run Stage A 3× with different seeds before declaring the gate passed**. Variance characterization matters; a single seed's pass could be lucky.

### Hyperparameter tuning under your compute constraint

Don't grid-search on the full A100. Two-step protocol:

1. **Proxy model search**: train a 1/10th-scale model (~5–8M params) for 24 hours with Bayesian optimization (Optuna) over: peak LR (log-uniform 1e-5 to 1e-3), batch size (256K to 2M tokens), warmup (500–4000), λ-weights for aux losses. ~50 trials.
2. **Top-3 transfer to real scale**: run the top 3 configs from the proxy on the real model. Pick the best.

**Heuristic from SmolTulu** ([arXiv 2412.08347](https://arxiv.org/html/2412.08347v1)): for reasoning-heavy stages (C, E, G), higher LR-to-batch-size ratios produce better reasoning in small models. So bias your sweep toward higher LRs at smaller batch sizes when tuning Stages C, E, G specifically.

### Within-batch composition for multi-modal stages (Stage B, D, F)

Naive concatenation of modalities is wasteful. Use **modality-balanced sampling**:
- For each batch, sample equally from each modality.
- For each example, randomly mask the modality with 30% probability — forces cross-modal prediction.
- Schedule the mask ratio: start at 10% (model needs to see all modalities aligned), increase to 30% as training proceeds (forces cross-modal generalization).

### Monitoring (what to watch every step)

A tight set of dashboards prevents silent failure:

- **Loss panels**: each loss component separately, plus weighted total
- **Gradient norm**: per parameter group; alert on > 2× EMA
- **Slot routing entropy**: average over batch; alert on collapse (entropy → 0) or explosion (entropy → uniform)
- **Energy convergence rate**: fraction of examples where E settles within iteration cap; should rise during Stage C
- **Probe accuracy on the foundation regression suite**: every 5K steps, on the held-out reference set
- **Replay loss**: loss on current-stage data vs. replay buffer — if replay loss rises while current-stage loss falls, you're forgetting

These are the early-warning signals that prevent wasting your A100 budget on a run that has silently broken.

---

## Inference behavior

A single forward pass for a user query:

1. Tokenize query into `S`.
2. Run Absorb once: concepts `C₀` are populated from the query.
3. Loop Resonate: `C_{t+1} = Resonate(C_t)` until either (a) `|E(C_{t+1}) − E(C_t)| < ε`, or (b) iteration cap reached.
4. Run Emit autoregressively: at each token step, attend to settled `C_T` and to surface history. Sample next token. Append. Repeat until EOS.

**Key property**: easy queries use 1 Resonate iteration (chat-speed). Hard queries use many (think-then-speak). The model decides via the energy head.

**Calibrating the energy head**: in Stage C, supervise `E` against ground-truth correctness of intermediate predictions — `E` should drop monotonically as predictions become correct. If it doesn't, the energy head is uncalibrated and adaptive halting won't work.

---

---

## Input understanding, output faithfulness, and avoiding the "trained-concepts-only" trap

A real risk with any concept-centric architecture is that the model collapses to a fixed concept vocabulary and stops being able to handle novel inputs the way an AR model does — where every input token informs every output token through direct attention. Generation must not be limited to concepts seen in training; it must diversify, understand the user's actual prompt, and stay aligned to it. The architecture must guarantee four properties to do this.

### Property 1 — Full input coverage via dual-path Emit

The Emit sublayer cross-attends to **both** the concept register `C` *and* the surface register `S` containing the original input tokens. At every output token, three attention paths run simultaneously:

- **Token-level self-attention** over the output history — preserves AR-style fluency, identical to a normal LM.
- **Concept cross-attention** to `C` — provides reasoning content from settled concepts.
- **Direct surface cross-attention** to `S` (the input tokens) — provides raw input fidelity. Every input token can influence every output token, exactly as in a transformer LM.

The third path is the critical one. **Concepts are not the only channel from input to output.** The model keeps direct, unmediated line-of-sight to the user's actual words. So:

- Stylistic nuance, exact phrasing, rare tokens, and details that don't fit cleanly into concept slots still affect output.
- If concepts are undertrained or the input is out-of-distribution, the surface path still carries the input through to generation.
- The AR property "every input token informs every output token" is preserved structurally — it does not depend on concepts being good.

This is a deliberate redundancy: the model has a *concept channel* and a *surface channel* in parallel. Either alone could produce output; together, they produce grounded *and* faithful output.

### Property 2 — Novel concepts via composition, not lookup

Slots do not hold a finite vocabulary of trained concepts. Each slot's content is a continuous high-dimensional vector that can land anywhere in concept space. Two mechanisms make novel concepts cheap:

- **Compositional binding via TPR**: novel role-filler combinations are first-class. "Blue jellyfish filing taxes" is just bind(blue, jellyfish) in a noun-slot and bind(filing, taxes) in a verb-slot. The composition has never been seen, but every component has, and TPR composition is what the architecture is *for*.
- **Smooth interpolation in slot space**: a concept halfway between two trained attractors is just a point in the convex region between them. The energy landscape is continuous, so novel inputs land in interpolated states instead of failing.

So a prompt like *"explain ant-colony social dynamics as if it were a Renaissance court"* requires no new concept. The model represents ant-colony, social-dynamics, Renaissance-court, and analogy-mapping as separate slot contents and lets Resonate compose them. The architecture *expects* novel compositions because composition is the entire point of having a TPR-bound slot register.

### Property 3 — Faithfulness loss ties output to input

During Stages D, E, F, an explicit loss term enforces input-output alignment:

```
L_faithful = α₁ · contrastive_align(C_input, C_output)
           + α₂ · reconstruct(input_tokens | C_emitted)
```

- The first term keeps the concept register the model uses to *generate* close to the concept register the input *induced*. They must encode the same meaning, not drift.
- The second term: re-encode the generated output through Absorb, get its concept register, and require that register can predict the original input. The generated answer's concepts must be able to explain the question.

This is faithful-CoT in spirit but stronger: the *answer's* concept content must reconstruct the *input's* meaning, not just internal reasoning steps. It directly prevents the "plausible but unrelated" failure mode that token-LMs slide into when they're uncertain.

### Property 4 — Resonate retains input access throughout reasoning

During iterative Resonate (latent reasoning loop), the surface register `S` is **not** dropped. Every Resonate iteration cross-attends back to `S`:

```
C_{t+1} = SelfAttn(C_t) + CrossAttn(C_t, S_input)
```

This means:

- Reasoning does not leave the input behind. At every iteration, concepts re-ground against the user's actual prompt.
- If reasoning starts drifting toward plausible-but-irrelevant attractors, input attention pulls it back.
- A 32-iteration Resonate cannot wander off into tangents — every iteration is anchored.

Without this, deep reasoning loops could end up answering a different question than was asked. This is a structural defense, not a tuning hyperparameter.

### Energy gate as uncertainty signal

If the energy gate fails to converge for an input (energy stays high after the iteration cap), this is a real signal: the input is outside the model's well-grounded territory. Two valid behaviors fall out naturally:

- **Express uncertainty**: emit a hedged answer ("I'm not sure, but my best guess is...") rather than a confident one.
- **Ask a clarifying question**: emit a question instead of an answer when concepts haven't settled.

Both are trained at Stage E with a small synthetic corpus of "uncertain answer" templates conditioned on high terminal energy. This is the architectural defense against confident hallucination on out-of-distribution prompts — the model *knows* when it's reasoning poorly.

### What this guarantees, end-to-end

| Concern (yours) | Architectural answer |
|---|---|
| Generation must understand the input prompt | Absorb populates `C` from the full input + dual-path Emit attends back to `S` directly |
| Generation must not be limited to trained concepts | Continuous slot space + TPR composition + smooth interpolation = novel concepts at inference |
| Generation must diversify thinking | Multiple Resonate iterations explore concept compositions; energy descent allows search, not greedy paths |
| Generation must align to user input | Faithfulness loss + dual-path Emit + Resonate's retained input access + uncertainty signaling |
| Worst-case fallback (concepts useless) | Surface path through `S` still produces AR-style generation; model degrades gracefully to baseline LM |

Generation does not depend on the concept register being perfect. The dual-path Emit means even an undertrained concept register cannot break input-grounded generation. The faithfulness loss prevents drift even when the concept register is overtrained. Together, these properties ensure the model behaves *at least* as well as a same-size AR LM on input understanding, and better when concepts contribute usefully.

---

## Metrics, defined

Loose metric definitions are the easiest way for a research plan to silently fail. Pinning them down:

- **Linear probe accuracy**: train a single linear layer from frozen slot contents to ground-truth concept labels (color, shape, etc.). Held out from training. >80% means slot contents linearly encode the concept.
- **Mutual-Information Gap (MIG)**: for each ground-truth factor, find the two slots most informative about it; gap = (top-1 MI − top-2 MI), averaged across factors, normalized. >0.4 means each factor is dominantly captured by one slot rather than spread across many. (Standard disentanglement metric, Chen et al. 2018.)
- **Compositional held-out probe**: split (color × shape) pairs into seen vs. unseen. Train probe only on seen. Test accuracy on unseen pairs measures whether slot contents *compose*.
- **Causal slot-swap intervention**: take two scenes A and B. Replace one slot in A's `C` with the corresponding slot from B's `C`. Run Emit. Did the prediction change in the way expected by ground truth? Pass if ≥70% of swaps produce the expected change.

These four together resist false-positive "concepts" that look meaningful to a probe but aren't actually composable or causal.

---

## Stage A failure diagnostics

If Stage A's gate fails, the failure mode tells you what to fix. Probable patterns:

| Symptom | Likely cause | First thing to try |
|---|---|---|
| Probe accuracy low across all factors | Slots aren't capturing factor structure at all | Increase Sinkhorn routing entropy weight; decrease `K`; verify VICReg is actually firing |
| Probe high but MIG low | Information is in slots but spread across many | Stronger covariance penalty in VICReg; sparser routing; smaller `d_filler` |
| MIG good but compositional split fails | Slots specialize but don't compose | Switch from soft to hard slot routing during eval; check TPR role basis is being used |
| Causal swap fails | Slots are correlated by coincidence, not causally separable | Add explicit slot-consistency loss across augmentations; the slot identity isn't stable enough |
| Training loss good, all probes bad | Posterior collapse — slots are bypassed | Information bottleneck β too low; reduce slot dimensionality |
| Loss diverges or oscillates | Optimization instability | Smaller LR; warmup; gradient clipping; check VICReg invariance term sign |

Each row corresponds to a specific 1–2 day debug experiment with a known fix to try. Treat Stage A as a research run with a well-defined diagnostic loop, not a single-shot training job.

---

## The concept-stability regularizer (Stage F's load-bearing piece)

When Stage F exposes the model to messy real text, the concept register *will* try to drift to fit it. Without a regularizer, this is the catastrophic forgetting of foundations that motivated the whole curriculum. Concretely:

Let `C_E` be the concept register's distribution at the end of Stage E (foundation locked-in). During Stage F, add to the loss:

```
L_stability = λ_1 · D_KL( p(C_F) ‖ p(C_E) )            # slot-content distribution drift
            + λ_2 · ‖A_F − A_E‖²_F                      # routing-attention drift
            + λ_3 · prober_loss(C_F → ground_truth_A–C) # explicit downstream-task anchoring
```

- `λ_1` keeps slot-content statistics close to Stage E.
- `λ_2` keeps the routing pattern (which inputs go to which slot) close to Stage E.
- `λ_3` is the strongest defense: keep a frozen probe head from Stage A–C around, and penalize Stage F drift if the probe's accuracy drops on a held-out CLEVR-style test set.

The third term is what literally makes the foundation load-bearing — concepts can extend to handle new content (Wikipedia, code) but can't reorganize away from what they were in Stages A–C. This is the elastic-weight-consolidation idea (Kirkpatrick et al. 2017) applied at the concept-register level rather than the parameter level.

Tuning the `λ`s is empirical. Start with `λ_1 = λ_2 = 0.1, λ_3 = 1.0`. If concept probes drift, raise. If the model can't absorb new domains, lower.

---

## Comparison to closest existing work

| Approach | Grounded? | Latent reasoning? | Fluent generation? | Curriculum? |
|---|---|---|---|---|
| GPT / Llama | No (concepts emerge accidentally) | No (token CoT only) | Yes | No |
| Coconut | No | Yes (continuous thoughts) | Yes (degraded) | Two-stage CoT-replace only |
| Huginn (recurrent depth) | No | Yes (depth recurrence) | Yes | No |
| V-JEPA 2 / 2.1 | Yes (visual) | Yes (action prediction) | No (no language head) | Implicit (video → action) |
| Slot Attention / SLATE | Yes (object-level) | Limited | No | No |
| LV-EBM / Kona | Partial | Yes (energy descent) | Limited | No |
| **This proposal** | **Yes (multi-modal, by design)** | **Yes (Resonate + energy gate)** | **Yes (Emit + warm-started LM)** | **Yes (Stages A–F)** |

The novelty isn't any single row's content; it's the cell that says *yes* in every column. No prior work hits all four. The curriculum is the mechanism that makes that combination *trainable*.

---

## Compute budget per stage (rough order of magnitude)

| Stage | What | A100-hours | Cost @ $1/hr | Realistic timeline |
|---|---|---|---|---|
| A | Concept Physics | ~100 | ~$100 (or your existing 5-day allocation) | 1 week |
| B | Concept Dynamics | ~300 | ~$300 | 2 weeks |
| C | Compositional Reasoning | ~400 | ~$400 | 2–3 weeks |
| D | Language Grounding | ~500 | ~$500 | 3–4 weeks |
| E | Compositional Language | ~800 | ~$800 | 1–2 months |
| F | Real-world enrichment | ~1500–3000 | ~$1500–3000 | 2–3 months |

**Total realistic budget**: ~$3–5K and ~6–9 months of full-time work, *if every gate passes on the first try.* If not, multiply by 1.5–2x. Still feasible for a solo researcher with a small grant or savings buffer. This is the regime your constraints sit in.

The point of doing Stage A first is that **if it fails, you've spent ~$100 / 5 days finding out, not $5K / 9 months.**

---

---

## Deployment path: Creor as first surface, foundation preserved

Your stated business goal: deploy this model in **Creor**, your existing IDE in the Cursor / Windsurf / Copilot space, and later expand to other domains. Code is a strong first surface — *if* the foundation discipline above holds. If it doesn't, Creor becomes another autocomplete clone with extra steps.

### Why code is structurally well-suited (without being the right starting point)

Code aligns with this architecture better than natural language because it is **structured, compositional, causal, and verifiable**:
- recursion, async flow, state machines, dependency injection, event systems, graph traversal, type composition — these are *concept types*, not surface patterns. They are exactly the things slot-bound concept attractors are designed to represent.
- correctness signals are exact (compile, run, test). Concept formation has measurable feedback that natural language can't give.
- composition is explicit (functions, modules, packages). Compositional generalization can be tested directly.

So eventually, code is one of the *best* domains for this architecture to operate in. But:

### Why code cannot be in the foundation stages

The whole reason this curriculum works is that concepts are formed before being labeled with language. Code looks structured but it is still language — it has surface conventions, naming biases, library idioms, framework-specific shortcuts. A model that learns concepts *from code* would learn code-shaped concepts and fail to transfer. That is exactly the failure mode you've been correcting throughout this conversation, applied to one specific domain.

**Foundation discipline restated:**
- Stages A–C: visual / symbolic synthetic data only. No code.
- Stages D–E: structured language including simple symbolic compositions, but not real code.
- Stage F: code enters as *one* modality alongside books, Wikipedia, math, conversation. Roughly equal weighting. Code is in the diet, not the substrate.

Only after Stage F passes does code-specific specialization begin.

### Stage G — Code as a first-class modality (the Creor-shippable artifact)

**Goal**: extend the concept register's content distribution to code-specific concepts (functions, classes, types, modules, dependencies, control flow), without disturbing the general foundation built in A–F.

**Mechanism**:
- **AST-aware Absorb**: code is parsed by tree-sitter into ASTs. AST nodes contribute structured patches into the surface register `S` alongside raw token patches. Slots can align to AST subtrees, not just to text spans. Tree-sitter is mature, fast, and handles every language Creor supports.
- **Code-only curriculum**: starts from primitives (single-function definitions, type composition, simple recursion) and builds up (multi-file projects, refactors, architectural patterns). Procedurally generated where possible (you can synthesize unlimited correct code from grammar rules).
- **Concept-stability regularizer at maximum strength.** This is where Stage F's regularizer earns its keep — you do *not* let Stage G drift the concept inventory away from what was learned in A–F. If it drifts, the foundation is lost and you've reverted to a code LM.

**Datasets**:
- Filtered subsets of The Stack (quality + license filtered).
- Real Creor user telemetry, with consent — accept/reject signals on completions are exceptionally high-value training data.
- Synthetic codebases generated to teach specific architectural concepts (a microservices migration, a state-machine refactor, an async refactor).

**Gate**: code completion accuracy on held-out code matches a same-size pure code LM baseline, *while* concept probes from Stages A–F remain within 5pt of pre-Stage-G accuracy. **Both conditions must hold.** If completion is great but probes drop, foundation has been corrupted; you've shipped a code LM, not a grounded reasoner.

### Stage H — Architectural reasoning over real codebases (where Creor becomes structurally different)

This is where Creor's competitive moat lives. Token-completion copilots cannot reach this stage by scaling, because they have no concept register to specialize.

**Mechanism — codebase-level concept memory**:
- Maintain a **persistent concept graph per Creor project**, populated by indexing. Slots correspond to architectural concepts (services, modules, data flows, types, dependencies, contracts).
- When a user opens a project in Creor, the concept register is initialized from that graph instead of being empty.
- Architectural reasoning (refactor, dependency-impact analysis, migration planning) happens by iterating Resonate over the project-level graph. The energy gate stops when the architectural plan settles.
- Code emission via Emit consults this graph through cross-attention. Generated code is therefore consistent with project architecture, not just statistically plausible.

**Capabilities this enables, mapped to your list**:

| Capability you described | How Stage H delivers it |
|---|---|
| Deep code understanding | Concepts (architecture, data flow, dependencies) are first-class slots, not implicit token patterns |
| Real project memory | Persistent concept graph per project — categorically different from RAG over chunks |
| Better refactoring | Schema change → Resonate propagates through concept register → Emit produces consistent multi-file edits |
| Long-context coding | Architectural concepts persist as slots even when surface tokens scroll out of context |
| Causal debugging | Error → traverse causal slot relations → root-cause hypothesis. Concrete instance of energy-descent reasoning |
| Autonomous coding agents | "Plan in concepts, then emit" is the natural execution model — Resonate iterates the plan, Emit writes the code |

**Gate**: an architectural-task evaluation set (refactor across files, migrate a pattern, identify cross-module impact, propose service boundary) shows margin over a same-size code LM that lacks the persistent concept graph. The margin is the part of the Creor product pitch that competitors cannot replicate by scaling tokens.

### Stage I and beyond — other domains, cheaply

The point of foundation discipline is that expansion is *not* a new training program. To deploy this in medicine, law, scientific writing, or conversational support, you reuse the same machinery as Stage G–H with domain-specific concept seeding and curriculum content. Each new domain costs roughly Stage G's budget, not Stage A–F's. The foundation does the heavy lifting once and is amortized across all future surfaces.

### The deployment timeline you can actually commit to

| Phase | Status of Creor | What ships |
|---|---|---|
| Today through Stage A passing (~1 week) | Existing Creor with normal LLM | No model change |
| Stages B–F (~6–9 months) | Existing Creor with normal LLM | No model change yet — research period |
| Stage G shippable (~9–12 months from now) | Creor with grounded code completion | First version of the Creor-specific model |
| Stage H shippable (~12–18 months from now) | Creor with architectural reasoning | The differentiated product |
| Stage I (later) | Other domains | Optionality on the foundation investment |

Concretely: **do not promise users any of this for at least 9 months.** Keep Creor's current LLM running through that period. Sliding the foundation timeline to ship faster is exactly how Stage G becomes a code LM with no Stage H advantage.

### What you should *not* do (foundation hygiene under business pressure)

- Don't put code in Stages A–E to "get a head start." Code looks structured, but it is still language and will bias concept formation toward code-shaped patterns that won't generalize.
- Don't quietly increase code's weight in Stage F's mix. Stage F's value is that code is *one* of many modalities; over-weighting it collapses to early specialization.
- Don't let Creor product asks ("can you train a model that does X for our users this quarter?") leak into foundation stages. The foundation discipline is what makes the Stage H differentiator possible. Trade it away and you have nothing.
- Don't ship Stage G if its gate fails the concept-probe condition, even if completion accuracy is great. That's the gate that protects the foundation.

The leverage of doing this correctly: the same model that ships in Creor (Stage G–H) becomes the substrate for Stage I across any other domain you want to enter, at low marginal cost. The leverage of doing it incorrectly: you build a code LM that is structurally indistinguishable from what Cursor and Windsurf already have, and the entire research investment is sunk.

---

## Stage A implementation checklist

The deliverables you need to write or fork for the 5-day run:

1. **Repo skeleton at `/Users/chintu/TinyTest`**: `git init`, `pyproject.toml`, PyTorch ≥ 2.2, basic project structure.
2. **Slot encoder**: fork [Slot Attention reference](https://github.com/google-research/google-research/tree/master/slot_attention), adapt to take image patches → K slots.
3. **TPR binding layer**: not off-the-shelf — write it. ~50 lines: roles as a learned `(d_role, d_role)` basis, fillers as `(K, d_filler)`, bind via einsum. [Schlag & Schmidhuber 1808.03578](https://arxiv.org/abs/1808.03578) is the closest reference implementation.
4. **VICReg loss**: standard PyTorch; ~30 lines or use existing implementation from VICReg repo.
5. **Sinkhorn routing**: there are good open implementations; pick one and inline it.
6. **Stage A objective**: masked patch reconstruction in slot space + VICReg + slot-consistency under augmentation.
7. **Datasets**: hook up dSprites (npz), 3D Shapes (h5), CLEVR (image+JSON metadata). Probably the most engineering time goes here.
8. **Evaluation harness**: implement the four metrics above as automated scripts.
9. **Train script**: AdamW, cosine LR, single-GPU. Should fit in <500 lines.
10. **Eval cadence**: run all four metrics every 5K steps; auto-stop if no improvement for 50K steps.

Total code estimate: ~2K–3K lines of PyTorch including all losses, eval, data loaders. One person can write this in 2–3 days, leaving 2–3 days for actually training and iterating.

---

## Software / hardware stack (no surprises)

- **PyTorch ≥ 2.2** with `torch.compile` for the BCS layer.
- **Data**: dSprites (~300MB), 3D Shapes (~1GB), CLEVR (~17GB images + metadata). Cache locally; avoid re-downloading.
- **Logging**: Weights & Biases (free tier is enough) or TensorBoard. Log all four metrics every checkpoint.
- **Single A100 80GB**: enough for the 50–80M model. Batch size ~256, grad accumulation if needed.
- **Reproducibility**: pin random seeds; run Stage A 3× to characterize variance before declaring the gate passed/failed.

---

## What goes on the A100 in the next 5 days

Stage A. Just Stage A. Nothing else.

Specifically:
1. Set up workspace at `/Users/chintu/TinyTest` (currently not a git repo). Pick PyTorch, slot-attention base, TPR binding implementation.
2. Implement Stage A model (~50M params): slot encoder + TPR concept register + masked-prediction loss + VICReg + Sinkhorn routing.
3. Implement Stage A gate metrics (linear probe, MIG, compositional split, causal swap) as automated evals.
4. Train on dSprites first (simplest), then 3D Shapes, then CLEVR. Each lift is a sanity check that the gates monotonically improve as data complexity rises.
5. **At the end of 5 days, you have a single answer: did Stage A's gate pass?**

If yes — your curriculum hypothesis is alive, the architecture is alive, you have a publishable result on object-centric concept formation under TPR + VICReg + Sinkhorn, and you're ready to go to Stage B with Phase D borrowed compute.

If no — you've narrowed where the problem lives. Probable next moves are: scale K (slot count), tune VICReg coefficients, or replace TPR with HRR. Every failure is informative because the dataset has known ground truth.

---

## Honest residual risks (still real, still cannot be hand-waved)

These do not go away just because the curriculum is right. The curriculum *reduces* their probability — it does not eliminate them.

1. **Synthetic-to-natural transfer cliff.** Concepts that emerge cleanly in CLEVR may not survive Stage F. The concept-stability regularizer is the defense, but its strength has to be tuned empirically.
2. **Curriculum amplifies over-specialization.** A child who only ever saw blocks may struggle with liquids. Your synthetic stages must be diverse enough that the concept substrate isn't biased toward one ontology. CLEVR-only training will produce a CLEVR-specialized model.
3. **Stage gates may pass for the wrong reason.** Probes can read out concept-like structure even from random representations under specific conditions. Causal-intervention tests are the strongest defense against false positives, which is why they're in every gate.
4. **The curriculum is long.** Stages A–F is realistically a year of research, not a weekend. Stage A is a 5-day experiment; everything after is open-ended.
5. **Distillation-as-curriculum requires a teacher whose concepts roughly align with yours.** If the teacher's implicit concept inventory is wildly different (because it was trained on internet noise), its generated curricular content may pull yours back toward the noise. Mitigation: filter teacher outputs through your model's existing concept register and discard outputs that don't parse cleanly.

---

## Why this is worth doing

The shift from *language-first* to *concepts-first* is not new as an idea (LeCun has argued this for years; Smolensky argued this in the 1990s; Piaget argued it in the 1920s). What's missing in the literature is **a concrete curriculum, with concrete datasets and concrete gates, executed end-to-end on a model architecture that supports it.** Every paper either (a) uses concept-centric training but doesn't scale to language, or (b) trains on language and hopes concepts emerge.

Doing the curriculum end-to-end, even partially, even at a small scale, would be a substantive contribution. The 5-day Stage A experiment is the first defensible step.

## Sources

- [I-JEPA blog (Meta)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) · [V-JEPA repo](https://github.com/facebookresearch/jepa)
- [Slot Attention — 2006.15055](https://arxiv.org/abs/2006.15055) · [SLATE — 2110.11405](https://arxiv.org/abs/2110.11405)
- [VICReg — 2105.04906](https://arxiv.org/abs/2105.04906)
- Smolensky 1990 (Tensor Product Representations) · [Schlag & Schmidhuber TPR revival 1808.03578](https://arxiv.org/abs/1808.03578)
- [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) · [CLEVRER 1910.01442](https://arxiv.org/abs/1910.01442)
- [BabyAI 1810.08272](https://arxiv.org/abs/1810.08272) · [MiniGrid](https://github.com/Farama-Foundation/Minigrid)
- [SCAN 1711.00350](https://arxiv.org/abs/1711.00350) · [COGS 2010.05465](https://arxiv.org/abs/2010.05465) · [ConceptARC 2305.07141](https://arxiv.org/abs/2305.07141)
- [Coconut — 2412.06769](https://arxiv.org/abs/2412.06769) · [Huginn — 2502.05171](https://arxiv.org/abs/2502.05171)
- [Phi-2 / textbook-quality data](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) · [BeyondWeb (synthetic data scaling)](https://www.datologyai.com/blog/beyondweb)
- [Curriculum learning for LLMs — 2601.21698](https://arxiv.org/html/2601.21698v1) · [Compositional generalization survey](https://arxiv.org/abs/2305.07141)
- [Survey on Latent Reasoning — 2507.06203](https://arxiv.org/abs/2507.06203)
