# Research Progress & Ideas Tracker

## Completed

### Infrastructure
- Per-epoch re-randomization of task splitting, masking, and NL fraction
- Pair task groups (aspect+sentiment, aspect+polarity, etc.) for task splitting
- Full quad support: ACOS JSONL loader, all task combinations (4 singles, 6 pairs, 4 triples, 1 quad)
- Cosine LR scheduler (configurable: cosine/linear/constant)
- Label smoothing (configurable, default 0.05; 0.1 confirmed worse)
- Early stopping re-added as configurable (patience 0 = disabled)
- Consistent IMPLICIT handling across all data loaders (ASTE, silviolima, ACOS)
- NONE for missing categories (vs IMPLICIT for implied-but-unstated aspects/opinions)
- data.filter_implicit flag — strips IMPLICIT from train/val/test at load time
- data.infer_implicit flag — uses IMPLICIT:{term} or implied aspect NL template
- test.eval_implicit_split flag — separate metrics for implicit vs explicit triplets
- Date-stamped experiment directories (experiments/YYYY-MM-DD/)
- Fixed DDP directory race condition (env var approach)
- Incremental history saving to disk (survives DDP)
- Experiment name printed at start of each run
- Removed DEBUG prints from validation
- Config merge: tasks_partition, scopes, datasets now replace instead of deep-merge

### Output Formats
- Natural-language output templates for all task combinations
- Prompt format: Task / Input / Syntax (optional) / Output: structured|natural language
- NL parser for evaluation (regex-based template matching)
- Implicit aspect NL template: "the implied aspect is {term}, described as ..."
- Structured implicit format: IMPLICIT:{term}

### Syntax Enrichment
- 4 modes: dep-tree, dep-compact, dep-inline, pos-inline
- dep-compact: content words only, compact edge format (word->head:dep), separate Syntax line
- dep-inline/pos-inline: inline annotations replacing input tokens
- Punctuation filtered from all modes
- Cached via spaCy en_core_web_sm at data load time

### Augmentation
- Masking: fixed mask_target default to false, per-epoch re-randomization
- Duplicate augmenter for controlled ablation
- LLM paraphrase scaffolding: loader, selection logic, ollama/llama3.1 generation function
- Mock generation function for testing pipeline
- Curriculum learning: waypoint-based interpolation of task partitions across epochs

### Masking
- Fixed mask_target bug: default now false (real aspect in target, not sentinel)
- Per-epoch re-randomization of which examples get masked

## Completed Experiments & Results

### Full NL Output (2026-05-01, ASTE, 30 epochs, best results)
| config | Epochs | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|---|
| nl-baseline | 30 | 0.7166 | 0.5211 |
| nl-split | 25 | 0.7161 | 0.5249 |
| nl-dep-compact | 30 | 0.7083 | **0.5414** |
| nl-split-dep-compact | 24 | 0.7091 | 0.5116 |

Key finding: NL output format is the single biggest improvement for OOD (+7 points over structured baseline). dep-compact adds +2 more OOD on top. Combining split+dep-compact+NL hurts — too many techniques dilute the signal.

### Beam Search (2026-05-01, ASTE, beam=4 at test time only)
| Config | Rest14 (ID) | | Laptop14 (OOD) | |
|---|---|---|---|---|
| | beam=1 | beam=4 | beam=1 | beam=4 |
| nl-baseline | 0.7166 | 0.7212 (+0.5) | 0.5211 | 0.5226 (+0.2) |
| nl-split | 0.7161 | 0.7168 (+0.1) | 0.5249 | 0.5213 (-0.4) |
| nl-dep-compact | 0.7083 | 0.7125 (+0.4) | **0.5414** | 0.5304 (-1.1) |
| nl-split-dep-compact | 0.7091 | 0.7227 (+1.4) | 0.5116 | 0.5308 (+1.9) |

Beam search gives small ID gains across the board. OOD is mixed: hurts nl-dep-compact (-1.1), helps nl-split-dep-compact (+1.9). nl-split-dep-compact recovers from weakest to competitive OOD with beam=4. Not a universal win — config-dependent.

### Curriculum Learning (2026-05-01, ASTE, 30 epochs, full NL)
| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| cur-overlap | 0.7156 | 0.5094 |
| cur-overlap-dep | 0.7178 | 0.5135 |
| cur-fast-ramp | 0.7248 | 0.5034 |
| cur-fast-ramp-dep | 0.7058 | 0.5227 |
| cur-sandwich | 0.7281 | 0.5166 |
| cur-sandwich-dep | 0.7148 | 0.5014 |

Curriculum learning doesn't help OOD. No curriculum config beats nl-dep-compact (0.5414). +dep variants consistently help OOD relative to non-dep counterparts, confirming dep-compact as the strongest OOD lever. Negative result for dissertation.

### NL Fraction Experiments (2026-05-01, ASTE, 20 epochs)
| config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| baseline (structured) | 0.6798 | 0.4296 |
| baseline-nl10 | 0.7041 | 0.4354 |
| baseline-nl30 | 0.7004 | 0.4374 |
| baseline-nl50 | 0.7135 | 0.4575 |
| split-nl10 | 0.7135 | 0.4502 |
| split-nl30 | 0.7020 | 0.4511 |
| split-nl50 | 0.7074 | 0.4567 |

NL training consistently helps both ID and OOD. Split+NL is more efficient (split-nl10 gets ~0.45 OOD with only 10% NL).

### Syntax Enrichment (2026-05-01, ASTE, 20 epochs)
| config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| baseline (no syntax) | 0.6798 | 0.4296 |
| baseline + dep-inline | 0.6546 | 0.3926 |
| baseline + pos-inline | 0.6391 | 0.3851 |
| baseline + dep-compact | 0.7160 | 0.4216 |
| split-nl10 + dep-inline | 0.6543 | 0.4095 |
| split-nl10 + pos-inline | 0.6497 | 0.3958 |
| split-nl10 + dep-compact | 0.6992 | 0.4609 |

Inline syntax hurts across the board. dep-compact (separate Syntax line) helps ID when alone, helps OOD when combined with split+NL.

### ACOS Quad Comparison vs STAR (2026-04-30)
| | ACOS-Laptop | ACOS-Rest16 |
|---|---|---|
| Our params (flan-t5-base, cosine, ls=0.05) | 42.53 | 55.56 |
| STAR-like params (t5-base, constant, no ls) | 41.90 | 54.27 |
| STAR paper (full framework) | 45.15 | 61.07 |

### Silviolima OOD (2026-04-30, implicit filtered)
- split-10-90 and split-50-50 tied best (avg F1 0.2258 vs baseline 0.2135)
- Masking with replace didn't help on top of 50-50 split

### Optimizer Sanity (2026-04-30)
- ls=0.1 hurt OOD vs ls=0.0 (0.4269 vs 0.4493). Settled on ls=0.05.
- Early stopping + cosine is bad — cosine needs full training.

## In Progress
- None currently running

## Future Work (Prioritized)

### High Priority
- LLM paraphrasing augmentation (scaffolding done, generation function ready)
  - Refine prompt to avoid domain hallucination
  - Generate full restaurant dataset paraphrases
  - Test with infer_implicit on ASTE
- Beam search at test time — re-test best checkpoints with num_beams=4
- Curriculum learning experiments — use waypoint interpolation to combine split + dep-compact sequentially

### Medium Priority
- Longer training for dep-compact variants (40+ epochs)
- Lighter split ratios (10-90) with dep-compact + NL
- STAR-style data multiplication vs partition comparison
- MvP-style majority voting at inference
- Test on silviolima with NL output format

### Exploratory
- Attention-based learned masking
- Two-phase training (split first, then dep-compact from checkpoint)
- Contextual soft F1 extension

## Related Papers (Key)
- STAR (Xie et al., 2025): task decomposition for ASQP, validates our approach, no OOD eval
- Paraphrase (Zhang et al., 2021): foundation for generative ABSA output format
- MvP (Gou et al., 2023): multi-view prompting with order permutations
- BGCA (Deng et al., 2023): bidirectional generative cross-domain ABSA
- MELM (Zhou et al., 2021): entity masking for NER augmentation
- Full list in articles/references.md


## Natural-Language Output Templates

Config: `data.natural_language_fraction: 1.0` for full NL, or fractional for mixed training.
Test: `test.output_format: "natural-language"` to evaluate with NL output.

### Singles
- aspect: "the aspect being discussed is {aspect}"
- sentiment: "the opinion expressed is {sentiment}"
- polarity: "the overall sentiment is {polarity}"
- category: "the category being discussed is {category}"

### Pairs
- aspect+sentiment: "{aspect} is described as {sentiment}"
- aspect+polarity: "the opinion about {aspect} is {polarity}"
- aspect+category: "{aspect} falls under the category {category}"
- sentiment+polarity: "the opinion {sentiment} conveys a {polarity} sentiment"
- sentiment+category: "the opinion {sentiment} is about the category {category}"
- polarity+category: "the sentiment toward {category} is {polarity}"

### Triples
- aspect+sentiment+polarity: "{aspect} is described as {sentiment}, expressing a {polarity} sentiment"
- aspect+sentiment+category: "{aspect} is described as {sentiment}, under the category {category}"
- aspect+polarity+category: "the {polarity} opinion about {aspect} falls under {category}"
- sentiment+polarity+category: "the opinion {sentiment} conveys a {polarity} sentiment about {category}"

### Quad
- aspect+sentiment+polarity+category: "{aspect} is described as {sentiment}, expressing a {polarity} sentiment about {category}"

### Implicit variant (Option B)
- Explicit: "pizza is described as delicious, expressing a positive sentiment"
- Implicit: "the implied aspect is pizza, described as delicious, expressing a positive sentiment"

### Notes
- Multiple annotations joined with " ; "
- Parser splits on " ; " and uses regex template matching
- Implicit detected by "the implied aspect is" prefix in NL, "IMPLICIT:" prefix in structured

## Implicit Inference Design (Scaffolding Implemented)

- `data.infer_implicit: true/false` — controls output format for IMPLICIT aspects with known original terms
- NL: "the implied aspect is {term}, described as ..." / Structured: `[IMPLICIT:{term}, ...]`
- `aspect_original` field in canonical format stores the known original term
- `data.augmentation.llm_paraphrase` — points to pre-generated file, fraction, replace
- Generation via ollama + llama3.1:8b (tested, prompt needs refinement)
- Mock generation function available for pipeline testing


## LLM Paraphrase Generation Notes

### Prompt Development
- Tested llama3.1:8b and gemma2:27b via ollama
- llama 8B: tends to substitute synonyms/hypernyms, changes domain, poor constraint following
- gemma 27B: much better at following constraints, keeps sentiment intensity
- Best prompt includes: no-synonym rule, keep original opinion words, contextual clues requirement, no-new-aspects rule, few-shot good/bad examples
- Remaining issue: tension between contextual clues (which tend to introduce aspect-like nouns) and avoiding new aspects

### Multi-Triplet Strategy (Not Yet Implemented)
- 53% of restaurant ASTE sentences have single triplet (1447/2728) — safe to paraphrase
- 47% have multiple triplets — paraphrasing one aspect risks changing/dropping others
- Approach 1 (recommended first): only paraphrase single-triplet sentences
- Approach 2 (future): paraphrase one aspect in multi-triplet sentences, verify others remain unchanged via post-generation check
- Verification: after LLM generation, check that non-paraphrased aspects still appear in the rewritten sentence; reject if missing


## ASTE Benchmark Comparison (In-Domain, SemEval)

### Discriminative (BERT-based)
| Method | 14Res | 14Lap | 15Res | 16Res |
|---|---|---|---|---|
| GTS (2020) | 70.92 | 59.46 | 62.53 | 68.71 |
| Span-ASTE (2021) | 72.89 | 62.40 | 64.45 | 71.85 |
| BDTF (2022) | 74.35 | 62.59 | 66.12 | 72.27 |
| BTF-CCL (2025, SOTA) | 75.88 | 63.29 | 67.68 | 73.80 |

### Generative (T5/BART-based)
| Method | 14Res | 14Lap |
|---|---|---|
| GAS (2021) | ~70 | ~58 |
| Paraphrase (2021) | ~70 | ~58 |
| **Ours (nl-baseline)** | **71.66** | — |
| **Ours (nl-dep-compact)** | **70.83** | — |

### OOD (Restaurant → Laptop)
| Method | 14Res→14Lap |
|---|---|
| Chia et al. generative avg drop | ~14.6 points |
| Chia et al. discriminative avg drop | ~16.8 points |
| **Ours (nl-dep-compact)** | **54.14** (from 70.83 ID, drop=16.7) |
| **Ours (nl-baseline)** | **52.11** (from 71.66 ID, drop=19.6) |

Note: Our OOD numbers are Restaurant14+15+16 → Laptop14 (train on 3 restaurant datasets, test on laptop). Standard benchmarks train and test on same domain. No direct OOD comparison exists for this exact setup.


## Dissertation

Working title: "A Systematic Empirical Study of Training Strategies for Out-of-Domain Generalisation in Generative Aspect-Based Sentiment Analysis"
