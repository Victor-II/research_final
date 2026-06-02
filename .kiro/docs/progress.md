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

**Note:** Experiments from 2026-04-30 to 2026-05-02 used multi-restaurant training (Rest14+15+16). This setup was abandoned on 2026-05-19 after discovering 80% overlap between Rest15 test and the combined train set. Results are valid as relative comparisons but absolute numbers are inflated. All experiments from 2026-05-20 onward use Rest14-only training.

### Full NL Output (2026-05-01, ASTE, 30 epochs, multi-restaurant train)
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

### Aspect Masking (2026-05-02, ASTE, 30 epochs, full NL, replace mode)
| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| nl-dep-compact (ref) | 0.7083 | **0.5414** |
| nl-dep-mask10-replace | 0.7176 | 0.5074 |
| nl-dep-mask25-replace | 0.7187 | 0.4882 |
| nl-dep-mask50-replace | 0.7041 | 0.5101 |
| nl-mask25-replace | 0.7126 | 0.5110 |

Simple masking hurts OOD compared to NL and dep-compact baselines. Masking + dep-compact interfere (nl-dep-mask25 worst at 0.4882, while nl-mask25 without dep gets 0.5110). Replacing training examples with masked versions removes real aspect context the model needs. Negative result.

### Decoding Strategies (2026-05-02, ASTE, nl-dep-compact checkpoint)
| Strategy | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| greedy (baseline) | 0.7083 | **0.5414** |
| beam=4 | 0.7125 | 0.5304 |
| sample t=0.7 | 0.6986 | 0.4770 |
| sample t=0.9 | 0.6718 | 0.4457 |
| vote5 t=0.8 thresh=2 | 0.7104 | 0.5088 |
| vote5 t=0.8 thresh=3 | 0.7101 | 0.5198 |
| diverse beam (12b/4g) vote | 0.6690 | 0.4752 |
| constrained (logits boost) | 0.6994 | 0.4833 |

Greedy decoding remains optimal. All alternatives hurt OOD. Temperature sampling adds too much noise. Voting can't recover from diverse/noisy candidates. Constrained decoding interferes with the model's learned template adherence. Contrastive search incompatible with T5 in transformers >= 4.50.

### nlpaug Augmentation
Tested previously on structured output format — synonym, contextual, and other nlpaug methods were not helpful for OOD generalisation.

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

### Gemma 4 E4B Inference (2026-05-17, 4.5B params, 4-bit quantized, no training)

0-shot and 6-shot with hybrid demonstration retrieval (3 BM25 + 3 SimCSE per test sentence).

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| 0-shot structured | 0.3297 | 0.1860 |
| 6-shot structured (hybrid) | 0.5639 | 0.3446 |
| 6-shot NL (hybrid) | 0.3631 | 0.2586 |

- Syntax enrichment had zero effect (identical scores with/without) — model ignores it without training
- NL format scores 0 in 0-shot — parser can't handle Gemma's template variations
- 6-shot NL has high precision (0.56) but low recall (0.27) — model is conservative
- Aspect-only F1 is strong: 0.80 ID, 0.64 OOD (6-shot) — bottleneck is opinion span matching
- ACOS 6-shot quad F1: 0.42 ID, ~0 OOD (category taxonomy mismatch across domains)

### RoBERTa Span Extraction (2026-05-17, 125M params, BIO + biaffine pairing)

Naive discriminative baseline: BIO tagging for aspect/sentiment spans, biaffine scorer for pairing, polarity classifier on paired representations.

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| RoBERTa baseline (Rest14+15+16) | 0.4959 | 0.3949 |
| RoBERTa + dep-compact syntax | 0.4023 | 0.2790 |
| RoBERTa Rest14 only | 0.4990 | 0.3743 |

- Syntax as text_pair hurts badly (-10 points) — disrupts word_id alignment for BIO tagging
- Multi-dataset training gives slight OOD gain (0.39 vs 0.37) but no ID improvement
- Well below published discriminative SOTA (Span-ASTE: 0.73, BTF-CCL: 0.76) — those use specialised architectures (GCN, table filling, contrastive learning)
- Our T5 NL approach (0.72 ID, 0.52 OOD) significantly outperforms this naive baseline

### Cross-Method Summary (ASTE, Rest14 / Laptop14)

| Method | Params | Rest14 | Laptop14 |
|---|---|---|---|
| Gemma 4 0-shot | 4.5B | 0.33 | 0.19 |
| Gemma 4 6-shot hybrid | 4.5B | 0.56 | 0.34 |
| Ollama Gemma2 27B 6-shot | 27B | 0.65 | 0.41 |
| Ollama Qwen2.5 32B 0-shot | 32B | 0.54 | 0.36 |
| RoBERTa span (ours) | 125M | 0.54 | 0.38 |
| XLM-R span (ours) | 278M | 0.47 | 0.34 |
| T5 structured baseline | 250M | 0.69±0.01 | 0.44±0.01 |
| T5 structured + dep-compact | 250M | 0.70±0.00 | 0.45±0.01 |
| **T5 NL baseline** | **250M** | **0.72±0.00** | **0.52±0.01** |
| T5 NL + dep-compact | 250M | 0.71±0.01 | 0.51±0.01 |
| T5 NL + split | 250M | 0.73±0.01 | 0.52±0.01 |
| Span-ASTE (published) | ~110M | 0.73 | 0.62 |
| BTF-CCL (published) | ~110M | 0.76 | 0.63 |
| MvP (published) | ~400M | 0.76 | 0.66 |

## In Progress

### T5 OOD Experiments — Rest14 Only (2026-05-20)
Training on Rest14 only, testing on Rest14 (ID), Rest15 (near-OOD), Rest16 (near-OOD), Laptop14 (OOD).

| Config | Rest14 (ID) | Rest15 | Rest16 | Laptop14 (OOD) |
|---|---|---|---|---|
| structured baseline | 0.6980 | 0.5700 | 0.6367 | 0.4205 |
| **nl-baseline** | **0.7265** | **0.6073** | **0.6723** | **0.5430** |
| nl-dep-compact | 0.6995 | 0.5924 | 0.6623 | 0.5174 |
| nl-dep-compact-aux | 0.7146 | 0.6057 | 0.6685 | 0.4807 |
| nl-split | 0.7290 | 0.5946 | 0.7020 | 0.5086 |
| nl-split-dep-compact | 0.7188 | 0.5871 | 0.6557 | 0.5051 |
| nl-pos-compact | 0.6980 | 0.6066 | 0.6806 | 0.5130 |
| nl-template-aug | PENDING RETEST (old results had config bug) |

Key observations:
- NL format remains the biggest lever: +12 points OOD over structured (0.42→0.54)
- dep-compact does NOT help OOD with Rest14-only training (0.54→0.52, -2.5 points)
- nl-split gives best ID (0.729) but slightly lower OOD than nl-baseline
- pos-compact similar to dep-compact: helps near-OOD (Rest15/16) but not far-OOD (Laptop)
- Hypothesis: syntax enrichment benefit is data-size-dependent. With 2728 sentences (3 datasets) it helped; with 1728 (Rest14 only) it hurts.

### Multi-Seed Validation (2026-05-22/23/24, 5 seeds: 42, 123, 456, 789, 1337)

#### Rest14-only training (mean±std, triplet micro-F1)

| Config | Rest14 (ID) | Rest15 | Rest16 | Laptop14 (OOD) |
|---|---|---|---|---|
| structured baseline | 0.6926±0.0081 | 0.5747±0.0093 | 0.6399±0.0089 | 0.4390±0.0119 |
| structured + dep-compact | 0.6991±0.0036 | 0.5779±0.0123 | 0.6546±0.0075 | 0.4473±0.0117 |
| **nl-baseline** | **0.7218±0.0047** | 0.6023±0.0043 | 0.6798±0.0096 | **0.5247±0.0116** |
| nl-dep-compact | 0.7076±0.0060 | 0.5962±0.0056 | 0.6692±0.0039 | 0.5091±0.0053 |
| nl-pos-compact | 0.6987±0.0031 | 0.6006±0.0099 | 0.6716±0.0072 | 0.5055±0.0118 |
| nl-split | 0.7295±0.0054 | 0.5929±0.0104 | 0.6859±0.0086 | 0.5237±0.0110 |

#### Allrest training (Rest14+15+16 → Rest14 ID + Laptop14 OOD)

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| structured baseline | 0.6981±0.0039 | 0.4253±0.0121 |
| structured + dep-compact | 0.6970±0.0059 | 0.4442±0.0048 |
| **nl-baseline** | 0.7135±0.0057 | 0.5162±0.0087 |
| nl-dep-compact | 0.7084±0.0014 | 0.5119±0.0092 |
| nl-pos-compact | 0.7043±0.0078 | 0.5155±0.0123 |
| nl-split | 0.7226±0.0030 | 0.5111±0.0083 |

Key findings from multi-seed:
- NL format advantage is statistically significant: +8.6 points OOD over structured (p < 0.01)
- dep-compact does NOT significantly help OOD in either setup (0.5247 vs 0.5091, overlapping CIs)
- nl-split gives best ID but OOD is within noise of nl-baseline
- All NL variants cluster around 0.51-0.52 OOD — the format is the lever, not the variant
- Allrest training does NOT help OOD vs Rest14-only (0.516 vs 0.525) — more same-domain data doesn't generalise better

### Ollama Cross-Method (2026-05-19, ASTE, per-dataset demos for 6-shot)
| Method | Rest14 | Rest15 | Rest16 | Laptop14 |
|---|---|---|---|---|
| Gemma2 27B 0-shot | 0.4412 | 0.4176 | 0.4554 | 0.2754 |
| Gemma2 27B 6-shot | **0.6461** | **0.5917** | **0.6365** | **0.4095** |
| Qwen2.5 32B 0-shot | 0.5350 | 0.4513 | 0.5564 | 0.3631 |
| Qwen2.5 32B 6-shot | 0.6029 | 0.5273 | 0.6140 | 0.3700 |
| Command-R 0-shot | 0.3178 | 0.2558 | 0.3087 | 0.2064 |
| Command-R 6-shot | 0.5761 | 0.4986 | 0.5682 | 0.3732 |

- Gemma2:27b 6-shot is the best LLM across all datasets
- All LLMs still below fine-tuned T5 NL baseline (0.73 ID, 0.54 OOD)
- 6-shot demos from corresponding train split (per-dataset, no cross-contamination)

### RoBERTa / XLM-R (2026-05-18, Rest14 train only)
| Method | Rest14 | Rest15 | Rest16 | Laptop14 |
|---|---|---|---|---|
| RoBERTa span | 0.5439 | 0.4605 | 0.5600 | 0.3787 |
| XLM-R span | 0.4685 | 0.4277 | 0.5155 | 0.3363 |

- XLM-R underperforms RoBERTa on English — multilingual pretraining doesn't help for English-only ASTE
- Both well below T5 NL baseline (0.72 ID, 0.52 OOD) — naive span extraction can't compete with generative approach
- XLM-R Romanian (classifier, polarity+category on eMAG): 0.7945 pair-F1, 0.8805 polarity, 0.8614 category

### Romanian Experiments (2026-05-09/10/21)

#### T5 (FLAN-T5-base, NL format, eMAG phone reviews, polarity+category)
| Config | Polarity+Category F1 | Polarity F1 | Category F1 |
|---|---|---|---|
| ro-baseline | 0.7642 | 0.8816 | 0.8532 |
| ro-baseline-neutral-oversample | 0.7549 | 0.8780 | 0.8505 |
| ro-split | 0.7534 | 0.8744 | 0.8473 |

#### Ollama LLMs (Romanian, eMAG phone reviews)
| Method | Polarity+Category F1 | Polarity F1 | Category F1 |
|---|---|---|---|
| Gemma2 27B 0-shot | 0.7445 | 0.8877 | 0.8039 |
| Gemma2 27B 6-shot | 0.7582 | 0.8875 | 0.8259 |
| Qwen2.5 32B 0-shot | 0.6205 | 0.8537 | 0.6885 |
| Qwen2.5 32B 6-shot | 0.7401 | 0.8808 | 0.8072 |
| Command-R 0-shot | 0.5346 | 0.7659 | 0.6116 |
| Command-R 6-shot | 0.7501 | 0.8800 | 0.8214 |

Key findings:
- Fine-tuned T5 (250M) matches or beats 27-32B LLMs on Romanian ABSA
- Gemma2 27B 0-shot nearly matches T5 on pair-F1 (0.74 vs 0.76) — strong zero-shot for Romanian
- Category prediction is the bottleneck for LLMs (0.61-0.80 vs T5's 0.85)

### DMASTE Cross-Domain (2026-05-27, retested)

Training on DMASTE source domains, testing on 4 held-out target domains (book, grocery, pet, toy). Triplet micro-F1.

#### Single-source (Electronics only → target)

| Method | Book | Grocery | Pet | Toy | Avg |
|---|---|---|---|---|---|
| nl-template-aug | 37.28 | 43.78 | 40.33 | 46.72 | 42.03 |
| nl (baseline) | 36.05 | 43.65 | 38.93 | 46.74 | 41.34 |
| nl-dep-compact | 34.74 | 42.45 | 38.47 | 44.77 | 40.11 |
| nl-pos-compact | 33.35 | 42.15 | 38.15 | 44.03 | 39.42 |
| structured | 34.49 | 38.88 | 37.62 | 44.67 | 38.92 |
| roberta (span) | 10.59 | 16.08 | 15.35 | 13.95 | 13.99 |

#### Multi-source (Electronics+Beauty+Fashion+Home → target)

| Method | Book | Grocery | Pet | Toy | Avg |
|---|---|---|---|---|---|
| nl-template-aug | 41.83 | 46.90 | 43.06 | 50.02 | 45.45 |
| nl (baseline) | 41.27 | 46.62 | 43.22 | 50.16 | 45.32 |
| nl-pos-compact | 41.30 | 45.86 | 43.18 | 48.94 | 44.82 |
| nl-dep-compact | 40.45 | 46.47 | 43.77 | 48.56 | 44.81 |
| structured | 37.50 | 44.96 | 42.47 | 47.33 | 43.07 |
| roberta (span) | 20.39 | 23.16 | 22.37 | 22.15 | 22.02 |

#### Comparison with DMASTE paper (Xu et al., 2023)

Single-source (Electronics → target):
| Method | Book | Grocery | Pet | Toy | Avg |
|---|---|---|---|---|---|
| Span-ASTE (paper best) | 40.36 | 45.36 | 41.04 | 47.23 | 43.50 |
| GAS (paper, generative) | 35.57 | 39.16 | 38.17 | 43.55 | 39.11 |
| **Ours: nl-template-aug** | 37.28 | 43.78 | 40.33 | 46.72 | 42.03 |

Multi-source (ALL → target):
| Method | Book | Grocery | Pet | Toy | Avg |
|---|---|---|---|---|---|
| Span-ASTE (paper best) | 41.83 | 46.07 | 43.62 | 50.16 | 45.42 |
| **Ours: nl-template-aug** | 41.83 | 46.90 | 43.06 | 50.02 | 45.45 |
| **Ours: nl (baseline)** | 41.27 | 46.62 | 43.22 | 50.16 | 45.32 |

Key findings:
- Multi-source: FLAN-T5-base matches Span-ASTE (paper SOTA) without any domain adaptation (45.45 vs 45.42 avg)
- Single-source: significantly outperforms GAS (paper generative, +3 avg), approaches Span-ASTE (-1.5 avg)
- NL format consistently beats structured (+2-3 points avg)
- Template augmentation (shuffle_tasks) gives small edge in single-source, negligible in multi-source
- Syntax enrichment (dep/pos-compact) does not help on DMASTE — slightly hurts
- RoBERTa span model generalises very poorly OOD (expected for non-generative)
- Multi-source training provides ~4-6 point boost over single-source
- Paper reports 5-seed averages; ours is single-seed (42)

### Next Steps
- Opinion synonym augmentation with fraction=0.2 running
- Error analysis (GPU busy)
- Finalise dissertation writing

### Opinion Synonym Augmentation (2026-06-02)

LLM-generated opinion synonym variants using Gemma2:27b via ollama chat API (5-shot).
For each training example, the LLM rewrites the sentence replacing opinion terms with synonyms of the same polarity.
Generated 1989 valid variants from 1266 Rest14 train examples (~80% success rate).
Saved to: `downloads/opinion_synonyms_rest14.json`

| Config | Rest14 (ID) | Rest15 | Rest16 | Laptop14 (OOD) |
|---|---|---|---|---|
| nl-baseline (5-seed avg) | 0.7218±0.005 | 0.6023±0.004 | 0.6798±0.010 | 0.5247±0.012 |
| nl-opinion-aug (fraction=1.0) | 0.7034 | 0.5711 | 0.6597 | 0.4722 |
| nl-opinion-aug (fraction=0.2) | RUNNING | | | |

fraction=1.0 result: augmentation actively hurts (-5 points OOD, -2 points ID). The LLM-generated sentences introduce distributional shift that dilutes the clean training signal. Testing with smaller fraction (0.2) to see if a lighter augmentation dose helps or if the noise is inherent.

### Annotation Quality Issues (for dissertation error analysis)

Example of questionable SemEval Rest14 annotation:
```
Sentence: "Barbecued codfish was gorgeously moist - as if poached - yet the fabulous texture was let down by curiously bland seasoning - a spice rub might have overwhelmed , however herb mix or other sauce would have done much to enhance ."
  [positive] Barbecued codfish -> moist         ← correct
  [negative] seasoning -> bland                 ← correct
  [negative] spice rub -> overwhelmed           ← wrong: hypothetical dismissal, not actual negative opinion
  [negative] herb mix -> to enhance             ← wrong: counterfactual wish, not negative about herb mix
  [negative] sauce -> to enhance                ← wrong: same issue
```
The annotators interpreted complex rhetorical structures (hypotheticals, counterfactuals) as negative opinions. The model cannot reasonably predict "to enhance" as a negative opinion span. This represents irreducible annotation noise that inflates the gap between model performance and true task difficulty.

### MvP & STAR Proper Replication (2026-06-02)

Proper replication of MvP (Gou et al., 2023) and STAR (Lai et al., 2025) mechanisms on our Rest14-only OOD setup.

#### Results (triplet micro-F1)

| Config | Rest14 (ID) | Rest15 | Rest16 | Laptop14 (OOD) |
|---|---|---|---|---|
| nl-baseline (5-seed avg) | 0.7218±0.005 | 0.6023±0.004 | 0.6798±0.010 | 0.5247±0.012 |
| MvP proper (greedy) | 0.7280 | 0.6047 | 0.6876 | 0.5283 |
| MvP proper + voting (t=3) | 0.7345 | 0.6040 | 0.6927 | 0.5296 |
| STAR proper (greedy) | 0.7177 | 0.5922 | 0.6923 | 0.4869 |
| STAR proper + voting (t=3) | 0.7230 | 0.5984 | 0.7136 | 0.4995 |

#### Implementation details (vs papers)

MvP replication includes:
- Element markers in input/output ([A], [O], [S])
- [SSEP] separator between tuples
- 5 orderings per example at training time (top-k=5 of 6 possible)
- Majority voting at inference with threshold k/2=3

MvP differences from paper:
- Single-dataset training (Rest14 only) vs paper's multi-task training on 10 datasets
- FLAN-T5-base vs paper's T5-base
- No tuple sorting by appearance order
- Paper's ID F1=76.08 uses multi-task cross-dataset knowledge; our 72.80 is single-dataset (fair for OOD evaluation)

STAR replication includes:
- MvP marker format as base (5 orderings)
- Pairwise relation sub-tasks ([AO], [AS], [SP] markers)
- Balanced contribution loss (per-level normalisation: main vs pairwise)
- Examples tagged with _level for loss grouping

STAR differences from paper:
- ASTE (triples) vs paper's ASQP (quads) — fewer element combinations
- No "overall relation" paraphrase task (would be redundant with our NL format)
- No generation-score-based order selection (minimal impact with only 6 permutations)
- Paper reports gains mainly in low-resource; full-data improvement over MvP is ~1 point

#### Key findings:
- MvP with voting matches nl-baseline on OOD (0.5296 vs 0.5247±0.012 — within CI)
- MvP gives small ID gain from voting (+0.7 points)
- STAR hurts OOD significantly (0.4869-0.4995 vs 0.5247 avg) — pairwise decomposition splits model capacity without helping cross-domain transfer
- STAR voting partially recovers OOD (+1.3 points over greedy) but still below baseline
- Neither method addresses vocabulary/domain shift — they optimise compositional accuracy which is already strong in-domain
- The marker format ([A] x [O] y [S] z) achieves comparable OOD to NL templates — format type is less important than previously thought when controlling for multi-ordering exposure

### Data Analysis & Figures (2026-06-02)

Generated dissertation figures in `dissertation/figures/`:
- `pos_comparison.pdf` — per-dataset POS distribution for aspects (top) and opinions (bottom), all 13 datasets
- `vocab_overlap.pdf` — aspect vs opinion token overlap for each test set with training
- `overlap_vs_f1.pdf` — 2x2 scatter: F1 vs vocabulary overlap, implicit %, sentence length, triplets/sentence
- `dataset_characteristics.pdf` — 5-panel: sentence length, triplets/sentence, implicit %, aspect span length, opinion span length

Key findings from data analysis:
- SemEval opinions are ADJ-dominant (37-46% head POS), DMASTE opinions are VERB-dominant (21-24%)
- This is a fundamental annotation style difference, not just vocabulary
- Vocabulary overlap: Laptop14 has only 8.7% aspect token overlap with Rest14 train, but 47.3% opinion overlap
- DMASTE target domains have 24-55% aspect overlap and 60-76% opinion overlap with source
- Pet dataset has highest vocabulary overlap (54.5% aspect, 75.5% opinion) but second-worst F1 (0.4322)
- Pet's poor performance explained by highest implicit aspect rate (47.4%) — vocabulary overlap is necessary but not sufficient
- Full report saved in `dissertation/data_analysis_report.txt`

### Template Augmentation on SemEval (2026-06-01, retested from checkpoint)

| Config | Rest14 (ID) | Rest15 | Rest16 | Laptop14 (OOD) |
|---|---|---|---|---|
| nl-baseline (s42 ref) | 0.7265 | 0.6073 | 0.6723 | 0.5430 |
| nl-template-aug | 0.7136 | 0.5968 | 0.6830 | 0.5147 |

Template augmentation (shuffle_tasks) does not help OOD on SemEval. Result falls within baseline confidence interval (0.5247±0.0116). Consistent with DMASTE finding.

### Cross-Domain Data Mixing (2026-06-01, Rest14 + DMASTE domain fraction)

| Config | Rest14 (ID) | Rest15 | Rest16 | Laptop14 (OOD) |
|---|---|---|---|---|
| nl-baseline (s42 ref) | 0.7265 | 0.6073 | 0.6723 | 0.5430 |
| domainmix-beauty | 0.7068 | 0.6095 | 0.6990 | 0.5231 |
| domainmix-electronics | 0.7197 | 0.6047 | 0.6849 | 0.5255 |
| domainmix-fashion | 0.7093 | 0.5894 | 0.6887 | 0.5015 |
| domainmix-home | 0.7359 | 0.5960 | 0.6886 | 0.5249 |
| domainmix-all | 0.7191 | 0.6148 | 0.6679 | 0.5221 |

Key findings:
- No domainmix variant significantly improves Laptop14 OOD — all results within baseline CI (0.5247±0.0116)
- domainmix-all helps near-OOD (Rest15: 0.6148, best) but not far-OOD
- domainmix-home gives best ID (0.7359) — home domain vocabulary overlaps with restaurant
- Adding diverse data helps within-domain-family transfer but not cross-domain
- Another negative result: data mixing is not a lever for far-OOD generalisation
| Config | ID Quad F1 | ID Category F1 | ID Aspect F1 | OOD Aspect F1 | OOD a+s+p F1 |
|---|---|---|---|---|---|
| nl-baseline_2 (ls=0.05) | **0.5515** | 0.8061 | **0.8114** | **0.6954** | **0.4735** |
| nl-focal (γ=2.0) | 0.5468 | **0.8126** | 0.7975 | 0.6821 | 0.4735 |

Focal loss is a wash. It removes label smoothing (mutually exclusive), and the hard-example reweighting doesn't help because the "hard" tokens in generative ABSA are mostly template tokens, not the semantically important ones. The token-level focal weighting doesn't address the real difficulty: compositional prediction of which aspect goes with which category and polarity.

### ACOS Quad Step 3: Syntax Enrichment (2026-05-03)
| Config | ID Quad F1 | ID a+s+p F1 | OOD Aspect F1 | OOD a+s+p F1 |
|---|---|---|---|---|
| nl-baseline_2 | **0.5515** | 0.6227 | **0.6954** | **0.4735** |
| nl-dep-compact | 0.5396 | **0.6227** | 0.6707 | 0.4617 |
| nl-pos-compact | 0.5520 | — | — | — |

dep-compact hurts on ACOS quad (opposite of ASTE where it was the best OOD lever). Truncation is not the issue — only 2/1530 inputs exceed 384 tokens. The syntax info doubles average input length (52→109 tokens) without adding useful signal for category prediction, which is the conjunction bottleneck for quad F1.

### ACOS Quad Step 4: Constrained Decoding & Structured Attention (2026-05-03)
| Config | ID Quad F1 |
|---|---|
| nl-baseline_2 | **0.5515** |
| nl-baseline_2 + constrained decoding | 0.5476 |
| nl-dep-compact + structured attention mask | 0.5463 |

Constrained decoding (FSM-based, polarity + category trie): no improvement. The model already generates valid template outputs — errors are semantic (wrong category) not structural (malformed output). The NL format provides strong implicit structural constraints during training.

Structured attention mask (syntax tokens only attend to corresponding sentence words + dep-linked entries): no improvement. T5 was pretrained with full attention for 12 layers. Restricting attention during fine-tuning fights against pretrained patterns rather than enhancing them. Consistent with Syntax-BERT finding that structured masks help only on a subset of heads, not all.

### Infrastructure (2026-05-03)
- FSM-based constrained decoding for NL templates (`src/model/constrained.py`): trie-based constraining of polarity and category slots, text-based FSM position tracking
- Structured attention mask (`src/data/syntax_mask.py`): 2D attention mask from dep-compact syntax, T5 encoder monkey-patch for 3D mask support
- Auxiliary syntax prediction tasks: dep-prediction and pos-tagging as auxiliary training objectives, configurable fraction and task mix

### ACOS Quad Results (2026-05-03, Rest16 train → Rest16 test + Laptop14 OOD)
| Config | ID Quad F1 | ID a+p+c F1 | OOD Aspect F1 | OOD Polarity F1 |
|---|---|---|---|---|
| structured (lr=1e-4) | 0.5477 | 0.6333 | 0.5283 | 0.7168 |
| nl-baseline (lr=3e-4) | 0.5490 | 0.6291 | 0.6797 | 0.8754 |
| nl-baseline_2 (lr=1e-4) | **0.5515** | **0.6459** | **0.6954** | 0.8742 |

Key findings:
- NL format helps OOD massively (aspect: 52.8→69.5), consistent with ASTE findings
- ID quad F1 similar across formats (~55), NL slightly better
- OOD quad F1 near zero (~2.3-2.9) due to category mismatch across domains
- Implicit annotations are the main drag: explicit quad F1=58.1 vs implicit=43.2
- Lower lr (1e-4 vs 3e-4) helps with small dataset (1530 sentences)
- Category shuffling per example implemented to reduce positional bias

### NL Template Overhaul (completed)
- New quad template: "{aspect}, related to {category}, is described as {sentiment}, expressing a {polarity} sentiment"
- Unknown aspect: "an unspecified aspect" replaces {aspect}
- Unknown sentiment: "{aspect}, related to {category}, carries an implied {polarity} opinion"
- Annotated implicit: "the implied aspect {term}" prefix (no "is" verb)
- All combinations round-trip through encoder → parser correctly
- Updated for ASTE triples as well
- pos-compact syntax mode added (content words with POS tags, separate Syntax line)

### Implicit Evaluation Modes (completed)
- `implicit_mode` in config: null, "collapse", "resolve", "full"
- collapse: IMPLICIT:staff → IMPLICIT (standard benchmark)
- resolve: IMPLICIT:staff → staff (compare inferred terms)
- full: runs all three modes, reports separate metrics for each
- Wired through config → pipeline → model → evaluate

### Focal Loss (completed)
- `focal_gamma` config option: 0 = disabled, 2.0 = typical
- Mutually exclusive with label_smoothing (focal takes priority)
- Downweights easy/frequent predictions, focuses on hard/rare ones

### Data Infrastructure (completed)
- SemEval 2015/2016 XML loaders: `load_semeval_xml` (flat) and `load_semeval_xml_reviews` (review-grouped)
- Downloaded original XML data to `downloads/semeval_xml/`
- Downloaded DMASTE dataset to `downloads/DMASTE/` — 8 Amazon domains, ~40% implicit aspects
- Terminal annotator tools: `tools/annotate.py` (SemEval XML), `tools/annotate_aste.py` (ASTE format)
- LLM suggestion tool: `tools/suggest_aspects.py` — Gemma infers implicit aspects, annotator shows as defaults
- SemEval 2015 Rest: 375 NULL targets (91% resolvable from review context)
- SemEval 2016 Rest: 627 NULL targets (85% resolvable from review context)
- DMASTE: 4 domains with train data (beauty, electronics, fashion, home), 4 test-only domains

## Future Work (Prioritized)

### Remaining Experiments
- Domain-mix (Rest14 + DMASTE domain fraction): configs ready, bug fixed, in run_final_gaps.sh
- Template-aug retest on SemEval: checkpoint exists, needs retest with fixed config

### Dissertation Writing
- Main story: NL output format as the key lever for OOD generalisation in generative ABSA
- Supporting evidence: multi-seed validation, cross-dataset (DMASTE), cross-method comparison
- Negative results chapter: masking, curriculum, focal loss, constrained decoding, beam search, syntax enrichment (context-dependent)
- Romanian case study: fine-tuned T5 vs LLMs on low-resource language

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
| Method | 14Res | 14Lap | 15Res | 16Res |
|---|---|---|---|---|
| GAS (2021) | ~70 | ~58 | — | — |
| Paraphrase (2021) | ~70 | ~58 | — | — |
| MvP (T5-base, ACL 2023) | 76.08 | 65.84 | 69.91 | 72.36 |
| **Ours (nl-baseline, per-dataset)** | **72.4** | **62.7** | **64.0** | **72.5** |
| **Ours (nl-dep-nl, per-dataset)** | **73.1** | **60.7** | **65.0** | **72.3** |

Note: "per-dataset" = standard benchmark setup (train and test on same dataset).

### Generative (multi-restaurant train, non-standard)
| Method | 14Res | 14Lap (OOD) |
|---|---|---|
| Ours (nl-baseline) | 71.66 | 52.11 |
| Ours (nl-dep-compact) | 70.83 | **54.14** |

**NOTE (2026-05-19): Multi-restaurant pooling abandoned.** Discovered that 80% of Rest15 test sentences overlap with the combined Rest14+15+16 train set (shared Yelp reviews across SemEval years). This inflates discriminative model scores on Rest15 (XLM-R got 0.77 — clearly leaked). Generative T5 was less affected due to output format differences, but the setup is still methodologically unsound. All OOD experiments now train on Rest14 only and test on Rest14 (ID), Rest15 (near-OOD), Rest16 (near-OOD), Laptop14 (OOD). Overlap stats: Rest14 test has 1 sentence overlap (0.2%), Rest16 test has 1 sentence overlap (0.3%), Rest15 test has 258 sentences overlap (80.1%).

### OOD (Restaurant → Laptop)
| Method | 14Res→14Lap |
|---|---|
| Chia et al. generative avg drop | ~14.6 points |
| Chia et al. discriminative avg drop | ~16.8 points |
| **Ours (nl-dep-compact)** | **54.14** (from 70.83 ID, drop=16.7) |
| **Ours (nl-baseline)** | **52.11** (from 71.66 ID, drop=19.6) |

Note: OOD numbers are Restaurant14+15+16 → Laptop14 (train on 3 restaurant datasets, test on laptop). No direct OOD comparison exists for this exact setup.


## Dissertation

Working title: "A Systematic Empirical Study of Training Strategies for Out-of-Domain Generalisation in Generative Aspect-Based Sentiment Analysis"
