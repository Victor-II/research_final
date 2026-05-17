# Research Progress Report

## Goal

Improve out-of-domain (OOD) generalisation in generative ABSA without scaling up models. Focus on training strategies that extract more value from existing model + data.

---

## Experimental Setups

### Setup A: OOD Generalisation (ASTE)
- **Model:** FLAN-T5-base (250M params)
- **Train:** SemEval Restaurant 14 + 15 + 16 combined (~3.5k sentences)
- **ID test:** Restaurant14 test
- **OOD test:** Laptop14 test (no laptop data during training)
- **Task:** Aspect Sentiment Triplet Extraction — predict (aspect, opinion, polarity)
- **Metric:** Micro F1 on exact triplet match

### Setup B: Per-Dataset Benchmarks (ASTE)
- **Model:** FLAN-T5-base
- **Train/test:** Standard SemEval splits (train on Rest14 → test on Rest14, etc.)
- **Purpose:** Compare against published results

### Setup C: ACOS Quad (OOD)
- **Model:** FLAN-T5-base
- **Train:** ACOS-Rest16 (~1.5k sentences)
- **OOD test:** ACOS-Laptop14
- **Task:** Predict (aspect, sentiment, polarity, category) quads

### Setup D: Romanian (Category + Polarity)
- **Model:** mT5-small (300M params)
- **Train:** eMAG phone reviews (~27k sentences, Romanian)
- **Test:** eMAG phone reviews test (~3k sentences)
- **Task:** Predict (category, polarity) pairs from review text

---

## Results: Setup A — OOD Generalisation

### Natural Language Output Format

The single biggest improvement. Replacing structured output `[aspect, opinion, polarity]` with templates like "pizza is described as delicious, expressing a positive sentiment".

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| Structured baseline | 0.6798 | 0.4296 |
| NL 10% | 0.7041 | 0.4354 |
| NL 50% | 0.7135 | 0.4575 |
| **NL 100%** | **0.7166** | **0.5211** |

**+9.2 points OOD** from NL format. The model generates semantically meaningful text rather than memorising positional patterns, which transfers better across domains.

### Syntax Enrichment (dep-compact)

Adding dependency parse as a separate "Syntax:" line (content words with head→dep edges).

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| NL baseline | 0.7166 | 0.5211 |
| **NL + dep-compact** | 0.7083 | **0.5414** |
| NL + dep-inline | 0.6546 | 0.3926 |
| NL + pos-inline | 0.6391 | 0.3851 |

dep-compact adds **+2 points OOD**. Inline modes destroy the original text signal and hurt badly.

### Task Splitting + NL

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| NL baseline | 0.7166 | 0.5211 |
| NL + split | 0.7161 | 0.5249 |
| NL + split + dep-compact | 0.7091 | 0.5116 |

Split alone: marginal gain. Combining split + dep-compact + NL hurts — too many techniques dilute the signal.

### Curriculum Learning (negative result)

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| NL + dep-compact (reference) | 0.7083 | **0.5414** |
| cur-overlap | 0.7156 | 0.5094 |
| cur-fast-ramp + dep | 0.7058 | 0.5227 |
| cur-sandwich | 0.7281 | 0.5166 |

No curriculum config beats the simple NL + dep-compact baseline for OOD.

### Aspect Masking (negative result)

| Config | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| NL + dep-compact (reference) | 0.7083 | **0.5414** |
| NL + dep + mask 10% | 0.7176 | 0.5074 |
| NL + dep + mask 25% | 0.7187 | 0.4882 |
| NL + mask 25% (no dep) | 0.7126 | 0.5110 |

Masking removes real aspect context the model needs for OOD transfer.

### Decoding Strategies (negative result)

| Strategy | Rest14 (ID) | Laptop14 (OOD) |
|---|---|---|
| Greedy (baseline) | 0.7083 | **0.5414** |
| Beam=4 | 0.7125 | 0.5304 |
| Sample t=0.7 | 0.6986 | 0.4770 |
| Vote (5 samples, thresh=3) | 0.7101 | 0.5198 |
| Constrained decoding | 0.6994 | 0.4833 |

Greedy remains optimal. All alternatives hurt OOD.

---

## Results: Setup B — Per-Dataset Benchmarks

Standard setup: train and test on the same dataset (30 epochs each).

| Config | Rest14 | Rest15 | Rest16 |
|---|---|---|---|
| NL baseline | 0.7243 | 0.6401 | 0.7248 |
| NL + dep-compact (NL format) | **0.7305** | **0.6502** | 0.7230 |
| NL + dep-compact + aux (dep) | 0.7176 | 0.6328 | **0.7256** |
| NL + pos-compact (NL format) | 0.7128 | 0.6270 | 0.7133 |
| NL + pos-compact + aux | 0.7272 | 0.6401 | 0.7175 |

dep-compact in NL format helps Rest14 (+0.6) and Rest15 (+1.0), neutral on Rest16. Auxiliary syntax prediction tasks don't consistently help.

### Comparison with published methods

| Method | Rest14 | Rest15 | Rest16 | Laptop14 |
|---|---|---|---|---|
| GTS (2020, discriminative) | 70.92 | 62.53 | 68.71 | 59.46 |
| Span-ASTE (2021, discriminative) | 72.89 | 64.45 | 71.85 | 62.40 |
| BDTF (2022, discriminative) | 74.35 | 66.12 | 72.27 | 62.59 |
| BTF-CCL (2025, discriminative SOTA) | 75.88 | 67.68 | 73.80 | 63.29 |
| MvP (2023, generative SOTA) | 76.08 | 69.91 | 72.36 | 65.84 |
| Ours: nl-baseline | 72.43 | 64.01 | 72.48 | **62.67** |
| Ours: nl-dep-nl | **73.05** | **65.02** | 72.30 | 60.74 |
| Ours: nl-dep-nl-aux | 71.76 | 63.28 | **72.56** | 59.11 |

dep-compact helps on Rest14 (+0.6) and Rest15 (+1.0) but hurts on Laptop14 (-1.9). Auxiliary syntax tasks are inconsistent. The plain NL baseline is the most robust — never catastrophically fails and is best on Laptop14.

---

## Results: Setup C — ACOS Quad

| Config | ID Quad F1 | OOD Aspect F1 |
|---|---|---|
| Structured | 0.5477 | 0.5283 |
| **NL baseline** | **0.5515** | **0.6954** |
| NL + dep-compact | 0.5396 | 0.6707 |
| NL + focal loss | 0.5468 | 0.6821 |
| NL + constrained decoding | 0.5476 | — |
| NL + structured attention | 0.5463 | — |

NL format: **+16.7 points OOD** on aspect extraction. dep-compact hurts here (opposite of ASTE) — syntax doubles input length without helping category prediction. Focal loss, constrained decoding, structured attention: all neutral or negative.

---

## Results: Setup D — Romanian (mT5-small)

Category + polarity prediction on eMAG phone reviews. Class distribution: positive 72%, negative 23%, neutral 5%.

| Config | pol+cat F1 | pol macro F1 | cat F1 | neutral F1 |
|---|---|---|---|---|
| 3ep, lr=1e-4 | 0.674 | 0.419 | 0.774 | 0.002 |
| 8ep, lr=3e-4 | 0.760 | 0.529 | 0.848 | 0.103 |
| **12ep, lr=3e-4** | **0.764** | 0.538 | **0.853** | 0.108 |
| 8ep + task split | 0.753 | 0.510 | 0.847 | 0.068 |
| 10ep + neutral 3x oversample | 0.755 | **0.574** | 0.851 | **0.221** |

**Findings:**
- mT5-small needs higher lr (3e-4) due to large vocab (250k tokens)
- Task splitting doesn't help — task is too simple for decomposition to add value
- Neutral oversampling doubles neutral F1 but slightly hurts joint F1 (tradeoff)
- Romanian NL templates with order-dependent variants implemented for future template augmentation experiments

---

## Summary: What Works for OOD

| Technique | OOD Gain | Notes |
|---|---|---|
| NL output format | +9.2 (ASTE), +16.7 (ACOS) | Biggest single lever |
| dep-compact syntax | +2.0 (ASTE) | Helps span extraction tasks, hurts category tasks |
| Higher lr for mT5 | +8.6 (Romanian) | Critical for large-vocab models |

## What Doesn't Work for OOD

Curriculum learning, aspect masking, beam search, temperature sampling, majority voting, constrained decoding, structured attention, nlpaug augmentation, focal loss, task splitting (on simple tasks).

---

## Error Analysis

### OOD Model (NL + dep-compact, trained Rest14+15+16 → tested Laptop14)

**Overall:** P=0.538, R=0.512, F1=0.525 | 328 sentences, 543 gold triplets

**False Positives (239):**
| Error Type | Count | % |
|---|---|---|
| Spurious other (valid-looking but not annotated) | 83 | 34.7% |
| Polarity error (right aspect+opinion, wrong polarity) | 57 | 23.8% |
| Spurious product name (e.g. "MBP", "Air", "Mac Mini") | 42 | 17.6% |
| Sentiment boundary error | 27 | 11.3% |
| Aspect boundary error | 15 | 6.3% |

**False Negatives (265):**
| Error Type | Count | % |
|---|---|---|
| Unseen aspect (never in training data) | 152 | 57.4% |
| Near-miss aspect (close but not exact match) | 54 | 20.4% |
| Near-miss sentiment (close opinion span) | 50 | 18.9% |
| Missed seen (aspect was in training) | 5 | 1.9% |

**Key insights:**
- 57% of missed triplets involve aspects never seen during training (e.g. "OS", "setup", "runs", "portability") — this is the core OOD challenge
- Product names are a major source of false positives (17.6%) — the model treats brand names as aspects because restaurant training has no equivalent pattern
- Polarity confusion is mostly neutral↔negative (21 cases) and neutral↔positive (16 cases) — neutral is rare in restaurant training data
- The model rarely hallucinates completely — most errors are boundary/granularity issues

**Most missed unseen aspects:** works (8), use (5), OS (5), setup (3), features (3), performance (3), runs (3)

---

### Benchmark Model (NL + dep-compact, trained Rest14 → tested Rest14)

**Overall:** P=0.710, R=0.693, F1=0.701 | 492 sentences, 994 gold triplets

**False Positives (282):**
| Error Type | Count | % |
|---|---|---|
| Unannotated aspect (valid but not in gold) | 69 | 24.5% |
| Sentiment boundary error | 66 | 23.4% |
| Polarity error | 55 | 19.5% |
| Spurious other | 48 | 17.0% |
| Aspect boundary error | 37 | 13.1% |

**False Negatives (305):**
| Error Type | Count | % |
|---|---|---|
| Missed seen (aspect was in training) | 88 | 28.9% |
| Near-miss sentiment | 77 | 25.2% |
| Near-miss aspect | 61 | 20.0% |
| Unseen aspect | 58 | 19.0% |
| Near-miss aspect+sentiment | 18 | 5.9% |

**Key insights:**
- 24.5% of FPs are "unannotated aspects" — the model correctly identifies aspects that the gold annotations missed (e.g. "atmosphere", "food", "place", "prices"). This suggests the true F1 is higher than reported.
- Sentiment boundary errors are the biggest issue (23.4% FP) — the model captures slightly different opinion spans than the gold (e.g. "successes" vs "classic successes", "below quality" vs "below")
- Only 19% of FNs involve truly unseen aspects (vs 57% in OOD) — confirming that domain shift is the primary OOD challenge
- Polarity confusion is mostly positive↔neutral — the model defaults to positive for ambiguous cases
- 11 malformed outputs (1.1%) — rare but non-zero template adherence failures

---

## Next Steps

 - LLM paraphrasing augmentation — rewrite explicit aspects as implicit (scaffolding ready, gemma2:27b prompt refined) - seems to hard to pull off right, will likely abandon
 - Implicit aspect annotation on SemEval 2015/2016 (tools built, annotation in progress)
 - Cross-domain testing on DMASTE (8 Amazon product domains) - done, good aspect / polarity scores, bad sentiment extraction due to different annotation conventions between datasets.
 - Lighter neutral oversampling for Romanian (1x instead of 3x).
 - xlm-roberta

