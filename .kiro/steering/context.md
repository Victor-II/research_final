# Project Context & Catch-Up Summary

## What this is
Master's dissertation (IISC program, University Politehnica of Bucharest) on out-of-domain generalisation in generative Aspect-Based Sentiment Analysis (ABSA). The thesis is structured as a series of research reports (reports 3 and 4 are written, the final dissertation integrates them).

## Core finding
Natural-language output templates are the single biggest lever for OOD generalisation in generative ABSA (+8.6 F1 over structured output, statistically significant across 5 seeds). Most other things we tried (syntax enrichment, curriculum learning, masking, data mixing, task decomposition, decoding strategies, opinion augmentation) did not help OOD. On the DMASTE benchmark our FLAN-T5-base model matches published Span-ASTE SOTA without domain adaptation.

## What we tested (all documented in .kiro/docs/progress.md)
- Output format: structured vs NL templates (NL wins big)
- Syntax enrichment: dep-compact, pos-compact, dep-inline, pos-inline (doesn't help OOD)
- Multi-ordering training / MvP replication (matches baseline, doesn't help OOD)
- STAR sub-task decomposition replication (hurts OOD)
- Curriculum learning: overlap-first, fast-ramp, sandwich (doesn't help OOD)
- Aspect masking at various fractions (hurts OOD)
- Cross-domain data mixing with DMASTE domains (doesn't help far-OOD)
- Template augmentation / shuffle_tasks (negligible)
- Opinion synonym augmentation via LLM (hurts)
- Decoding strategies: beam, sampling, voting, constrained (greedy remains best)
- LLM few-shot: Gemma2 27B, Qwen2.5 32B, Command-R (all below fine-tuned T5)
- Romanian ABSA: FLAN-T5, XLM-R, RoBERT, bert-ro-cased, LLMs
- RoBERTa/XLM-R span extraction baselines

## Current documents
- `dissertation/research_report_3.tex` — main English experiments report (output format, syntax, MvP, STAR, curriculum, masking, data mixing, DMASTE, error analysis)
- `dissertation/research_report_4.tex` — LLM comparison + Romanian experiments
- `dissertation/dissertation.tex` — the final integrated dissertation (being written)
- `dissertation/bibliography3.bib` / `bibliography4.bib` — per-report bibliographies
- `dissertation/bibliographyDizertatie.bib` — dissertation bibliography
- `.kiro/docs/progress.md` — full experiment log with all results and notes

## Key numbers (5-seed, Rest14 train only → Laptop14 OOD)
- Structured baseline: 0.439±0.012
- NL baseline: 0.525±0.012
- NL + dep-compact: 0.509±0.005
- NL + split: 0.524±0.011
- Best LLM (Gemma2 27B 6-shot): 0.410

## Experiment infrastructure
- `main.py` CLI with --config overlays, --mode train|test|aggregate|plot
- `config/base.yaml` + `config/overlays/` for experiment configs
- `experiments/YYYY-MM-DD/<name>/` for outputs (config snapshot, checkpoints, results.json)
- Results in `results.json` → test array → aspect+sentiment+polarity.micro.f1

## Current phase
Writing and polishing the dissertation. Experiments are complete. Recent tasks:
- Fixed citation errors across reports 3 and 4 (wrong authors, wrong papers, wrong venues)
- Updated curriculum learning and masking tables with fresh Rest14-only reruns (2026-06-07)
- All reports compile cleanly with pdflatex + bibtex
