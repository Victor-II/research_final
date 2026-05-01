# Project & Developer Preferences

## Research goal
Master's dissertation focused on improving ABSA performance, specifically:
- Primary goal: out-of-domain generalisation (this is the main focus)
- Implicit aspect extraction as a means to improve out-of-domain performance
- Efficiency-first approach — no scaling up models, extract more value from existing model + data
- Key research direction: better data augmentation methods (syntactic masking, implicit aspect generation via LLM paraphrasing)
- Comparison of custom augmentation strategies (masking, LLM paraphrasing) against standard baselines (nlpaug: EDA, synonym, contextual embeddings, back-translation)

## Project
- NLP research project: Aspect-Based Sentiment Analysis (ABSA) with generative models (FLAN-T5)
- PyTorch + PyTorch Lightning + HuggingFace Transformers stack
- Data from pyabsa / ABSADatasets (ASTE, APC formats)
- venv located at `/home/victor-ii/research/research_final/venv`

## Architecture overview
- `src/data/data.py` — canonical format (sentence, tokens, annotations with span indices), loaders, `to_generative_format`, `split_by_task`
- `src/model/model.py` — T5ABSAModel (PL Lightning module), `src/model/utils.py` for DDP gather helpers
- `src/eval/eval.py` — building block metric functions: `prf` (micro), `macro_prf`, `soft_prf` (embedding similarity for aspect), `evaluate`, `save_results`, `save_metrics_table`
- `src/eval/aggregate.py` — cross-experiment results aggregation (tables, LaTeX, plots)
- `src/augment/masking.py` — aspect span masking with `<extra_id_X>` sentinels
- `src/augment/registry.py` — augmentation dispatch: config → augmentation chain
- `src/pipelines/pipeline.py` — universal run logic (train, test, or both), decoupled from experiment paths
- `config/base.yaml` — single base config with all knobs, all augmentation options off by default
- `experiments/<name>/` — auto-generated experiment output (config snapshot, checkpoints, results)
- `aggregated/` — cross-experiment plots and tables
- `main.py` — CLI entry point with --config, --set, --mode train|test|aggregate|plot

## Canonical data format
All dataset loaders produce dicts with: `sentence`, `tokens`, `annotations` (list of dicts with `aspect`, `aspect_idx`, `sentiment`, `sentiment_idx`, `polarity`, `category` — all nullable). Augmenters operate on this format. `to_generative_format` converts to model input/target at the end of the pipeline.

## Upcoming work
- Implicit aspect paraphrasing augmentation via LLM API (rewrite explicit aspect sentences so aspect is implied, keep polarity label)
- Soft F1 evaluation using sentence-transformers (`all-MiniLM-L6-v2`) — already implemented
- Contextual soft F1 extension (embed aspect spans in full sentence context)
- Evaluation on out-of-domain datasets (cross-dataset testing already supported via config)
- Dissertation direction: empirical study of augmentation strategies for generative ABSA, with focus on implicit aspect generalisation

## Code style
- Minimal code — only what's needed, no verbose boilerplate
- No unnecessary comments or docstrings on obvious things
- Prefer clean, readable Python over clever one-liners
- Type hints are welcome but not mandatory everywhere

## Behaviour preferences
- Don't make changes without being asked — if something is unclear, ask first
- When asked for an explanation, give the explanation. Don't also make a code change unless asked
- Don't revert or undo things silently — always say what you're doing and why
- Strict output format evaluation: malformed predictions score zero, no lenient parsing
- Prefer `frombuffer(bytearray(...))` over `torch.tensor(list(...))` for byte-to-tensor conversion (zero-copy)

## Communication
- Be direct and concise
- Don't repeat what was just said or summarise unnecessarily
- Don't bold text
- Don't use markdown headers in short responses
- If something looks wrong, flag it and ask before changing it

## Development setup
- Working directly on the Ubuntu PC (no remote access / SSH)
- venv at `/home/victor-ii/research/research_final/venv`
