# Project & Developer Preferences

## Research goal
Master's dissertation focused on improving ABSA performance, specifically:
- Out-of-domain generalisation
- Implicit aspect extraction (aspects not explicitly mentioned in text)
- Efficiency-first approach — no scaling up models, extract more value from existing model + data
- Key research direction: better data augmentation methods (e.g. syntactic masking, paraphrase-based augmentation)

## Project
- NLP research project: Aspect-Based Sentiment Analysis (ABSA) with generative models (FLAN-T5)
- PyTorch + PyTorch Lightning + HuggingFace Transformers stack
- Data from pyabsa / ABSADatasets (ASTE, APC formats)
- venv located at `/home/victor-ii/research/research_final/venv`

## Upcoming work
- Expand evaluation module (`src/eval/eval.py`) with more relevant metrics (e.g. per-class F1, macro averages, token-level overlap for span extraction)
- Replace hardcoded values in `main.py` and model with a config system (likely YAML-based, see `experiments/demo/config/config.yaml` for existing structure)

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
