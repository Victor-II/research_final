# Relevant Papers

## Entity/Aspect Masking as Augmentation

- **MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER**
  Zhou et al., 2021 — Mask entity spans, use fine-tuned MLM to fill in novel entities. Increases entity diversity for low-resource NER.
  https://arxiv.org/abs/2108.13655

- **EnTDA: Entity-to-Text based Data Augmentation for various NER Tasks**
  Hu et al., 2022 — Add/delete/replace/swap entities in entity lists, generate coherent text around them.
  https://arxiv.org/abs/2210.10343

## ABSA Data Augmentation

- **Exploring ChatGPT-based Augmentation Strategies for Contrastive ABSA**
  Xie et al., 2024 — Context-focused, aspect-focused, and combined augmentation using ChatGPT. Context-focused is conceptually similar to masking.
  https://arxiv.org/abs/2409.11218

- **CDGDA: A Cross-Domain Generative Data Augmentation Framework for ABSA**
  Xue et al., 2023 — Cross-domain generative augmentation using aspect replacement and aspect-sentiment pair replacement prompts.
  https://www.mdpi.com/2079-9292/12/13/2949

- **Bidirectional Generative Framework for Cross-domain ABSA**
  Deng et al., 2023 — T5-style text→label and label→text training. Label→text generates sentences from noisy labels for augmentation.
  https://arxiv.org/abs/2305.09509

- **Iterative Data Generation with LLMs for ABSA**
  2024 — Iterative LLM generation to produce ABSA training data.
  https://arxiv.org/abs/2407.00341

- **Data augmentation for aspect-based sentiment analysis (survey)**
  2022 — Overview of DA methods for ABSA.
  https://link.springer.com/article/10.1007/s13042-022-01535-5

- **Label-Consistent Data Generation for ABSA Using LLM Agents**
  2025 — Agentic augmentation with iterative generation and verification.
  https://arxiv.org/abs/2602.16379

- **ABSA-ESA: Aspect-Based Sentiment Analysis with Explicit Sentiment Augmentations**
  2023 — Adds explicit sentiment clues to help with implicit sentiment cases.
  https://arxiv.org/abs/2312.10961

## Implicit Aspect / Sentiment

- **Learning Implicit Sentiment in ABSA with Supervised Contrastive Pre-Training**
  Li et al., 2021 — ~30% of reviews lack explicit opinion words. Contrastive pre-training aligns implicit sentiment representations.
  https://arxiv.org/abs/2111.02194

- **BERT-ASC: Auxiliary-Sentence Construction for Implicit Aspect Learning**
  Ahmed et al., 2022 — Constructs auxiliary sentences from corpus semantics to help BERT learn implicit aspect representations.
  https://arxiv.org/abs/2203.11702

- **Metrics, Synthetic Data, and Aspect Extraction for ABSA with LLMs**
  Neveditsin et al., 2025 — LLMs for implicit aspect extraction in novel domain (sports). Proposes evaluation metric for implicit aspects. Uses synthetic data.
  https://arxiv.org/abs/2503.20715

- **Graph-enhanced Implicit Aspect-Level Sentiment Analysis based on Multi-Prompt Fusion**
  2025 — Graph-based approach for implicit aspect-level sentiment.
  https://www.nature.com/articles/s41598-025-02609-4

- **Implicit-Feature Alignment with Corpus Filtering for ABSA**
  2024 — NLI-based filtering as implicit feature for aspect category detection.
  https://arxiv.org/abs/2407.00342

## Cross-Domain / OOD Generalisation (NER & ABSA)

- **Are Data Augmentation Methods in NER Applicable for Uncertainty Estimation?**
  2024 — DA improves calibration and uncertainty in cross-genre and cross-lingual NER.
  https://arxiv.org/abs/2407.02062

- **An Experimental Study on Data Augmentation Techniques for NER on Low-Resource Domains**
  2024 — Comparative study of augmentation techniques for domain-specific NER.
  https://arxiv.org/abs/2411.14551

- **How Fragile is Relation Extraction under Entity Replacements?**
  2023 — RE models memorize entity name patterns, ignoring context. Relevant to the entity memorization problem.
  https://arxiv.org/abs/2305.13551

- **A Paraphrase-Augmented Framework for Low-Resource NER**
  2024 — Paraphrases surrounding context while preserving entity info. Similar spirit to your LLM paraphrasing direction.
  https://arxiv.org/abs/2510.17720

## Generative ABSA — Task Decomposition & Augmentation

- **STAR: Stepwise Task Augmentation with Relation Learning for ASQP**
  Xie et al., 2025 — Decomposes ASQP into pairwise and overall relation subtasks with increasing granularity. Augments training data with these auxiliary tasks. Uses balanced contribution loss for multi-task training. T5-base/large. SOTA on Rest15/16, ACOS-Laptop/Rest. Particularly strong in low-resource. Very relevant — validates task-splitting approach, but no OOD evaluation.
  https://arxiv.org/abs/2501.16093

- **Label-Consistent Data Generation for ABSA Using LLM Agents**
  Karimi et al., 2026 — Agentic augmentation with iterative generation + verification for T5-Base. Tests on SemEval ATE, ATSC, ASPE. Agentic outperforms raw prompting for label preservation.
  https://arxiv.org/abs/2602.16379

- **Balanced Training Data Augmentation for ABSA**
  Liu et al., 2025 — LLM generates augmented data with balanced label distributions. Uses RL to optimize augmentation quality. Addresses class imbalance (positive-heavy datasets). SemEval English benchmarks.
  https://arxiv.org/abs/2507.09485

- **Paraphrase (ASQP baseline)**
  Zhang et al., 2021 — Transforms sentiment quads into natural language using fixed template. T5-base. Rest15: 45.54, Rest16: 57.82, ACOS-Laptop: 43.06, ACOS-Rest: 59.63.
  https://arxiv.org/abs/2110.00796

- **MvP: Multi-view Prompting for ASQP**
  Gou et al., 2023 — Augments input with element order templates, multi-task learning. T5-base. Rest15: 50.16, Rest16: 61.05, ACOS-Laptop: 43.60, ACOS-Rest: 60.62.
  https://arxiv.org/abs/2305.xxxxx

## Selective / Non-Random Masking

- **Selective Masking based on Genre and Topicality for Domain Adaptation**
  2024 — Ranks words by domain significance, masks accordingly during continual pre-training. Outperforms random masking for domain-specific tasks. Legal domain.
  https://arxiv.org/abs/2402.12036

- **Token Masking Improves Transformer-Based Text Classification**
  2025 — Random token masking as regularisation during training. Acts as implicit gradient averaging, improves generalisation.
  https://arxiv.org/abs/2505.11746

- **ACLM: Selective-Denoising based Generative Data Augmentation for Low-Resource NER**
  2023 — Attention-guided selective masking for NER. Masks everything except entities and keywords, uses BART to reconstruct. Inverse of aspect masking.
  https://arxiv.org/abs/2306.00928

## ASTE Benchmarks

- **BTF-CCL: Boundary-Driven Table-Filling with Cross-Granularity Contrastive Learning for ASTE**
  2025 — BERT-based table-filling. SOTA-ish ASTE F1: 14Res ~75.88, 14Lap ~63.29, 15Res ~67.68, 16Res ~73.80.
  https://arxiv.org/abs/2502.01942


## OOD Generalisation for ASTE

- **Domain-Expanded ASTE: Rethinking Generalization in Aspect Sentiment Triplet Extraction**
  Chia et al., 2024 (EMNLP Workshop) — The only paper systematically evaluating ASTE OOD generalisation. Adds Hotel and Cosmetics domains to Restaurant and Laptop. Tests GTS, Span-ASTE, RoBMRC (discriminative), GAS, Paraphrase (generative), ChatGPT. Key findings: generative methods generalise better OOD (~14.6 point drop vs ~16.8 for discriminative), ChatGPT has smallest domain shift gap (7.4 points) but worst in-domain. Proposes CASE decoding (confidence-aware sampling) which improves both ID and OOD. Directly relevant — same research question as ours.
  https://arxiv.org/abs/2305.14434
  https://aclanthology.org/2024.sicon-1.11
  Data: https://github.com/DAMO-NLP-SG/domain-expanded-aste (may need alternative source)

- **ASTE-Transformer: Modelling Dependencies in Aspect-Sentiment Triplet Extraction**
  2024 (EMNLP Findings) — Transformer-inspired layers for modelling dependencies between phrases and classifier decisions. Pre-training technique further improves performance. Higher F1 than other methods on standard benchmarks.
  https://arxiv.org/abs/2409.15202


## ASTE Methods & Benchmarks (In-Domain)

- **GTS: Grid Tagging Scheme for Aspect-Oriented Fine-Grained Opinion Extraction**
  Wu et al., 2020 — Sequence tagging with grid structure. ASTE F1: 14Res ~70.92, 14Lap ~59.46.
  https://arxiv.org/abs/2010.04640

- **Span-ASTE: Learning Span-Level Interactions for ASTE**
  Xu et al., 2021 — Span enumeration approach. F1: 14Res ~72.89, 14Lap ~62.40.
  https://arxiv.org/abs/2107.12214

- **Span-ASTE with POS & Contrastive Learning**
  Li et al., 2024 — POS filter + contrastive learning on top of Span-ASTE. F1: 14Res ~74.79, 14Lap ~62.59.
  Neural Networks, 2024.

- **BDTF: Boundary-Driven Table-Filling for ASTE**
  Zhang et al., 2022 — Boundary-driven table-filling. F1: 14Res ~74.35, 14Lap ~62.59.
  https://arxiv.org/abs/2209.00820

- **BTF-CCL: Boundary-Driven Table-Filling with Cross-Granularity Contrastive Learning**
  2025 — Current SOTA (BERT-based). F1: 14Res ~75.88, 14Lap ~63.29, 15Res ~67.68, 16Res ~73.80.
  https://arxiv.org/abs/2502.01942

- **GAS: Towards Generative Aspect-Based Sentiment Analysis**
  Zhang et al., 2021 — First generative approach for ABSA using T5/BART. F1: 14Res ~70, 14Lap ~58.
  https://arxiv.org/abs/2103.01175

- **Paraphrase: Aspect Sentiment Quad Prediction as Paraphrase Generation**
  Zhang et al., 2021 — Natural language paraphrase output for ASQP. Also applicable to ASTE. Foundation for generative ABSA.
  https://arxiv.org/abs/2110.00796

- **RoBMRC: A Robustly Optimized BMRC for ASTE**
  Liu et al., 2022 — Machine reading comprehension approach for ASTE.
  https://aclanthology.org/2022.naacl-main.20

- **A Pairing Enhancement Approach for ASTE**
  2023 — Pairing enhancement on four ASTE datasets.
  https://arxiv.org/abs/2306.10042

- **Knowledge-Augmented GCN for ASTE**
  2025 — GCN with knowledge augmentation for aspect-opinion interactions.
  https://www.mdpi.com/2076-3417/16/3/1250
