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


## NEW — Added from Literature Search (2026-06-xx)

### Generative ABSA — Recent Methods (2024-2026)

- **DOT: Dynamic Order Template Prediction for Generative ABSA**
  ACL 2025 (Short Papers) — Improves on MvP by dynamically selecting which element orderings to use per instance based on entropy, rather than exhaustively generating all orderings. Reduces inference cost while maintaining or improving F1. Directly relevant to our template augmentation / MvP comparison.
  https://aclanthology.org/2025.acl-short.48/

- **DS2-ABSA: Dual-Stream Data Synthesis with Label Refinement for Few-Shot ABSA**
  ACL 2025 (Long Papers) — LLM-based dual-stream data synthesis for few-shot ABSA. Generates data via both label-to-text and text-to-label streams, with a label refinement step to filter noise. Very relevant to our LLM augmentation experiments (opinion synonyms).
  https://acl.ldc.upenn.edu/2025.acl-long.752/

- **LLM-MvP: Multi-view Prompting Large Language Models for ABSA**
  arXiv 2026 — Extends the MvP multi-view prompting to LLMs directly (without fine-tuning). Closes the gap between few-shot prompting and fine-tuned models. Good context for our LLM comparison section.
  https://arxiv.org/abs/2605.28058

- **Self-Training with Pseudo-Label Scorer for ASQP**
  ACL 2024 (Long Papers) — Self-training framework with a scorer to assess quality of pseudo-labels for generative ASQP. Uses unlabeled data to improve T5/BART-based models. Achieves SOTA on several ASQP benchmarks.
  https://acl.ldc.upenn.edu/2024.acl-long.640/

- **ADA: Adaptive Data Augmentation for Aspect Sentiment Quad Prediction**
  2024 — Addresses polarity imbalance in ASQP through targeted augmentation of minority classes.
  https://arxiv.org/abs/2401.06394

- **Gen-mABSA-T5: A Multilingual Zero-Shot Generative Framework for ABSA**
  BLP Workshop 2025 — T5-based multilingual generative ABSA, zero-shot cross-lingual transfer. Relevant to our Romanian experiments.
  https://acl.ldc.upenn.edu/2025.banglalp-1.12/

### ASTE — Discriminative SOTA (2024-2025)

- **ASTE-Transformer: Modelling Dependencies in ASTE**
  EMNLP 2024 Findings — Novel transformer-inspired layers that model dependencies between phrases for ASTE. Achieves competitive or better results than table-filling methods. Pre-training further improves. Important for our cross-method comparison.
  https://aclanthology.org/2024.findings-emnlp.129/

- **DeBERTa Enhanced Syntactic-Semantic ASTE**
  arXiv 2025 — Uses DeBERTa with syntactic and semantic enrichment for ASTE. Relevant to our syntax experiments with RoBERTa.
  https://arxiv.org/abs/2511.10577

- **A Transitional Approach for Efficient ASTE**
  arXiv 2024 — Unified framework for AOPE and ASTE avoiding pipeline error propagation. Claims +6.98% F1 over SOTA.
  https://ar5iv.labs.arxiv.org/html/2412.00208

- **Near Ultimate Minimalist Contrastive Grid Tagging Scheme for ASTE**
  arXiv 2024 — Minimalist grid tagging approach with contrastive learning. Reports competitive F1 on standard benchmarks.
  https://ar5iv.labs.arxiv.org/html/2406.11234

### Cross-Domain ASTE (2024-2025)

- **TFMT: Table-Filling via Mean Teacher for Cross-domain ASTE**
  arXiv 2024 — Table-filling + mean teacher for pseudo-label generation on unlabeled target domain data. Semi-supervised cross-domain ASTE. Uses DMASTE benchmark. Directly comparable to our work.
  https://arxiv.org/abs/2407.21052

- **FOAL: Fine-grained Contrastive Learning for Cross-domain ASTE**
  arXiv 2023 — Contrastive learning with fine-grained category alignment to reduce domain discrepancy in ASTE. Discriminative approach requiring target domain data.
  https://arxiv.org/abs/2311.10373

- **Prototype-Regularized Federated Learning for Cross-Domain ASTE**
  arXiv 2026 — Federated learning approach allowing cross-domain knowledge sharing without data sharing. Novel angle on cross-domain ASTE.
  https://arxiv.org/html/2604.09123v1

- **Source-free Domain Adaptation for ABSA**
  LREC-COLING 2024 — Domain adaptation without access to source data at adaptation time. Different setting from ours (we do zero-shot OOD without any target data) but relevant.
  https://acl.ldc.upenn.edu/2024.lrec-main.1310/

- **Zero-Shot Cross-Domain ABSA via Domain-Contextualized Chain-of-Thought Reasoning**
  EMNLP 2025 Findings — LLM-based cross-domain ABSA using CoT prompting with domain context. No fine-tuning required. Directly relevant to our LLM experiments.
  https://acl.ldc.upenn.edu/2025.findings-emnlp.245/

### LLMs for ABSA (2024-2026)

- **Large Language Models for Aspect-Based Sentiment Analysis**
  arXiv 2024 (Scaria et al.) — Comprehensive evaluation of GPT-3.5 and GPT-4 on ABSA. Zero-shot, few-shot, and fine-tuned settings. Fine-tuned GPT-3.5 achieves 83.8 F1 on joint ATE+ATSC (SemEval 2014). GPT-4 matches SOTA in 6-shot. Important context for our LLM comparison. Note: evaluates ATE/ATSC subtasks, not full ASTE triplets.
  https://arxiv.org/abs/2310.18025

- **InstructABSA: Instruction Learning for ABSA**
  NAACL 2023 — Instruction tuning for ATE, ATSC, AOPE subtasks. Tk-Instruct-based (11B params). SOTA on several SemEval subtasks, outperforming 7x larger models. Important baseline for instruction-tuned ABSA.
  https://arxiv.org/abs/2302.08624

- **LLaMA-Based Models for ABSA**
  WASSA Workshop 2024 — Fine-tuning LLaMA for compound ABSA tasks. Shows that fine-tuned LLMs can approach specialized models. Relevant to the LLM fine-tuning vs small model debate.
  https://acl.ldc.upenn.edu/2024.wassa-1.6/

- **Heuristic-enhanced Candidates Selection for GPTs on Few-Shot ABSA**
  arXiv 2024 — Proposes candidate selection strategies to improve GPT few-shot performance on ABSA. Shows that vanilla GPT few-shot underperforms PLM-based methods on fine-grained tasks.
  https://arxiv.org/abs/2404.06063

- **Sentiment Analysis in the Age of Generative AI**
  2024 Survey — Broad survey on LLMs for sentiment analysis. Finds LLMs sometimes surpass traditional transfer learning for classification but struggle with fine-grained extraction.
  https://www.researchgate.net/publication/378743627

### LLM-Based Data Augmentation for NLP (2024-2025)

- **Data Augmentation using LLMs: Data Perspectives, Learning Paradigms and Challenges**
  ACL 2024 Findings — Comprehensive survey of LLM-based augmentation across NLP tasks. Categorizes approaches and identifies when LLM augmentation helps vs hurts. Key reference for our augmentation experiments section.
  https://aclanthology.org/2024.findings-acl.97

- **Empowering Large Language Models for Textual Data Augmentation**
  ACL 2024 Findings — Framework for using LLMs as data augmenters with quality control. Shows augmentation helps most in low-resource settings, less in full-data regimes. Consistent with our negative augmentation results.
  https://aclanthology.org/2024.findings-acl.756/

- **Exploring ChatGPT-Based Augmentation Strategies for Contrastive ABSA**
  IEEE Intelligent Systems 2025 — Three augmentation strategies using ChatGPT (context-focused, aspect-focused, combined) + contrastive learning. Shows augmentation helps for ATSC (classification) but does not evaluate on extraction or OOD settings.
  https://www.computer.org/csdl/magazine/ex/2025/01/10897267/24uGPgXHRkY

- **CKG: Improving ABSA with ChatGPT Augmentation and Knowledge-Enhanced GCN**
  PLoS ONE 2024 — ChatGPT text augmentation + knowledge graph-enhanced GCN for ABSA. Shows modest improvements in in-domain ATSC settings.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301508

- **Aspect-Based Sentiment Analysis with Dual Contrastive Learning and LLMs Data Augmentation**
  Springer 2025 — LLM-generated context-aware augmented samples + dual contrastive learning. In-domain improvements on ABSA classification.
  https://link.springer.com/chapter/10.1007/978-981-96-9994-0_13

### Pre-trained Models & Instruction Tuning

- **Scaling Instruction-Finetuned Language Models (FLAN-T5)**
  Chung et al., 2022/2024 — Instruction tuning across 1800+ tasks. FLAN-T5 achieves strong few-shot performance comparable to much larger models. Foundation for our choice of FLAN-T5-base over plain T5-base.
  https://arxiv.org/abs/2210.11416

### Surveys & Overviews

- **A Unified Review of ASTE Methods in ABSA**
  Springer Knowledge and Information Systems, 2025 — Comprehensive survey of ASTE methods, covering discriminative, generative, and hybrid approaches. Good for the State of the Art chapter structure.
  https://link.springer.com/article/10.1007/s10115-025-02519-x

### Cross-Lingual ABSA (for Romanian context)

- **Improving Cross-lingual ABSA with LLM Data Augmentation**
  arXiv 2025 — Trains ABSA model, gets predictions on unlabeled target language data, then uses LLM to generate better-aligned sentences from noisy predictions. Semi-supervised cross-lingual approach. Relevant to our Romanian experiments showing that fine-tuned FLAN-T5 matches large LLMs.
  https://arxiv.org/html/2508.09515v1

- **Cross-lingual Transfer Strategies for ABSA**
  arXiv 2026 — Systematic comparison of cross-lingual transfer strategies for ABSA. Covers translate-train, translate-test, and multilingual model approaches. Relevant context for our XLM-R Romanian results.
  https://arxiv.org/html/2604.26619v1

### Key Comparison Numbers (for fairness check)

Published ASTE F1 scores (in-domain, standard benchmarks):
- BTF-CCL (AAAI 2025): Rest14=75.88, Lap14=63.29, Rest15=67.68, Rest16=73.80
- Span-ASTE (ACL 2021): Rest14=72.89, Lap14=62.40
- GAS (ACL 2021): Rest14=~70, Lap14=~58
- MvP (ACL 2023): Rest14=76.08 (multi-task, 10 datasets), Lap14=~66
- STAR (AAAI 2025): Primarily ASQP, gains mainly in low-resource
- Paraphrase (EMNLP 2021): Rest14=~72, Lap14=~60

Published OOD numbers:
- Chia et al. (2024, Domain-Expanded ASTE): Generative methods drop ~14.6 points OOD, discriminative ~16.8. ChatGPT has smallest gap (7.4 points) but worst in-domain.
- DMASTE (ACL 2023 Findings): Span-ASTE multi-source avg=45.42, GAS multi-source avg=~39

NOTE: Most methods above are evaluated ONLY in-domain. Our contribution is evaluating them in OOD settings. MvP's 76.08 uses multi-task training on 10 datasets (unfair for OOD comparison). Our single-dataset MvP replication gets 72.80 (fair comparison).
