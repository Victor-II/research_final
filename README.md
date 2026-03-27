## PLAN

- **Model**: T5 | FLAN‑T5
- **Augmentation**:
    - Keep compositional stuff from before, after refinement.
    - [NEW] Add masking for aspect, and maybe pos-specific.
    - [NEW] Add artificial data: paraphrasing, implicit aspects. Maybe use Dependency Parsing.

- **Goals**:
    - Out-of-domain performance.
    - Implicit aspect performance.
    - Efficiency: squueze as much info from data as possible.

- **Evaluation**:
    - Standard eval for unmasked triplet & atomic task extraction
    - Embedding-based eval for masked tokens & standard for rest (if aspect masked and sentiment not, embedding-based for aspect & standard for sentiment). Embedding-based eval should be done at sentence level by frozen embedding model. It should be thresholded (if similarity > T, aspect is considered correct, same as for standard eval). Threshold should be calibrated.
    - [?] Maybe be more strict to masked samples and only consider correct if both aspect & polarity are correct.
    - [?] Use counterfactual stability score. Eval original + masked sample, compare similarity of aspects, etc. 
    - [?] To prevent generic inferred aspects, weight score inverse document frequency.

- **Data**:
    - Use py-absa stuff.
    - Use romanian emag ds as well.

- consider prefix_allowed_tokens_fn for polarity extraction.
- consider using json for prompt formatting

**[NOTE]** Include both input & output format in every prompt.

[MAYBE] Expand romanian dataset.

[MAYBE] Experiment with custom model.

### TODO (not necessarely in that order)

- [ ] Decide project structure & configs.

- [ ] Refine & integrate previous work. 

- [ ] Build connector for py-absa data format & previous T5 format & romanian ds format.
- [ ] Build vizualization suite & decide on a logger.

- [ ] Build data masking module.
- [ ] Build data paraphrasing module.

- [ ] Research relevant potential metrics & eval strategies. (write them here).
- [ ] Build evaluation module.

- [ ] Build testing pipeline.

- [ ] (I) PRELIMINARY TESTING INCLUDING: Previous work.
- [ ] (II) PRELIMINARY TESTING INCLUDING: Masking.
- [ ] (III) PRELIMINARY TESTING INCLUDING: Paraphrasing. 

- [ ] Train & eval Previous work.
- [ ] Train & eval Masking.
- [ ] Train & eval Paraphrasing.
- [ ] Train & eval ALL.

- [ ] **FIND ARTICLES TO COMPARE AGAINST**

- [ ] Aggregate results.




