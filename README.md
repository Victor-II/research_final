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

- consider prefix_allowed_tokens_fn for polarity extraction.check to see if this approach has been tried
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



mai trebuie

 - raport sem 3 - tot pana la romana (practic tot cu ood and shit)
 - raport sem 4 - cu tot cu romana + lllms + llm aug


foloseste so robert pentru emag si compara cu multilingual (daca ala doar romana bate multilingual sau nu) robert 2017

annotations issue - pusa ca discussions

pune data analysis inainte de metode / implementare / rezultate

fa un flowchart cu best method

baga si niste loss chart

arata si ca ai masurat max sentence length si ca te-a dus capu sa pui seq_len calumea

continual learning ca motivare pentrui ood

de ce crezi ca multilingual e mai bun ca monolingual? sau diff intre ele, arg pro si contra;

de ce ai ales sa faci ood (continual learning)

cand merita si cand nu sa faci ood

pune citari pentru ollama models

pune mai multe citari in general

setul de emag introdus ca resursa laboratorului (lab6 se pregateste un jurnal)




@inproceedings{masala2020robert,
  title={Robert--a romanian bert model},
  author={Masala, Mihai and Ruseti, Stefan and Dascalu, Mihai},
  booktitle={Proceedings of the 28th international conference on computational linguistics},
  pages={6626--6637},
  year={2020}
}
 
 
@inproceedings{dumitrescu2020birth,
  title={The birth of Romanian BERT},
  author={Dumitrescu, Stefan and Avram, Andrei-Marius and Pyysalo, Sampo},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages={4324--4328},
  year={2020}
}
 
@inproceedings{aldeen2023chatgpt,
  title={Chatgpt vs. human annotators: A comprehensive analysis of chatgpt for text annotation},
  author={Aldeen, Mohammed and Luo, Joshua and Lian, Ashley and Zheng, Venus and Hong, Allen and Yetukuri, Preethika and Cheng, Long},
  booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)},
  pages={602--609},
  year={2023},
  organization={IEEE}
}
 
fix e

@inproceedings{ding2024boosting,
  title={Boosting large language models with continual learning for aspect-based sentiment analysis},
  author={Ding, Xuanwen and Zhou, Jie and Dou, Liang and Chen, Qin and Wu, Yuanbin and Chen, Arlene and He, Liang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={4367--4377},
  year={2024}
}


