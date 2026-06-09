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


run latex:
`bibtex <file.tex> && pdflatex -interaction=nonstopmode <file.tex>`


FEEDBACK pentru report_3

[x] acesta merge in related work, este prea mult descris pentru introduction

[x] pune Out of domain si in paranteza OOD din moment ce nu se prezentat in introducere pana acum cu numele intreg, apoi poti pune peste tot direct OOD

[x] Asemenea, prima oara cand introduci un concept, abstractul este o exceptie, acolo pui doar denumirea intreaga, trebuie sa pui de ex Aspect Term Extraction (ATE), Aspect Sentiment Triplet Extraction ( ASTE), acesta deja a fost definit in introducere, putem sa punem abrevierile.

[x] Nu inteleg de ce este important sa previzam conferintele la care au fost acceptate, deja este scris in referinte, este posibil sa se dea flag la acest lucru. Daca template-ul sface acest lucru, atunci este ok.

[x] Suna putin a bragging, nu ar trebui sa spunem cat am obtinut noi, ci ceva de genul "noi am facut diferit in modul acesta" si e suficient. De exemplu la paragraful urmator este ok ca spui ca noua ne merge prost asta, pentru cazul nostru specific, dar incercam sa nu discutam numerele aici.

[x] 6 - pune mai bine \label{} la subsectiunile unde se foloseste un singur seed, astfel poti sa spui direct "The following ablation studies are using only the 42 seed .... " si le dai referinta cu \ref{}. Este foarte vag, nu se intelege care este seed-ul folosit, care sunt ablatiile care folosesc acel seed.

[x] 7 - trebuie motivat mai mult, de ce crezi ca nu merge, care este trade-ul dintre overfit si intreruperea learning rate decay, de ce facem asta, avem un tabel in care sa aratam cum merge pe 0.05, 0.1 and so on? Daca da, il referentiem.

[x] 8 - In caz ca aici a dat flag de AI, motivul ar fi ca lor le place sa zica "chestia asta este importanta / usoara / etc", In cazul in care da, poti reformula doar prin " This implementation do not require specialised architectures used in published discriminative methods", nu mai enumeram care nu ar fi. Si published discriminative models suna putin ciudat, sigur sunt si paper-uri publicate care nu folosesc... trebuie regandita pe acolo.
the 
[x] 9 - Este suficient sa specifici doar " The full pipeline", fara sa enumeri care sunt componentele. Un approach mai bun ar fi aici sa incepi cu Figure 1 shows si sa descrii vizual cum se ce se intampla acolo, ideal ar fi sa faci asa la fiecare imagine, fiindca oamenilor le este mai usor sa inteleaga din scris decat sa incerce sa faca legaturile in imagine.

[x] 10 - Vezi daca poti face tabele si sa voresti pe tabele, ar intelege cititorul mai bine de unde vin valorile si cum.

[x] 11 - Ar merge si cate o mica introducere a metodei, 2 randuri in care sa explici ce este curriculum learning for example. Dar merge oriunde e un concept.

[x] 12 - Asumarile acestea sunt foarte directe, ar trebui ceva de genul " We conclude from Table 7 that the masking actually hurts the ood, we tried making it 20% and with dep., the (metrica folosita dar nu a fost explicata, ar trebui specificat in captionul tabelului ce metrica folosim) droped on every dataset used", iar daca ai avea si o explicatie de ce crezi ca se intampla acest lucru, ar fi fain de pus "we assume that is from.... " (acest lucru se aplica peste tot unde nu este explicat si doar pus tabelul si o concluzie directa)

[x] 13 - citeaza paper-ul

