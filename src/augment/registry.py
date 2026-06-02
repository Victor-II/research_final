from src.augment.masking import mask_aspects, mask_opinions, opinion_prediction_aux


def apply_augmentations(examples: list[dict], cfg: dict) -> list[dict]:
    aug_cfg = cfg.get("data", {}).get("augmentation", {})
    if not aug_cfg:
        return examples

    if aug_cfg.get("duplicate"):
        from src.augment.duplicate import duplicate_examples
        d = aug_cfg["duplicate"]
        examples = duplicate_examples(
            examples,
            fraction=d.get("fraction", 0.5),
            seed=cfg.get("seed", 42),
            filter_polarity=d.get("filter_polarity", None),
        )

    if aug_cfg.get("mask_aspects"):
        m = aug_cfg["mask_aspects"]
        examples = mask_aspects(
            examples,
            fraction=m.get("fraction", 0.5),
            replace=m.get("replace", False),
            mask_target=m.get("mask_target", False),
            seed=cfg.get("seed", 42),
        )

    if aug_cfg.get("mask_opinions"):
        m = aug_cfg["mask_opinions"]
        examples = mask_opinions(
            examples,
            fraction=m.get("fraction", 0.5),
            replace=m.get("replace", False),
            mask_target=m.get("mask_target", False),
            seed=cfg.get("seed", 42),
        )

    if aug_cfg.get("nlpaug"):
        from src.augment.nlpaug_aug import nlpaug_augment
        n = aug_cfg["nlpaug"]
        examples = nlpaug_augment(
            examples,
            method=n["method"],
            fraction=n.get("fraction", 0.5),
            replace=n.get("replace", False),
            protect_aspects=n.get("protect_aspects", True),
            seed=cfg.get("seed", 42),
            aug_p=n.get("aug_p", 0.1),
            model=n.get("model", "bert-base-uncased"),
        )

    if aug_cfg.get("llm_paraphrase"):
        from src.augment.llm_paraphrase import load_paraphrases, apply_paraphrase_augmentation
        p = aug_cfg["llm_paraphrase"]
        paraphrased = load_paraphrases(p["data"])
        examples = apply_paraphrase_augmentation(
            examples,
            paraphrased,
            fraction=p.get("fraction", 0.3),
            replace=p.get("replace", True),
            seed=cfg.get("seed", 42),
        )

    if aug_cfg.get("opinion_synonyms"):
        import json as _json
        o = aug_cfg["opinion_synonyms"]
        with open(o["data"]) as f:
            syn_data = _json.load(f)
        # convert to canonical format and append
        fraction = o.get("fraction", 1.0)
        if fraction < 1.0:
            import random as _rnd
            rng = _rnd.Random(cfg.get("seed", 42))
            n_keep = max(1, int(len(syn_data) * fraction))
            syn_data = rng.sample(syn_data, n_keep)
        for entry in syn_data:
            examples.append({
                "sentence": entry["sentence"],
                "tokens": entry["tokens"],
                "annotations": entry["annotations"],
            })

    return examples
