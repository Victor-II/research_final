"""Verified LLM paraphrase generation with ASTE model verification loop.

Pipeline:
1. Load ASTE training data (all sentences with explicit aspects)
2. For each sentence, pick one triplet's aspect to make implicit
3. Ask LLM to paraphrase removing that aspect, keeping the rest intact
4. Verify with trained ASTE checkpoint:
   - Targeted aspect not explicitly present (NP embedding check)
   - Original sentiment preserved (lemmatized match)
   - Polarity matches
   - Non-targeted aspects still present in paraphrased sentence
   - Verifier extracts a triplet with aspect semantically close to original
5. On failure, retry with targeted feedback (up to max_retries)
6. Save accepted paraphrases in the llm_paraphrase JSON format
"""

import json
import re
import numpy as np
import requests
import spacy
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.data.data import load_aste_file, enrich_syntax
from src.eval.eval import parse_output

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _lemmatize(text: str) -> str:
    doc = _get_nlp()(text.lower())
    return " ".join(t.lemma_ for t in doc)


def _stem_match(word: str, text: str) -> bool:
    """Check if a word's stem appears in text. Handles horrible/horribly, good/good etc."""
    # try exact match first
    if word.lower() in text.lower():
        return True
    # try lemma match
    word_lemma = _lemmatize(word)
    text_lemma = _lemmatize(text)
    if all(t in text_lemma.split() for t in word_lemma.split()):
        return True
    # try prefix match (handles horrible/horribly, amaze/amazing/amazed)
    # use a reasonable prefix (min 4 chars, strip up to 3 suffix chars)
    for w in word.lower().split():
        prefix = w[:max(4, len(w) - 3)]
        if any(t.startswith(prefix) for t in text.lower().split()):
            continue
        return False
    return True


def _get_noun_chunks(text: str) -> list[str]:
    doc = _get_nlp()(text)
    chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    # also add individual nouns not in chunks
    chunk_tokens = set()
    for chunk in doc.noun_chunks:
        for t in chunk:
            chunk_tokens.add(t.i)
    for t in doc:
        if t.pos_ in ("NOUN", "PROPN") and t.i not in chunk_tokens:
            chunks.append(t.text.lower())
    return chunks


def load_verifier(checkpoint_path: str, device: str = "cuda"):
    from src.model.t5_model import T5ABSAModel
    model = T5ABSAModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _check_aspect_explicit(
    original_aspect: str,
    paraphrased_sentence: str,
    embed_model: SentenceTransformer,
    explicit_threshold: float = 0.7,
) -> str | None:
    """Check if the original aspect is still explicitly present in the paraphrase.

    Uses NP embedding similarity — if any noun phrase in the paraphrase is
    too similar to the original aspect, it's still explicit.

    Returns the offending NP string if explicit, None if properly implicit.
    """
    # quick string check first
    if original_aspect.lower() in paraphrased_sentence.lower():
        return original_aspect

    # check all aspect tokens present
    aspect_tokens = set(original_aspect.lower().split())
    para_tokens = set(paraphrased_sentence.lower().split())
    if aspect_tokens and aspect_tokens.issubset(para_tokens):
        return original_aspect

    # NP embedding check
    noun_chunks = _get_noun_chunks(paraphrased_sentence)
    if not noun_chunks:
        return None

    all_texts = [original_aspect] + noun_chunks
    embs = embed_model.encode(all_texts, convert_to_numpy=True)
    aspect_emb = embs[0]

    for j, chunk in enumerate(noun_chunks):
        sim = _cosine_sim(aspect_emb, embs[j + 1])
        if sim >= explicit_threshold:
            return f"{chunk}(sim={sim:.2f})"

    return None


def _check_sentiment_present(original_sentiment: str, paraphrased_sentence: str) -> bool:
    """Check if the original sentiment is present (stem/prefix match)."""
    if not original_sentiment:
        return True
    return _stem_match(original_sentiment, paraphrased_sentence)


def _check_other_aspects_present(
    other_annotations: list[dict],
    paraphrased_sentence: str,
) -> list[str]:
    """Check that non-targeted aspects still appear in the paraphrased sentence.
    Returns list of missing aspect strings.
    """
    missing = []
    para_lower = paraphrased_sentence.lower()
    for ann in other_annotations:
        aspect = ann.get("aspect", "")
        if aspect and aspect != "IMPLICIT" and aspect.lower() not in para_lower:
            missing.append(aspect)
    return missing


def verify_paraphrase(
    original: dict,
    target_idx: int,
    paraphrased_sentence: str,
    verifier_model,
    embed_model: SentenceTransformer,
    aspect_sim_threshold: float = 0.5,
    explicit_threshold: float = 0.7,
    device: str = "cuda",
) -> dict:
    """Verify a paraphrased sentence against the original.

    target_idx: index of the annotation whose aspect was made implicit.
    Other annotations should remain intact.

    Returns dict with:
        "accepted": bool
        "reason": str (failure reason if rejected)
        "extracted": list[dict] (model's extracted triplets)
    """
    target_ann = original["annotations"][target_idx]
    other_anns = [a for i, a in enumerate(original["annotations"]) if i != target_idx]
    original_aspect = target_ann["aspect"]
    original_polarity = target_ann["polarity"]
    original_sentiment = target_ann.get("sentiment", "")

    # Check 0: sentiment preserved (lemmatized)
    if original_sentiment and not _check_sentiment_present(original_sentiment, paraphrased_sentence):
        return {"accepted": False, "reason": "sentiment_missing", "extracted": []}

    # Check 1: targeted aspect should be implicit (NP embedding check)
    explicit_match = _check_aspect_explicit(
        original_aspect, paraphrased_sentence, embed_model, explicit_threshold
    )
    if explicit_match:
        return {"accepted": False, "reason": f"aspect_explicit:{explicit_match}", "extracted": []}

    # Check 2: other aspects should still be present
    if other_anns:
        missing = _check_other_aspects_present(other_anns, paraphrased_sentence)
        if missing:
            return {
                "accepted": False,
                "reason": f"other_aspects_missing:{','.join(missing)}",
                "extracted": [],
            }

    # Check 3-4: run verifier model
    canon = {
        "sentence": paraphrased_sentence,
        "tokens": paraphrased_sentence.split(),
        "annotations": original["annotations"],
    }
    enriched = enrich_syntax([canon], "dep-compact")
    canon = enriched[0]

    input_text = f"Task: aspect, sentiment, polarity\nInput: {canon['sentence']}"
    if "_syntax" in canon:
        input_text += f"\nSyntax: {canon['_syntax']}"
    input_text += "\nOutput: natural language"

    tokenizer = verifier_model.tokenizer
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)

    with torch.no_grad():
        output_ids = verifier_model.model.generate(
            **inputs, max_new_tokens=128, num_beams=1,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    extracted = parse_output(decoded, ["aspect", "sentiment", "polarity"], "natural-language")

    # Check 3: at least one extracted triplet should match the target on polarity
    # and have aspect semantically close to original
    target_matched = False
    for pred in extracted:
        if pred.get("polarity", "").lower() != original_polarity.lower():
            continue
        pred_aspect = pred.get("aspect", "")
        if not pred_aspect:
            continue
        embs = embed_model.encode([original_aspect, pred_aspect], convert_to_numpy=True)
        sim = _cosine_sim(embs[0], embs[1])
        if sim >= aspect_sim_threshold:
            target_matched = True
            break

    if not target_matched:
        if not extracted:
            return {"accepted": False, "reason": "no_triplets_extracted", "extracted": extracted}
        # build a useful reason
        pred_aspects = [p.get("aspect", "?") for p in extracted]
        pred_polarities = [p.get("polarity", "?") for p in extracted]
        return {
            "accepted": False,
            "reason": f"target_not_found:aspects={pred_aspects}:polarities={pred_polarities}",
            "extracted": extracted,
        }

    return {"accepted": True, "reason": "ok", "extracted": extracted}


def build_prompt(
    sentence: str,
    target_aspect: str,
    target_sentiment: str,
    other_aspects: list[str],
    feedback: str = None,
) -> str:
    keep_others = ""
    if other_aspects:
        keep_others = (
            f"\n7. Keep these other aspects exactly as they appear: {', '.join(other_aspects)}"
            "\n8. Only modify the part of the sentence related to the targeted aspect"
        )

    base = (
        "Rewrite the following restaurant review sentence so that the aspect "
        f"'{target_aspect}' is not directly mentioned, but is still clearly implied "
        "through contextual clues. Rules:\n"
        f"1. Keep the opinion words (especially '{target_sentiment}') — do not replace them with synonyms\n"
        "2. The reader should still be able to infer what is being discussed\n"
        "3. Use contextual clues (actions, properties, related concepts) to hint at the aspect\n"
        "4. Do not introduce new aspects or opinions not in the original\n"
        "5. Keep the same sentiment intensity and polarity\n"
        "6. Output ONLY the rewritten sentence, nothing else"
        f"{keep_others}\n"
    )

    if feedback:
        base += f"\nPrevious attempt was rejected: {feedback}\nTry again.\n"

    base += f"\nOriginal: {sentence}\nRewritten:"
    return base


def build_feedback(reason: str, target_ann: dict, other_anns: list[dict]) -> str:
    aspect = target_ann["aspect"]
    sentiment = target_ann.get("sentiment", "")

    if reason == "sentiment_missing":
        return (
            f"The opinion word '{sentiment}' (or its inflected form) must appear "
            "in the rewritten sentence. Do not replace it with a synonym."
        )

    if reason.startswith("aspect_explicit:"):
        match = reason.split(":", 1)[1]
        return (
            f"The aspect '{aspect}' is still explicitly present as '{match}'. "
            "Remove it and use only contextual clues (actions, properties) instead."
        )

    if reason.startswith("other_aspects_missing:"):
        missing = reason.split(":", 1)[1]
        return (
            f"The following aspects must remain in the sentence: {missing}. "
            "Only remove the targeted aspect, keep everything else."
        )

    if reason == "no_triplets_extracted":
        return (
            "The rewritten sentence is too vague — a reader cannot tell what is being "
            f"discussed. Add contextual clues that hint at '{aspect}' without naming it."
        )

    if reason.startswith("target_not_found:"):
        return (
            f"The rewritten sentence doesn't hint at '{aspect}' well enough. "
            f"Add better contextual clues that point to '{aspect}' while keeping "
            f"the {target_ann['polarity']} sentiment."
        )

    if reason.startswith("polarity_mismatch"):
        pred_pol = reason.split(":")[-1]
        return (
            f"The rewritten sentence changed the sentiment from {target_ann['polarity']} "
            f"to {pred_pol}. Keep the original {target_ann['polarity']} sentiment."
        )

    return f"Previous attempt was rejected ({reason}). Try again with better contextual clues."


def call_ollama(prompt: str, model: str, url: str, timeout: int = 300) -> str | None:
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip().strip('"').strip("'").strip()
        text = text.split("\n")[0].strip()
        return text if text else None
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def generate_verified_paraphrases(
    train_files: list[str],
    checkpoint_path: str,
    output_path: str,
    model_name: str = "gemma2:27b",
    ollama_url: str = "http://localhost:11434",
    max_retries: int = 3,
    aspect_sim_threshold: float = 0.5,
    explicit_threshold: float = 0.7,
    device: str = "cuda",
    resume: bool = True,
    verbose: bool = False,
):
    """Generate and verify paraphrases for ASTE sentences.

    For each sentence, picks one explicit-aspect triplet to make implicit.
    Multi-triplet sentences are supported — only the targeted aspect is removed,
    other aspects must remain in the paraphrased sentence.
    """
    # Load data
    all_examples = []
    for f in train_files:
        all_examples.extend(load_aste_file(f))

    # Build candidates: (example, target_annotation_index) pairs
    # Only target explicit aspects
    candidates = []
    for ex in all_examples:
        for idx, ann in enumerate(ex["annotations"]):
            if ann.get("aspect") and ann["aspect"] != "IMPLICIT" and ann.get("polarity"):
                candidates.append((ex, idx))
                break  # one paraphrase per sentence for now

    print(f"Total examples: {len(all_examples)}, candidates: {len(candidates)}")

    # Load models
    print("Loading verifier model...")
    verifier = load_verifier(checkpoint_path, device)
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Warm up spacy
    print("Loading spaCy...")
    _get_nlp()

    # Resume support
    results = []
    done_sentences = set()
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            results = json.load(f)
        done_sentences = {r["original_sentence"] for r in results}
        print(f"Resuming: {len(results)} already done, {len(candidates) - len(done_sentences)} remaining")

    # Verify ollama is reachable
    print(f"Testing ollama connection ({ollama_url}, model={model_name})...")
    test_resp = call_ollama("Say OK", model_name, ollama_url, timeout=300)
    if not test_resp:
        print("ERROR: ollama is not responding. Start ollama and try again.")
        return []
    print(f"  ollama OK: \"{test_resp[:50]}\"")

    stats = {"accepted": 0, "rejected": 0, "errors": 0, "retries": 0}

    for i, (ex, target_idx) in enumerate(candidates):
        if ex["sentence"] in done_sentences:
            continue

        target_ann = ex["annotations"][target_idx]
        other_anns = [a for j, a in enumerate(ex["annotations"]) if j != target_idx]
        aspect = target_ann["aspect"]
        sentiment = target_ann.get("sentiment", "")
        other_aspects = [a["aspect"] for a in other_anns if a.get("aspect") and a["aspect"] != "IMPLICIT"]

        if verbose:
            n_triplets = len(ex["annotations"])
            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(candidates)}] Original: {ex['sentence']}")
            print(f"  Target: {aspect} | Sentiment: {sentiment} | Polarity: {target_ann['polarity']}")
            if other_aspects:
                print(f"  Other aspects (keep): {other_aspects}")
            print(f"  Triplets in sentence: {n_triplets}")

        feedback = None
        accepted = False

        for attempt in range(max_retries + 1):
            prompt = build_prompt(ex["sentence"], aspect, sentiment, other_aspects, feedback)
            paraphrased = call_ollama(prompt, model_name, ollama_url)

            if not paraphrased or paraphrased.lower() == ex["sentence"].lower():
                if verbose:
                    print(f"  Attempt {attempt+1}: LLM returned empty/identical — skipping")
                stats["errors"] += 1
                break

            if verbose:
                print(f"  Attempt {attempt+1}: \"{paraphrased}\"")

            result = verify_paraphrase(
                ex, target_idx, paraphrased, verifier, embed_model,
                aspect_sim_threshold=aspect_sim_threshold,
                explicit_threshold=explicit_threshold,
                device=device,
            )

            if result["accepted"]:
                if verbose:
                    print(f"    ✓ ACCEPTED — verifier extracted: {result['extracted']}")

                # Build output annotations: targeted one becomes IMPLICIT, others stay
                out_anns = []
                for j, ann in enumerate(ex["annotations"]):
                    new_ann = dict(ann)
                    if j == target_idx:
                        new_ann["aspect_original"] = aspect
                        new_ann["aspect"] = "IMPLICIT"
                        new_ann["aspect_idx"] = None
                    out_anns.append(new_ann)

                results.append({
                    "original_sentence": ex["sentence"],
                    "paraphrased_sentence": paraphrased,
                    "annotations": out_anns,
                    "attempts": attempt + 1,
                    "verified_extraction": result["extracted"],
                })
                stats["accepted"] += 1
                accepted = True
                break
            else:
                feedback = build_feedback(result["reason"], target_ann, other_anns)
                stats["retries"] += 1
                if verbose:
                    ext = result["extracted"]
                    print(f"    ✗ REJECTED ({result['reason']})")
                    if ext:
                        print(f"      Verifier extracted: {ext}")
                    print(f"      Feedback: {feedback}")

        if not accepted and feedback is not None:
            stats["rejected"] += 1
            if verbose:
                print(f"  FINAL: rejected after {max_retries+1} attempts")

        # Save incrementally every 50 examples
        total_processed = stats["accepted"] + stats["rejected"] + stats["errors"]
        if total_processed % 50 == 0 or i == len(candidates) - 1:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(
                f"  [{total_processed}/{len(candidates)}] "
                f"accepted={stats['accepted']} rejected={stats['rejected']} "
                f"errors={stats['errors']} retries={stats['retries']}"
            )

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} verified paraphrases saved to {output_path}")
    print(f"Stats: {stats}")
    return results
