"""Test error_analysis_v2 classify_span and analyze_sentence with mock data."""

import sys
sys.path.insert(0, ".")

import numpy as np
from tools.error_analysis_v2 import classify_span, classify_missed_span, analyze_sentence, spans_overlap


def make_embeddings(word_pairs_similar, all_words):
    """Create fake embeddings where specified pairs are similar (cosine > 0.8)."""
    embs = {}
    dim = 32
    rng = np.random.RandomState(42)
    for w in all_words:
        embs[w.lower()] = rng.randn(dim)
        embs[w.lower()] /= np.linalg.norm(embs[w.lower()])
    # make specified pairs similar
    for a, b in word_pairs_similar:
        base = rng.randn(dim)
        base /= np.linalg.norm(base)
        noise = rng.randn(dim) * 0.05
        embs[a.lower()] = base
        embs[b.lower()] = base + noise
        embs[b.lower()] /= np.linalg.norm(embs[b.lower()])
    return embs


def test_spans_overlap():
    assert spans_overlap("so horrible", "horrible")
    assert spans_overlap("battery life", "battery")
    assert not spans_overlap("food", "service")
    assert spans_overlap("great pizza", "pizza is great")  # shares "pizza" and "great"
    print("  spans_overlap: OK")


def test_classify_span_correct():
    golds = ["food", "service"]
    sentence = "The food was great but the service was slow."
    embs = make_embeddings([], ["food", "service"])
    cls, g = classify_span("food", golds, sentence, embs)
    assert cls == "correct" and g == "food"
    print("  classify_span correct: OK")


def test_classify_span_boundary():
    golds = ["horrible"]
    sentence = "The staff was so horrible to us."
    embs = make_embeddings([], ["horrible", "so horrible"])
    cls, g = classify_span("so horrible", golds, sentence, embs)
    assert cls == "boundary" and g == "horrible", f"Got {cls}, {g}"
    print("  classify_span boundary: OK")


def test_classify_span_spurious():
    golds = ["food"]
    sentence = "The food was great and the service was slow."
    embs = make_embeddings([], ["food", "service"])
    cls, g = classify_span("service", golds, sentence, embs)
    assert cls == "spurious" and g is None, f"Got {cls}, {g}"
    print("  classify_span spurious: OK")


def test_classify_span_paraphrase():
    golds = ["horrible"]
    sentence = "The staff was so horrible to us."
    # "horrid" is NOT in the sentence, but is similar to "horrible"
    embs = make_embeddings([("horrible", "horrid")], ["horrible", "horrid"])
    cls, g = classify_span("horrid", golds, sentence, embs)
    assert cls == "paraphrase" and g == "horrible", f"Got {cls}, {g}"
    print("  classify_span paraphrase: OK")


def test_classify_span_hallucination():
    golds = ["food"]
    sentence = "The food was great."
    # "keyboard" not in sentence and not similar to "food"
    embs = make_embeddings([], ["food", "keyboard"])
    cls, g = classify_span("keyboard", golds, sentence, embs)
    assert cls == "hallucination" and g is None, f"Got {cls}, {g}"
    print("  classify_span hallucination: OK")


def test_classify_missed_span():
    train_vocab = {"food", "service", "great"}
    assert classify_missed_span("food", train_vocab) == "seen"
    assert classify_missed_span("keyboard", train_vocab) == "unseen"
    print("  classify_missed_span: OK")


def test_analyze_sentence_all_correct():
    pred = [{"aspect": "food", "sentiment": "great", "polarity": "positive"}]
    gold = [{"aspect": "food", "sentiment": "great", "polarity": "positive"}]
    sentence = "The food was great."
    embs = make_embeddings([], ["food", "great"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["n_correct"] == 1
    assert len(errs["aspect_errors"]) == 0
    assert len(errs["opinion_errors"]) == 0
    assert len(errs["polarity_errors"]) == 0
    print("  analyze_sentence all correct: OK")


def test_analyze_sentence_polarity_error():
    pred = [{"aspect": "food", "sentiment": "great", "polarity": "negative"}]
    gold = [{"aspect": "food", "sentiment": "great", "polarity": "positive"}]
    sentence = "The food was great."
    embs = make_embeddings([], ["food", "great"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["n_correct"] == 0
    assert len(errs["polarity_errors"]) == 1
    assert errs["polarity_errors"][0]["pred"] == "negative"
    assert errs["polarity_errors"][0]["gold"] == "positive"
    print("  analyze_sentence polarity error: OK")


def test_analyze_sentence_boundary():
    pred = [{"aspect": "food", "sentiment": "so great", "polarity": "positive"}]
    gold = [{"aspect": "food", "sentiment": "great", "polarity": "positive"}]
    sentence = "The food was so great."
    embs = make_embeddings([], ["food", "great", "so great"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["n_correct"] == 0  # not an exact match
    # aspect is correct (exact match with a gold aspect)
    assert len(errs["aspect_errors"]) == 0
    # opinion is boundary error
    assert len(errs["opinion_errors"]) == 1
    assert errs["opinion_errors"][0]["type"] == "boundary"
    print("  analyze_sentence opinion boundary: OK")


def test_analyze_sentence_missed():
    pred = []
    gold = [{"aspect": "keyboard", "sentiment": "responsive", "polarity": "positive"}]
    sentence = "The keyboard is very responsive."
    embs = make_embeddings([], ["keyboard", "responsive"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["n_correct"] == 0
    assert len(errs["missed_aspects"]) == 1
    assert errs["missed_aspects"][0]["type"] == "unseen"
    assert len(errs["missed_opinions"]) == 1
    assert errs["missed_opinions"][0]["type"] == "unseen"
    print("  analyze_sentence missed (unseen): OK")


def test_triplet_pattern_polarity_only():
    """Aspect and opinion correct, only polarity wrong -> pattern is 'polarity'."""
    pred = [{"aspect": "food", "sentiment": "great", "polarity": "negative"}]
    gold = [{"aspect": "food", "sentiment": "great", "polarity": "positive"}]
    sentence = "The food was great."
    embs = make_embeddings([], ["food", "great"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["triplet_patterns"] == ["polarity"], f"Got {errs['triplet_patterns']}"
    print("  triplet_pattern polarity only: OK")


def test_triplet_pattern_opinion_only():
    """Aspect and polarity correct, opinion is boundary error -> pattern is 'opinion'."""
    pred = [{"aspect": "food", "sentiment": "so great", "polarity": "positive"}]
    gold = [{"aspect": "food", "sentiment": "great", "polarity": "positive"}]
    sentence = "The food was so great."
    embs = make_embeddings([], ["food", "great", "so great"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["triplet_patterns"] == ["opinion"], f"Got {errs['triplet_patterns']}"
    print("  triplet_pattern opinion only: OK")


def test_triplet_pattern_aspect_plus_polarity():
    """Aspect wrong (boundary), polarity wrong, opinion correct -> 'aspect+polarity'."""
    pred = [{"aspect": "battery life", "sentiment": "great", "polarity": "negative"}]
    gold = [{"aspect": "battery", "sentiment": "great", "polarity": "positive"}]
    sentence = "The battery life was great."
    embs = make_embeddings([], ["battery", "battery life", "great"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"battery"}, {"great"})
    assert errs["triplet_patterns"] == ["aspect+polarity"], f"Got {errs['triplet_patterns']}"
    print("  triplet_pattern aspect+polarity: OK")


def test_triplet_pattern_all_three():
    """All three wrong -> 'aspect+opinion+polarity'."""
    pred = [{"aspect": "keyboard", "sentiment": "terrible", "polarity": "positive"}]
    gold = [{"aspect": "food", "sentiment": "great", "polarity": "negative"}]
    sentence = "The food was great."
    # keyboard not in sentence, terrible not in sentence, neither similar to golds
    embs = make_embeddings([], ["food", "great", "keyboard", "terrible"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert errs["triplet_patterns"] == ["aspect+opinion+polarity"], f"Got {errs['triplet_patterns']}"
    print("  triplet_pattern all three: OK")


def test_triplet_pattern_multiple_preds():
    """Two wrong predictions in same sentence produce two patterns."""
    pred = [
        {"aspect": "food", "sentiment": "so great", "polarity": "positive"},  # opinion boundary
        {"aspect": "keyboard", "sentiment": "great", "polarity": "negative"},  # aspect hallucination + polarity
    ]
    gold = [
        {"aspect": "food", "sentiment": "great", "polarity": "positive"},
    ]
    sentence = "The food was so great."
    embs = make_embeddings([], ["food", "great", "so great", "keyboard"])
    errs = analyze_sentence(pred, gold, sentence, embs, {"food"}, {"great"})
    assert len(errs["triplet_patterns"]) == 2
    assert "opinion" in errs["triplet_patterns"]
    assert "aspect+polarity" in errs["triplet_patterns"]
    print("  triplet_pattern multiple preds: OK")


if __name__ == "__main__":
    print("Running tests...")
    test_spans_overlap()
    test_classify_span_correct()
    test_classify_span_boundary()
    test_classify_span_spurious()
    test_classify_span_paraphrase()
    test_classify_span_hallucination()
    test_classify_missed_span()
    test_analyze_sentence_all_correct()
    test_analyze_sentence_polarity_error()
    test_analyze_sentence_boundary()
    test_analyze_sentence_missed()
    test_triplet_pattern_polarity_only()
    test_triplet_pattern_opinion_only()
    test_triplet_pattern_aspect_plus_polarity()
    test_triplet_pattern_all_three()
    test_triplet_pattern_multiple_preds()
    print("\nAll tests passed!")
