import pytest
from synthetic_builder import generate_synthetic_samples
from complex_sentence_injector import generate_complex_samples


def test_synthetic_data_generation_balance_and_formatting():
    """Verify synthetic builder outputs balanced, properly formatted financial sentences."""
    samples = generate_synthetic_samples(samples_per_class=50)
    assert len(samples) == 150

    labels = [s["label"] for s in samples]
    assert labels.count("negative") == 50
    assert labels.count("neutral") == 50
    assert labels.count("positive") == 50

    # Ensure no unformatted curly braces remain in strings
    for s in samples:
        text = s["text"]
        assert "{" not in text and "}" not in text
        assert len(text) > 20

    # Verify presence of injected institutional distress vocabulary
    neg_texts = " ".join([s["text"] for s in samples if s["label"] == "negative"])
    assert any(w in neg_texts.lower() for w in ["covenant", "impairment", "burn", "restructuring", "loss", "warning"])


def test_complex_sentence_injector_nuance_and_polarity():
    """Verify complex sentence injector produces multi-clause sentences with valid structure."""
    samples = generate_complex_samples(samples_per_class=40)
    assert len(samples) == 120

    labels = [s["label"] for s in samples]
    assert labels.count("negative") == 40
    assert labels.count("neutral") == 40
    assert labels.count("positive") == 40

    # Verify that complex negatives contain concessive discourse conjunctions
    complex_negs = [s["text"] for s in samples if s["label"] == "negative"]
    concessive_markers = ["although", "despite", "while", "even though", "notwithstanding"]
    has_marker = sum(any(marker in t.lower() for marker in concessive_markers) for t in complex_negs)
    assert has_marker == len(complex_negs), "All complex negatives must contain concessive markers"

    # Verify no unformatted template strings
    for s in samples:
        assert "{" not in s["text"] and "}" not in s["text"]
