"""Evaluation metrics for summarization."""
from typing import Sequence
import evaluate


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict:
    """Compute ROUGE-1, ROUGE-2 and ROUGE-L scores.

    Args:
        predictions: Generated summaries.
        references: Ground-truth summaries.

    Returns:
        Dict with keys 'rouge1', 'rouge2', 'rougeL', 'rougeLsum' (F1 scores, 0-100).
    """
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=list(predictions),
        references=list(references),
        use_stemmer=True,
    )
    # evaluate returns values in [0,1]; rescale to [0,100] for readability.
    return {k: round(v * 100, 2) for k, v in scores.items()}


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "en",
) -> dict:
    """Compute BERTScore precision, recall and F1.

    Semantic similarity metric that complements ROUGE's lexical overlap.

    Args:
        predictions: Generated summaries.
        references: Ground-truth summaries.
        lang: Language code for the underlying model.

    Returns:
        Dict with mean precision, recall and F1 (0-100).
    """
    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=list(predictions),
        references=list(references),
        lang=lang,
    )
    return {
        "bertscore_precision": round(sum(scores["precision"]) / len(scores["precision"]) * 100, 2),
        "bertscore_recall": round(sum(scores["recall"]) / len(scores["recall"]) * 100, 2),
        "bertscore_f1": round(sum(scores["f1"]) / len(scores["f1"]) * 100, 2),
    }