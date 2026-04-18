"""LLM-as-judge evaluation using Gemini API.

Evaluates generated summaries on coherence, faithfulness to the source article,
and fluency, using a structured prompt that returns numeric scores (1-5) with
brief justifications. Complements ROUGE metrics with qualitative assessment.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import google.generativeai as genai
import pandas as pd


@dataclass
class JudgeResult:
    """Structured evaluation result for a single summary."""
    index: int
    model: str
    coherence: int        # 1-5
    faithfulness: int     # 1-5
    fluency: int          # 1-5
    justification: str
    error: str = ""


JUDGE_PROMPT = """\
You are an expert evaluator of news article summaries. Given an original article \
and a machine-generated summary, rate the summary on three dimensions using a \
1-5 scale (1=terrible, 5=excellent).

**Coherence (1-5):** Is the summary logically organized and easy to follow? \
Are there contradictions, repetitions, or disjointed sentences?

**Faithfulness (1-5):** Does the summary accurately reflect the source article? \
Is there any hallucinated information not present in the original?

**Fluency (1-5):** Is the summary grammatically correct and reads naturally? \
Does it sound like it was written by a professional journalist?

Respond ONLY with valid JSON in exactly this format, no markdown fences:
{{"coherence": <int>, "faithfulness": <int>, "fluency": <int>, "justification": "<brief explanation>"}}

---
ARTICLE:
{article}

---
SUMMARY:
{summary}
"""


def init_gemini(api_key: str | None = None, model_name: str = "gemini-2.0-flash") -> genai.GenerativeModel:
    """Initialize the Gemini client.

    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        model_name: Gemini model to use. Flash is fast and cheap.

    Returns:
        Configured GenerativeModel instance.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
        )
    genai.configure(api_key=key)
    return genai.GenerativeModel(model_name)


def judge_single(
    gemini_model: genai.GenerativeModel,
    article: str,
    summary: str,
    index: int,
    model_name: str,
    max_article_chars: int = 3000,
    max_retries: int = 2,
) -> JudgeResult:
    """Evaluate a single article-summary pair with the LLM judge.

    Truncates the article to max_article_chars to stay within token limits
    and reduce cost. The summary is passed in full.

    Args:
        gemini_model: Initialized Gemini model.
        article: Source news article.
        summary: Generated summary to evaluate.
        index: Sample index for tracking.
        model_name: Label for which model generated the summary.
        max_article_chars: Max chars of article to include in prompt.
        max_retries: Retry count on API/parse failures.

    Returns:
        JudgeResult with scores and justification.
    """
    prompt = JUDGE_PROMPT.format(
        article=article[:max_article_chars],
        summary=summary,
    )

    for attempt in range(max_retries + 1):
        try:
            response = gemini_model.generate_content(prompt)
            text = response.text.strip()
            # Strip markdown fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            return JudgeResult(
                index=index,
                model=model_name,
                coherence=int(data["coherence"]),
                faithfulness=int(data["faithfulness"]),
                fluency=int(data["fluency"]),
                justification=data.get("justification", ""),
            )
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return JudgeResult(
                index=index,
                model=model_name,
                coherence=0,
                faithfulness=0,
                fluency=0,
                justification="",
                error=f"{type(e).__name__}: {e}",
            )


def judge_batch(
    gemini_model: genai.GenerativeModel,
    articles: list[str],
    summaries: list[str],
    model_name: str,
    delay: float = 1.0,
) -> list[JudgeResult]:
    """Evaluate a batch of article-summary pairs sequentially.

    Includes a configurable delay between calls to respect rate limits.
    Progress is printed to stdout.

    Args:
        gemini_model: Initialized Gemini model.
        articles: Source articles.
        summaries: Generated summaries (same length as articles).
        model_name: Label for the model that generated summaries.
        delay: Seconds to wait between API calls.

    Returns:
        List of JudgeResult, one per sample.
    """
    results = []
    total = len(articles)
    for i, (art, summ) in enumerate(zip(articles, summaries)):
        result = judge_single(gemini_model, art, summ, index=i, model_name=model_name)
        results.append(result)
        status = "✅" if not result.error else f"❌ {result.error}"
        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  [{model_name}] {i+1}/{total} {status}")
        if i < total - 1:
            time.sleep(delay)
    return results


def results_to_dataframe(results: list[JudgeResult]) -> pd.DataFrame:
    """Convert a list of JudgeResult into a DataFrame.

    Filters out failed evaluations (scores = 0) for aggregate statistics.
    """
    rows = []
    for r in results:
        rows.append({
            "index": r.index,
            "model": r.model,
            "coherence": r.coherence,
            "faithfulness": r.faithfulness,
            "fluency": r.fluency,
            "justification": r.justification,
            "error": r.error,
        })
    df = pd.DataFrame(rows)
    return df


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean scores per model, excluding failed evaluations."""
    valid = df[df["error"] == ""].copy()
    return valid.groupby("model")[["coherence", "faithfulness", "fluency"]].agg(
        ["mean", "std", "count"]
    ).round(2)
