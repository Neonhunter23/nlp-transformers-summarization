"""LLM-as-judge evaluation using a local model via transformers.

Evaluates generated summaries on coherence, faithfulness to the source article,
and fluency, using a structured prompt that requests numeric scores (1-5) with
brief justifications. Runs entirely locally — no API keys or rate limits.

Default judge model: Qwen3-1.7B base (not the LoRA-adapted version being
evaluated). This is methodologically acceptable because the judge is the
general-purpose base model, while the evaluated outputs come from the
task-specific LoRA adapter — they are functionally different models.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

JSON response:"""


def init_judge(
    model_name: str = "Qwen/Qwen3-1.7B",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the judge model and tokenizer.

    Args:
        model_name: HuggingFace model identifier for the judge.
        dtype: Torch dtype for model weights.
        device: Target device.

    Returns:
        Tuple of (model, tokenizer) ready for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    print(f"Judge model loaded: {model_name} ({dtype})")
    return model, tokenizer


def free_judge(model, tokenizer) -> None:
    """Release judge model VRAM."""
    del model, tokenizer
    torch.cuda.empty_cache()


def _extract_json(text: str) -> dict:
    """Extract JSON from model output, handling common formatting issues."""
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r'\{[^{}]*"coherence"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from: {text[:200]}")


@torch.no_grad()
def judge_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    article: str,
    summary: str,
    index: int,
    model_label: str,
    max_article_chars: int = 2000,
    max_new_tokens: int = 150,
) -> JudgeResult:
    """Evaluate a single article-summary pair with the local judge.

    Args:
        model: Judge model.
        tokenizer: Judge tokenizer.
        article: Source news article (truncated to max_article_chars).
        summary: Generated summary to evaluate.
        index: Sample index for tracking.
        model_label: Label for which model generated the summary.
        max_article_chars: Max chars of article to include in prompt.
        max_new_tokens: Max tokens for the judge's response.

    Returns:
        JudgeResult with scores and justification.
    """
    prompt = JUDGE_PROMPT.format(
        article=article[:max_article_chars],
        summary=summary,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Strip the prompt from the output
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0, prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        data = _extract_json(text)

        return JudgeResult(
            index=index,
            model=model_label,
            coherence=int(data["coherence"]),
            faithfulness=int(data["faithfulness"]),
            fluency=int(data["fluency"]),
            justification=data.get("justification", ""),
        )
    except Exception as e:
        return JudgeResult(
            index=index,
            model=model_label,
            coherence=0,
            faithfulness=0,
            fluency=0,
            justification="",
            error=f"{type(e).__name__}: {e}",
        )


def judge_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    articles: list[str],
    summaries: list[str],
    model_label: str,
) -> list[JudgeResult]:
    """Evaluate a batch of article-summary pairs sequentially.

    No delay needed — everything runs locally.

    Args:
        model: Judge model.
        tokenizer: Judge tokenizer.
        articles: Source articles.
        summaries: Generated summaries (same length as articles).
        model_label: Label for the model that generated summaries.

    Returns:
        List of JudgeResult, one per sample.
    """
    results = []
    total = len(articles)
    errors = 0
    for i, (art, summ) in enumerate(zip(articles, summaries)):
        result = judge_single(
            model, tokenizer, art, summ,
            index=i, model_label=model_label,
        )
        results.append(result)
        if result.error:
            errors += 1
        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  [{model_label}] {i+1}/{total} ({errors} errors so far)")
    return results


def results_to_dataframe(results: list[JudgeResult]) -> pd.DataFrame:
    """Convert a list of JudgeResult into a DataFrame."""
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
    return pd.DataFrame(rows)


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean scores per model, excluding failed evaluations."""
    valid = df[df["error"] == ""].copy()
    return valid.groupby("model")[["coherence", "faithfulness", "fluency"]].agg(
        ["mean", "std", "count"]
    ).round(2)