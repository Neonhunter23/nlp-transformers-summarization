"""Inference utilities for generating summaries with different model types."""
from typing import Sequence
import torch
from tqdm.auto import tqdm

from src.models.loader import LoadedModel


@torch.no_grad()
def generate_summaries(
    loaded: LoadedModel,
    articles: Sequence[str],
    max_new_tokens: int = 128,
    num_beams: int = 4,
    batch_size: int = 4,
    generation_kwargs: dict | None = None,
) -> list[str]:
    """Generate summaries for a list of articles.

    Handles both seq2seq (T5, BART, Pegasus) and causal (Qwen3) models
    with a unified interface.

    Args:
        loaded: LoadedModel from src.models.loader.
        articles: List of article texts to summarize.
        max_new_tokens: Maximum summary length in tokens (ignored for
            models with a task-tuned generation config).
        num_beams: Beam search width (1 = greedy).
        batch_size: Batch size for generation.
        generation_kwargs: Extra kwargs forwarded to model.generate().

    Returns:
        List of generated summary strings, one per input article.
    """
    # Models fine-tuned for summarization (BART-cnn, Pegasus-cnn) ship optimized
    # generation configs (num_beams, length_penalty, min_length, max_length).
    # We respect them fully and only control max_new_tokens for generalist models.
    has_tuned_config = any(
        tag in loaded.name.lower() for tag in ["bart-large-cnn", "pegasus-cnn"]
    )

    if has_tuned_config:
        # Pass nothing: let the model use its own native generation_config.
        gen_kwargs = {}
    else:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "no_repeat_ngram_size": 3,
            "length_penalty": 2.0,
            "early_stopping": True,
        }
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)

    device = next(loaded.model.parameters()).device
    results: list[str] = []

    for start in tqdm(range(0, len(articles), batch_size), desc=f"Generating [{loaded.name}]"):
        batch = list(articles[start:start + batch_size])

        if loaded.model_type == "seq2seq":
            # T5 benefits from the "summarize: " prefix; harmless for BART/Pegasus.
            prefix = "summarize: " if "t5" in loaded.name.lower() else ""
            inputs = loaded.tokenizer(
                [prefix + a for a in batch],
                max_length=loaded.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = loaded.model.generate(**inputs, **gen_kwargs)
            decoded = loaded.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        else:  # causal
            prompts = [
                f"Summarize the following news article:\n\n{a}\n\nSummary:"
                for a in batch
            ]
            inputs = loaded.tokenizer(
                prompts,
                max_length=loaded.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = loaded.model.generate(
                **inputs,
                pad_token_id=loaded.tokenizer.pad_token_id,
                **gen_kwargs,
            )
            # Strip the prompt from the generated output.
            prompt_len = inputs["input_ids"].shape[1]
            generated_only = outputs[:, prompt_len:]
            decoded = loaded.tokenizer.batch_decode(generated_only, skip_special_tokens=True)

        # Pegasus emits literal "<n>" tokens as sentence separators;
        # normalize them to real newlines so ROUGE matches the references.
        if "pegasus" in loaded.name.lower():
            decoded = [d.replace("<n>", "\n") for d in decoded]

        results.extend([d.strip() for d in decoded])

    return results