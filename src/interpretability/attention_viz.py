"""Attention map extraction and visualization for summarization models.

Supports both encoder-decoder (T5) cross-attention and causal (Qwen3)
self-attention visualization. Generates heatmaps showing which input tokens
the model attends to when producing each summary token.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def extract_seq2seq_attention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    article: str,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> dict:
    """Extract cross-attention from an encoder-decoder model during generation.

    Returns input tokens, output tokens, and the averaged cross-attention matrix
    (mean across all layers and heads).

    Args:
        model: Seq2seq model (T5, BART, Pegasus).
        tokenizer: Corresponding tokenizer.
        article: Source article text.
        max_input_length: Truncation limit for the input.
        max_new_tokens: Max tokens to generate.
        device: 'cuda' or 'cpu'.

    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'attention' (2D numpy array).
    """
    prefix = "summarize: " if "t5" in tokenizer.name_or_path.lower() else ""
    inputs = tokenizer(
        prefix + article,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,  # greedy for clean attention maps
        output_attentions=True,
        return_dict_in_generate=True,
    )

    # Extract cross-attention: list of (layers, heads, 1, src_len) per generated token
    cross_attns = outputs.cross_attentions  # tuple of tuples
    if not cross_attns:
        raise ValueError("Model did not return cross-attentions. Check model config.")

    # Average across layers and heads for each generated token step
    attn_per_step = []
    for step_attns in cross_attns:
        # step_attns: tuple of (batch, heads, tgt_len, src_len) per layer
        stacked = torch.stack([layer[0].mean(dim=0) for layer in step_attns])
        # stacked: (n_layers, tgt_len, src_len) → mean across layers
        avg = stacked.mean(dim=0)[-1]  # last tgt position, shape: (src_len,)
        attn_per_step.append(avg.cpu().numpy())

    attention = np.stack(attn_per_step)  # (gen_len, src_len)

    input_ids = inputs["input_ids"][0].cpu().tolist()
    output_ids = outputs.sequences[0].cpu().tolist()

    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids[1:])  # skip BOS/pad

    # Trim to actual generated length
    attention = attention[:len(output_tokens)]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attention": attention,
    }


def plot_attention_heatmap(
    attn_data: dict,
    title: str = "Cross-attention heatmap",
    max_input_tokens: int = 60,
    max_output_tokens: int = 30,
    figsize: tuple = (14, 6),
    save_path: Path | None = None,
) -> None:
    """Plot a heatmap of attention weights.

    Truncates both axes to keep the plot readable. Input tokens (x-axis)
    are shown from the beginning of the article; output tokens (y-axis)
    are the first N generated tokens.

    Args:
        attn_data: Output of extract_seq2seq_attention().
        title: Plot title.
        max_input_tokens: Max input tokens to display on x-axis.
        max_output_tokens: Max output tokens to display on y-axis.
        figsize: Figure dimensions.
        save_path: If provided, saves the figure to this path.
    """
    attn = attn_data["attention"][:max_output_tokens, :max_input_tokens]
    in_tok = attn_data["input_tokens"][:max_input_tokens]
    out_tok = attn_data["output_tokens"][:max_output_tokens]

    # Clean up token display (remove sentencepiece underscores)
    clean = lambda t: t.replace("▁", " ").replace("Ġ", " ").strip()
    in_tok = [clean(t) for t in in_tok]
    out_tok = [clean(t) for t in out_tok]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        attn,
        xticklabels=in_tok,
        yticklabels=out_tok,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Attention weight"},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Input tokens (article)", fontsize=11)
    ax.set_ylabel("Output tokens (summary)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_token_importance(
    attn_data: dict,
    title: str = "Token importance (summed attention)",
    top_k: int = 25,
    figsize: tuple = (10, 5),
    save_path: Path | None = None,
) -> None:
    """Plot the most attended input tokens across all generated tokens.

    Sums attention weights across the output sequence to identify which
    input tokens contributed most to the summary overall.

    Args:
        attn_data: Output of extract_seq2seq_attention().
        title: Plot title.
        top_k: Number of top tokens to display.
        figsize: Figure dimensions.
        save_path: If provided, saves the figure.
    """
    attn = attn_data["attention"]
    total_attn = attn.sum(axis=0)  # sum across output tokens
    in_tok = attn_data["input_tokens"]

    clean = lambda t: t.replace("▁", " ").replace("Ġ", " ").strip()
    in_tok = [clean(t) for t in in_tok]

    # Get top-k indices
    top_idx = np.argsort(total_attn)[-top_k:][::-1]
    top_tokens = [in_tok[i] for i in top_idx]
    top_scores = total_attn[top_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_tokens)), top_scores[::-1], color="coral")
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens[::-1], fontsize=9)
    ax.set_xlabel("Cumulative attention weight")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
