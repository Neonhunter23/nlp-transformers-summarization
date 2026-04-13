"""Tokenization and preprocessing for summarization models."""
from typing import Callable
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict


def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer with sensible defaults.

    Args:
        model_name: HuggingFace model identifier.
        padding_side: 'right' for seq2seq, 'left' for causal generation.

    Returns:
        Initialized tokenizer with pad token and padding side set.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = padding_side
    # Causal LMs (Qwen, GPT-style) often lack a pad token; reuse EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_seq2seq_preprocess_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    max_target_length: int,
    prefix: str = "summarize: ",
) -> Callable:
    """Build a preprocessing function for encoder-decoder models (T5, BART).

    T5 expects a task prefix; BART/Pegasus ignore it harmlessly.

    Args:
        tokenizer: Model tokenizer.
        max_input_length: Max tokens for the article.
        max_target_length: Max tokens for the summary.
        prefix: Task instruction prepended to each input.

    Returns:
        Function mapping a batch of examples to model inputs.
    """
    def preprocess(examples: dict) -> dict:
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding=False,  # dynamic padding handled by the data collator
        )
        labels = tokenizer(
            text_target=examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def build_causal_preprocess_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    max_target_length: int,
) -> Callable:
    """Build a preprocessing function for causal LMs (Qwen3, GPT-style).

    Causal LMs receive a single concatenated prompt-completion sequence,
    where the loss is masked over the prompt tokens.

    Args:
        tokenizer: Model tokenizer.
        max_input_length: Max tokens for the article portion.
        max_target_length: Max tokens for the summary portion.

    Returns:
        Function mapping a batch of examples to model inputs.
    """
    def preprocess(examples: dict) -> dict:
        input_ids_batch, attention_batch, labels_batch = [], [], []

        for article, summary in zip(examples["article"], examples["highlights"]):
            prompt = (
                f"Summarize the following news article:\n\n{article}\n\nSummary:"
            )
            # Tokenize prompt and completion separately to know prompt length.
            prompt_ids = tokenizer(
                prompt, max_length=max_input_length, truncation=True
            )["input_ids"]
            completion_ids = tokenizer(
                " " + summary,
                max_length=max_target_length,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            completion_ids = completion_ids + [tokenizer.eos_token_id]

            input_ids = prompt_ids + completion_ids
            # Mask prompt tokens with -100 so loss is computed only on the summary.
            labels = [-100] * len(prompt_ids) + completion_ids
            attention = [1] * len(input_ids)

            input_ids_batch.append(input_ids)
            attention_batch.append(attention)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_batch,
            "labels": labels_batch,
        }

    return preprocess


def tokenize_dataset(
    dataset: DatasetDict,
    preprocess_fn: Callable,
    remove_original_columns: bool = True,
) -> DatasetDict:
    """Apply a preprocessing function to all splits of a DatasetDict.

    Args:
        dataset: Raw DatasetDict with 'article' and 'highlights' columns.
        preprocess_fn: Function returned by one of the build_*_preprocess_fn.
        remove_original_columns: Whether to drop raw text columns after tokenizing.

    Returns:
        Tokenized DatasetDict ready for a Trainer.
    """
    columns_to_remove = (
        dataset["train"].column_names if remove_original_columns else None
    )
    return dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing",
    )