"""Unified model loading for seq2seq and causal LMs."""
from dataclasses import dataclass
from typing import Literal
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.data.preprocessing import load_tokenizer


ModelType = Literal["seq2seq", "causal"]


@dataclass
class LoadedModel:
    """Container bundling a model with its tokenizer and metadata."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    name: str
    model_type: ModelType
    max_input_length: int


def load_model(
    model_cfg: dict,
    device: str = "cuda",
) -> LoadedModel:
    """Load a model and its tokenizer according to a config entry.

    Reads dtype and padding requirements from the model config to ensure
    each architecture loads with the settings appropriate to its type.

    Args:
        model_cfg: Dict with keys 'name', 'type', 'max_input_length',
            and optionally 'dtype' ('float32' | 'float16' | 'bfloat16').
        device: Target device ('cuda' or 'cpu').

    Returns:
        LoadedModel dataclass ready for inference.
    """
    name = model_cfg["name"]
    model_type: ModelType = model_cfg["type"]

    # Resolve dtype from config; default to bf16 for modern training.
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[model_cfg.get("dtype", "bfloat16")]

    # Causal LMs require left-padding for correct generation.
    padding_side = "left" if model_type == "causal" else "right"
    tokenizer = load_tokenizer(name, padding_side=padding_side)

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=dtype)
    elif model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        model_type=model_type,
        max_input_length=model_cfg["max_input_length"],
    )


def free_model(loaded: LoadedModel) -> None:
    """Release a loaded model from VRAM.

    Essential when iterating over multiple models on a single GPU.
    """
    del loaded.model
    del loaded.tokenizer
    torch.cuda.empty_cache()