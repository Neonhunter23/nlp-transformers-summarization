"""Reusable experiment runner for V3 hyperparameter search.

Encapsulates the full pipeline for a single configuration: build trainer,
train, evaluate on the test subset, and persist results to CSV.

The runner is model-family agnostic (handles both seq2seq and causal via
LoRA/full fine-tuning) so the same interface works for Flan-T5 and
Qwen3 experiments.

Features:
- Skip already-completed experiments (CSV exists) for safe re-runs.
- Resume interrupted training from the last saved checkpoint.
- Aggressive VRAM cleanup between experiments.
- Fault-tolerant suite runner: if one experiment crashes, the rest continue.
"""
from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.loader import load_cnn_dailymail
from src.data.preprocessing import (
    build_causal_preprocess_fn,
    build_seq2seq_preprocess_fn,
    tokenize_dataset,
)
from src.evaluation.inference import generate_summaries
from src.evaluation.metrics import compute_rouge
from src.models.loader import LoadedModel, load_model
from src.training.trainer import (
    apply_lora,
    build_causal_trainer,
    build_seq2seq_trainer,
)


@dataclass
class ExperimentSpec:
    """Specification for a single V3 experiment.

    Args:
        name: Short label used as subfolder name and CSV row index.
        model_key: Key in cfg['models'] ('t5' or 'qwen').
        train_subset: Number of training samples to use.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        per_device_batch_size: Micro-batch size per GPU step.
        gradient_accumulation_steps: Accumulation factor for effective batch.
        train_max_input: Override for max_input_length during training.
            If None, uses the model's config value. Useful for Qwen3 to fit
            in VRAM without sacrificing inference context.
        lora_overrides: Dict of LoRA config overrides (rank, target_modules, etc.).
            Only applies when model_key is a causal LM.
        extra_trainer_kwargs: Extra kwargs passed to the trainer builder.
    """
    name: str
    model_key: str
    train_subset: int = 10000
    num_epochs: int = 2
    learning_rate: float = 3e-5
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    train_max_input: int | None = None
    lora_overrides: dict[str, Any] = field(default_factory=dict)
    extra_trainer_kwargs: dict[str, Any] = field(default_factory=dict)


def _free_gpu_memory() -> None:
    """Aggressive VRAM cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _restore_inference_mode(loaded: LoadedModel) -> None:
    """Restore clean inference state after training.

    Training leaves gradient checkpointing on and KV cache off, which corrupts
    beam search generation. This must be called before evaluation.
    """
    try:
        loaded.model.gradient_checkpointing_disable()
    except Exception:
        pass
    loaded.model.config.use_cache = True
    loaded.model.eval()


def _prepare_lora_config(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Create a cfg copy with LoRA section patched by overrides.

    Allows per-experiment variation of rank, alpha, target_modules, etc.
    without mutating the shared config object.
    """
    new_cfg = copy.deepcopy(cfg)
    new_cfg["lora"].update(overrides)
    return new_cfg


def _get_tables_dir(cfg: dict) -> Path:
    """Resolve the tables output directory from config."""
    raw = cfg.get("paths", {}).get("tables", "results/tables")
    return Path(raw)


def _find_last_checkpoint(checkpoint_dir: Path) -> str | None:
    """Find the last saved checkpoint in a directory, if any.

    Returns the path as a string (for Trainer compatibility), or None
    if no checkpoint exists.
    """
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted(
        [p for p in checkpoint_dir.iterdir() if p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if ckpts:
        return str(ckpts[-1])
    return None


def run_experiment(
    cfg: dict,
    spec: ExperimentSpec,
    test_subset_size: int = 200,
) -> dict:
    """Run a single V3 experiment end-to-end.

    Steps: check if already done -> load dataset -> tokenize -> load model ->
    (optionally wrap LoRA) -> build trainer -> train (with resume support) ->
    restore inference mode -> evaluate on test -> write per-experiment CSV row.

    If the experiment's result CSV already exists, the experiment is skipped
    entirely and the saved results are returned. If a partial checkpoint exists
    but no result CSV, training resumes from the last checkpoint.

    Args:
        cfg: Global config dict (from load_config).
        spec: Experiment specification.
        test_subset_size: Number of test samples for final ROUGE evaluation.
            Kept at 200 to match V1 and V2 for direct comparability.

    Returns:
        Dict with ROUGE metrics, train/eval timing, and spec metadata.
        Also writes an individual CSV at results/tables/<spec.name>.csv.
    """
    print(f"\n{'='*70}\n[{spec.name}] Starting experiment\n{'='*70}")
    print(f"Spec: {asdict(spec)}")

    # --- SKIP CHECK ---
    # If the result CSV already exists, this experiment completed in a
    # previous run. Skip it entirely to save time on re-runs.
    tables_dir = _get_tables_dir(cfg)
    result_csv = tables_dir / f"{spec.name}.csv"
    if result_csv.exists():
        print(f"[{spec.name}] Already completed (CSV found at {result_csv}). Skipping.")
        return pd.read_csv(result_csv).iloc[0].to_dict()

    # --- CONFIG PATCHES ---
    # Deep-copy to avoid mutating the shared config across experiments.
    exp_cfg = copy.deepcopy(cfg)
    exp_cfg["dataset"]["train_subset"] = spec.train_subset
    exp_cfg["training"]["num_train_epochs"] = spec.num_epochs

    model_cfg = exp_cfg["models"][spec.model_key]
    model_type = model_cfg["type"]
    is_causal = model_type == "causal"

    # --- DATASET ---
    dataset = load_cnn_dailymail(exp_cfg)

    # --- MODEL ---
    loaded = load_model(model_cfg)

    # --- PREPROCESSING ---
    train_max_input = spec.train_max_input or model_cfg["max_input_length"]
    if is_causal:
        preprocess_fn = build_causal_preprocess_fn(
            tokenizer=loaded.tokenizer,
            max_input_length=train_max_input,
            max_target_length=exp_cfg["dataset"]["max_target_length"],
        )
    else:
        preprocess_fn = build_seq2seq_preprocess_fn(
            tokenizer=loaded.tokenizer,
            max_input_length=train_max_input,
            max_target_length=exp_cfg["dataset"]["max_target_length"],
            prefix="summarize: ",
        )
    tokenized = tokenize_dataset(dataset, preprocess_fn)

    # --- LORA (causal only) ---
    if is_causal:
        exp_cfg = _prepare_lora_config(exp_cfg, spec.lora_overrides)
        loaded = apply_lora(loaded, exp_cfg)

    # --- TRAINER ---
    if is_causal:
        trainer = build_causal_trainer(
            loaded=loaded,
            tokenized_dataset=tokenized,
            cfg=exp_cfg,
            output_subdir=spec.name,
            per_device_batch_size=spec.per_device_batch_size,
            gradient_accumulation_steps=spec.gradient_accumulation_steps,
            learning_rate=spec.learning_rate,
            **spec.extra_trainer_kwargs,
        )
    else:
        trainer = build_seq2seq_trainer(
            loaded=loaded,
            tokenized_dataset=tokenized,
            cfg=exp_cfg,
            output_subdir=spec.name,
            per_device_batch_size=spec.per_device_batch_size,
            gradient_accumulation_steps=spec.gradient_accumulation_steps,
            learning_rate=spec.learning_rate,
            **spec.extra_trainer_kwargs,
        )

    # --- TRAIN (with checkpoint resumption) ---
    checkpoint_dir = Path(exp_cfg["training"]["output_dir"]) / spec.name
    last_ckpt = _find_last_checkpoint(checkpoint_dir)
    if last_ckpt:
        print(f"[{spec.name}] Resuming training from {last_ckpt}")
    else:
        print(f"[{spec.name}] Starting training from scratch")

    train_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model()
    train_elapsed = time.time() - train_start
    print(f"[{spec.name}] Training finished in {train_elapsed/60:.1f} min")

    # --- RESTORE INFERENCE MODE ---
    # Training leaves gradient_checkpointing=True and use_cache=False,
    # which causes degenerate generation with beam search.
    _restore_inference_mode(loaded)
    loaded.tokenizer.padding_side = "left" if is_causal else "right"

    # --- EVALUATE ---
    test_subset = dataset["test"].select(range(test_subset_size))
    articles = test_subset["article"]
    references = test_subset["highlights"]

    eval_start = time.time()
    preds = generate_summaries(
        loaded,
        articles,
        max_new_tokens=exp_cfg["dataset"]["max_target_length"],
        num_beams=exp_cfg["generation"]["num_beams"],
        batch_size=2 if is_causal else 4,
    )
    eval_elapsed = time.time() - eval_start

    rouge_scores = compute_rouge(preds, references)

    # --- ASSEMBLE RESULT ---
    result = {
        "experiment": spec.name,
        "model_key": spec.model_key,
        **{k: float(v) for k, v in rouge_scores.items()},
        "train_minutes": round(train_elapsed / 60, 1),
        "eval_seconds": round(eval_elapsed, 1),
        "sec_per_sample": round(eval_elapsed / len(articles), 2),
        "train_subset": spec.train_subset,
        "num_epochs": spec.num_epochs,
        "learning_rate": spec.learning_rate,
        "effective_batch_size": spec.per_device_batch_size * spec.gradient_accumulation_steps,
        "lora_overrides": str(spec.lora_overrides) if spec.lora_overrides else "",
    }

    # --- PERSIST ---
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(result_csv, index=False)

    print(f"[{spec.name}] Results: {rouge_scores}")
    print(f"[{spec.name}] CSV saved to {result_csv}")

    # Show one qualitative example
    print(f"\n[{spec.name}] Qualitative example:")
    print(f"REFERENCE:\n{references[0]}")
    print(f"\nPREDICTION:\n{preds[0]}\n")

    # --- CLEANUP ---
    del trainer, loaded, tokenized, dataset
    _free_gpu_memory()

    return result


def run_experiment_suite(
    cfg: dict,
    specs: list[ExperimentSpec],
    combined_csv_name: str,
    test_subset_size: int = 200,
) -> pd.DataFrame:
    """Run a list of experiments sequentially and write a combined CSV.

    Experiments that crash are skipped with a logged error; the suite
    continues with the remaining specs. Already-completed experiments
    (from a previous interrupted run) are loaded from their saved CSVs
    instead of re-running.

    Args:
        cfg: Global config dict.
        specs: List of ExperimentSpec to run in order.
        combined_csv_name: Filename for the consolidated results CSV
            (saved under results/tables/).
        test_subset_size: Test samples for evaluation (default 200).

    Returns:
        DataFrame with one row per successful experiment, sorted by rougeL desc.
    """
    results: list[dict] = []

    for i, spec in enumerate(specs):
        print(f"\n Experiment {i+1}/{len(specs)}: {spec.name}")
        try:
            result = run_experiment(cfg, spec, test_subset_size=test_subset_size)
            results.append(result)
        except Exception as e:
            print(f"\n[{spec.name}] FAILED: {type(e).__name__}: {e}")
            print(f"[{spec.name}] Continuing with next experiment...")
            _free_gpu_memory()

    if not results:
        print("No experiments completed successfully.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("rougeL", ascending=False).reset_index(drop=True)

    tables_dir = _get_tables_dir(cfg)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / combined_csv_name, index=False)
    print(f"\n Suite finished. {len(results)}/{len(specs)} experiments completed.")
    print(f"Combined CSV: {tables_dir / combined_csv_name}")

    return df