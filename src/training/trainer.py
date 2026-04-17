"""Unified training utilities for seq2seq and causal summarization models."""
from pathlib import Path
import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.models.loader import LoadedModel
from src.evaluation.metrics import compute_rouge


def build_rouge_compute_metrics(tokenizer):
    """Build a compute_metrics function for Seq2SeqTrainer returning ROUGE scores.

    Handles the -100 label padding convention used to mask loss computation,
    and sanitizes any negative token ids in predictions that may appear as
    padding from variable-length generation.
    """
    def _compute(eval_pred):
        predictions, labels = eval_pred
        # Replace -100 and any negative ids with pad_token_id before decoding.
        # Seq2SeqTrainer pads generated predictions with -100 when sequences
        # in a batch have different lengths, which crashes the fast tokenizer.
        predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_rouge(decoded_preds, decoded_labels)
    return _compute


def build_seq2seq_trainer(
    loaded: LoadedModel,
    tokenized_dataset,
    cfg: dict,
    output_subdir: str,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    enable_gradient_checkpointing: bool = True,
    force_fp32: bool = False,
    optim: str = "adamw_torch",
) -> Seq2SeqTrainer:
    """Build a Seq2SeqTrainer for full fine-tuning of encoder-decoder models.

    Args:
        loaded: A LoadedModel with a seq2seq HuggingFace model.
        tokenized_dataset: DatasetDict with 'train' and 'validation' splits
            already tokenized via build_seq2seq_preprocess_fn.
        cfg: Global config dict.
        output_subdir: Subfolder under training.output_dir to save checkpoints.
        per_device_batch_size: Override for cfg['training']['per_device_train_batch_size'].
        gradient_accumulation_steps: Override for config value.
        learning_rate: Override for config value.
        enable_gradient_checkpointing: Trade compute for memory. Recommended
            when training seq2seq models on consumer GPUs.

    Returns:
        Initialized Seq2SeqTrainer ready to call .train().
    """
    train_cfg = cfg["training"]
    output_dir = Path(train_cfg["output_dir"]) / output_subdir

    # Gradient checkpointing is incompatible with KV cache; disable cache explicitly.
    if enable_gradient_checkpointing:
        loaded.model.gradient_checkpointing_enable()
        loaded.model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=loaded.tokenizer,
        model=loaded.model,
        padding=True,
    )

    # T5 is numerically unstable in bf16 training; allow forcing fp32.
    use_bf16 = train_cfg["bf16"] and not force_fp32
    use_fp16 = train_cfg["fp16"] and not force_fp32

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=per_device_batch_size or train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps or train_cfg["gradient_accumulation_steps"],
        learning_rate=learning_rate or train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim,
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=cfg["dataset"]["max_target_length"],
        generation_num_beams=cfg["generation"]["num_beams"],
        report_to="none",
        seed=cfg["seed"],
    )

    return Seq2SeqTrainer(
        model=loaded.model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=loaded.tokenizer,
        compute_metrics=build_rouge_compute_metrics(loaded.tokenizer),
    )


def apply_lora(loaded: LoadedModel, cfg: dict) -> LoadedModel:
    """Wrap a causal model with LoRA adapters for parameter-efficient fine-tuning.

    Only the LoRA matrices are trained; the base model stays frozen.

    Args:
        loaded: A LoadedModel with a causal HuggingFace model.
        cfg: Global config dict with a 'lora' section.

    Returns:
        The same LoadedModel with model wrapped in a PeftModel.
    """
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    loaded.model = get_peft_model(loaded.model, peft_config)
    loaded.model.print_trainable_parameters()
    return loaded


def build_causal_trainer(
    loaded: LoadedModel,
    tokenized_dataset,
    cfg: dict,
    output_subdir: str,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
) -> Trainer:
    """Build a standard Trainer for LoRA fine-tuning of causal models.

    For causal models we skip generation during eval (too slow with beam search).
    Eval loss is used for best-model selection; full ROUGE evaluation is done
    post-training via src.evaluation.inference.

    Args:
        loaded: LoadedModel already wrapped with LoRA via apply_lora().
        tokenized_dataset: DatasetDict tokenized via build_causal_preprocess_fn.
        cfg: Global config dict.
        output_subdir: Subfolder under training.output_dir to save checkpoints.
        per_device_batch_size: Override for config value.
        gradient_accumulation_steps: Override for config value.
        learning_rate: Override for config value. LoRA typically uses 1e-4 to 3e-4.

    Returns:
        Initialized Trainer ready to call .train().
    """
    train_cfg = cfg["training"]
    output_dir = Path(train_cfg["output_dir"]) / output_subdir

    # Enable gradient checkpointing to roughly halve activation memory.
    # For PEFT/LoRA models, input grads must be enabled manually because
    # the base model is frozen.
    loaded.model.gradient_checkpointing_enable()
    loaded.model.enable_input_require_grads()
    loaded.model.config.use_cache = False

    # Seq2Seq collator correctly handles dynamic padding for labels with -100.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=loaded.tokenizer,
        model=loaded.model,
        padding=True,
        label_pad_token_id=-100,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=per_device_batch_size or train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps or train_cfg["gradient_accumulation_steps"],
        learning_rate=learning_rate or train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        logging_steps=train_cfg["logging_steps"],
        # LoRA eval with beam search is prohibitively expensive; skip per-epoch
        # eval and run a single ROUGE evaluation after training via
        # src.evaluation.inference.generate_summaries.
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=cfg["seed"],
    )

    return Trainer(
        model=loaded.model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=loaded.tokenizer,
    )

def restore_inference_mode(loaded: LoadedModel) -> None:
    """Restore a model to clean inference mode after training.

    Training enables gradient checkpointing and disables KV cache, both of
    which corrupt generation if left active. This function should be called
    before running any post-training evaluation with model.generate().
    """
    loaded.model.gradient_checkpointing_disable()
    loaded.model.config.use_cache = True
    loaded.model.eval()