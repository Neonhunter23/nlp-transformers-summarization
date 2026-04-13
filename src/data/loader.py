"""CNN/DailyMail dataset preparation."""
from pathlib import Path
import yaml
from datasets import load_dataset, DatasetDict


def load_config(config_path: str = "config/config.yaml") -> dict:
    """loads config.yaml"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cnn_dailymail(config: dict) -> DatasetDict:
    """
    Loads CNN/DailyMail from HuggingFace, with subset support.

    Args:
        config: dict from config.yaml

    Returns:
        DatasetDict with splits 'train', 'validation', 'test'
    """
    ds_cfg = config["dataset"]
    cache_dir = Path(ds_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        ds_cfg["name"],
        ds_cfg["version"],
        cache_dir=str(cache_dir),
    )

    # Optional subsetting for faster iterations
    if ds_cfg.get("train_subset"):
        dataset["train"] = dataset["train"].select(range(ds_cfg["train_subset"]))
    if ds_cfg.get("val_subset"):
        dataset["validation"] = dataset["validation"].select(range(ds_cfg["val_subset"]))
    if ds_cfg.get("test_subset"):
        dataset["test"] = dataset["test"].select(range(ds_cfg["test_subset"]))

    return dataset


def get_dataset_stats(dataset: DatasetDict) -> dict:
    """Returns basic stats about the dataset splits"""
    stats = {}
    for split_name, split in dataset.items():
        article_lens = [len(x.split()) for x in split["article"]]
        summary_lens = [len(x.split()) for x in split["highlights"]]
        stats[split_name] = {
            "n_samples": len(split),
            "article_words_mean": sum(article_lens) / len(article_lens),
            "article_words_max": max(article_lens),
            "summary_words_mean": sum(summary_lens) / len(summary_lens),
            "summary_words_max": max(summary_lens),
            "compression_ratio": (
                sum(summary_lens) / sum(article_lens)
            ),
        }
    return stats


if __name__ == "__main__":
    cfg = load_config()
    ds = load_cnn_dailymail(cfg)
    print("Splits cargados:", list(ds.keys()))
    print("\nEstadísticas:")
    for split, s in get_dataset_stats(ds).items():
        print(f"\n[{split}]")
        for k, v in s.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")