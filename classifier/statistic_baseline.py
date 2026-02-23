import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from config_utils import (
    load_section_config as _load_section_config,
)


def calculate_entropy_baseline(dataset):
    if isinstance(dataset, np.ndarray) and dataset.ndim == 1:
        first_tokens = dataset
    else:
        if len(dataset) > 0 and isinstance(dataset[0], (int, float, np.integer)):
            first_tokens = dataset
        else:
            first_tokens = [
                item[0].item() if isinstance(item, torch.Tensor) else item[0]
                for item in dataset
            ]

    total_samples = len(first_tokens)
    counts = Counter(first_tokens)
    probs = np.array([count / total_samples for count in counts.values()])
    entropy = -np.sum(probs * np.log(probs))
    perplexity = np.exp(entropy)
    return entropy, perplexity


def main():
    parser = argparse.ArgumentParser(description="Compute statistic baselines for classifier labels")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "statistic_baseline", args.set)

    train_npz = cfg.get("train_npz")
    val_npz = cfg.get("val_npz")
    show_logicality = bool(cfg.get("show_logicality", False))

    if not train_npz:
        raise ValueError("statistic_baseline section requires train_npz")

    train_data = np.load(train_npz)

    first_tokens = train_data["begin_token_id"]
    entropy, ppl = calculate_entropy_baseline(first_tokens)
    print(f"Total train samples: {len(first_tokens)}")
    print(f"First token baseline entropy (loss): {entropy:.4f}")
    print(f"First token baseline perplexity: {ppl:.4f}")

    next_step_length = train_data["step_length"]
    mean_step_length = np.mean(next_step_length)
    std_step_length = np.std(next_step_length)
    print(f"Step length baseline mean: {mean_step_length:.4f}")
    print(f"Step length baseline std: {std_step_length:.4f}")

    if val_npz is not None:
        val_data = np.load(val_npz)
        val_correctness = val_data["correctness"]
        correctness_ratio = np.mean(val_correctness)
        print(f"Validation correctness positive ratio baseline: {correctness_ratio:.4f}")

        if show_logicality and "logicality" in val_data:
            val_logicality = val_data["logicality"]
            logicality_ratio = np.mean(val_logicality)
            print(f"Validation logicality positive ratio baseline: {logicality_ratio:.4f}")


if __name__ == "__main__":
    main()
