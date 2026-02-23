import argparse
import json
import logging
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config_utils import (
    load_section_config as _load_section_config,
)
from baselines_classifier_dataloader import BaselinesActsDataset, CollateFn
from classifier.classifier import Classifier


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError("Checkpoint missing 'model' state dict")
    return ckpt


def build_model(
    ckpt,
    task: Optional[str],
    inputs: Optional[str],
    input_dim: Optional[int],
    hidden_dim: Optional[int],
    dropout: Optional[float],
    device: str,
):
    config = ckpt.get("config", {})

    resolved_task = task or config.get("task")
    if resolved_task not in {"correctness", "logicality"}:
        raise ValueError("Task must be 'correctness' or 'logicality'")

    resolved_inputs = inputs or config.get("inputs", "sae_acts")
    if resolved_inputs not in {"layer_acts", "sae_acts"}:
        raise ValueError("inputs must be 'layer_acts' or 'sae_acts'")

    resolved_input_dim = input_dim or config.get("n_inputs") or config.get("input_dim")
    if resolved_input_dim is None:
        raise ValueError("Input dimension not found; set input_dim in token_sae_eval config")

    resolved_hidden_dim = hidden_dim or config.get("hidden_dim", 1024)
    resolved_dropout = dropout if dropout is not None else config.get("dropout", 0.1)

    model = Classifier(
        input_dim=resolved_input_dim,
        hidden_dim=resolved_hidden_dim,
        dropout=resolved_dropout,
        task=resolved_task,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, resolved_task, resolved_inputs


@torch.no_grad()
def evaluate(model, dataloader, task, device, threshold):
    total, correct = 0, 0
    for batch in dataloader:
        feats = batch["acts"].to(device)
        labels = batch[task].to(device).squeeze(-1)

        logits = model(feats).squeeze(-1)
        preds = (torch.sigmoid(logits) >= threshold).to(labels.dtype)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    accuracy = correct / total if total else 0.0
    return accuracy, total


def main():
    parser = argparse.ArgumentParser(description="Eval token-SAE baseline classifier")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "token_sae_eval", args.set)

    ckpt_path = cfg.get("ckpt")
    data_file = cfg.get("data_file")
    if ckpt_path is None or str(ckpt_path).strip() == "":
        raise ValueError("token_sae_eval.ckpt is required")
    if data_file is None or str(data_file).strip() == "":
        raise ValueError("token_sae_eval.data_file is required")

    ckpt_path = str(ckpt_path).strip()
    data_file = str(data_file).strip()

    batch_size = int(cfg.get("batch_size", 128))
    task = cfg.get("task")
    inputs = cfg.get("inputs")
    input_dim = cfg.get("input_dim")
    hidden_dim = cfg.get("hidden_dim")
    dropout = cfg.get("dropout")
    threshold = float(cfg.get("threshold", 0.5))
    num_workers = int(cfg.get("num_workers", 4))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ckpt = load_checkpoint(ckpt_path)
    model, task, resolved_inputs = build_model(
        ckpt,
        task=task,
        inputs=inputs,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device,
    )

    dataset = BaselinesActsDataset(data_file, resolved_inputs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CollateFn(),
        num_workers=num_workers,
        pin_memory=True,
    )

    logging.info(
        "Running eval | task=%s | inputs=%s | threshold=%.2f | device=%s",
        task,
        resolved_inputs,
        threshold,
        device,
    )
    acc, total = evaluate(model, dataloader, task, device, threshold)
    logging.info("Eval finished | samples=%d | accuracy=%.4f", total, acc)


if __name__ == "__main__":
    main()
