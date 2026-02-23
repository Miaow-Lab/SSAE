import argparse
import json
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from config_utils import (
    load_section_config as _load_section_config,
)
from classifier.classifier import Classifier
from baselines_classifier_dataloader import BaselinesActsDataset, CollateFn


def get_lr(it, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def compute_loss(model, batch):
    acts = batch["acts"].to(device)
    step_lengths = batch["step_lengths"].to(device)
    begin_token_ids = batch["begin_token_ids"].to(device)
    correctness = batch["correctness"].to(device)
    logicality = batch["logicality"].to(device)

    pred = model(acts)
    if task == "token":
        loss = F.cross_entropy(pred, begin_token_ids.long().squeeze(1))
    elif task == "len":
        loss = torch.nn.MSELoss()(pred, step_lengths)
    elif task == "correctness":
        loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), correctness.squeeze(-1))
    elif task == "logicality":
        loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), logicality.squeeze(-1))
    else:
        raise ValueError(f"Unknown task: {task}")
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train token-SAE baseline classifier")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "token_sae_train", args.set)

    required_keys = ["train_file", "val_file", "n_inputs"]
    for key in required_keys:
        if cfg.get(key) is None:
            raise ValueError(f"token_sae_train.{key} is required")

    out_dir = str(cfg.get("out_dir", "out_classifier/token_sae"))
    init_name = str(cfg.get("init_name", "Qwen/Qwen2.5-0.5B"))
    train_file = str(cfg["train_file"])
    val_file = str(cfg["val_file"])

    global task
    task = str(cfg.get("task", "correctness"))
    inputs = str(cfg.get("inputs", "sae_acts"))
    if inputs not in {"layer_acts", "sae_acts"}:
        raise ValueError("token_sae_train.inputs must be 'layer_acts' or 'sae_acts'")

    train_dataset = BaselinesActsDataset(train_file, inputs)
    val_dataset = BaselinesActsDataset(val_file, inputs)

    log_interval = int(cfg.get("log_interval", 1))
    eval_interval = int(cfg.get("eval_interval", 1))
    always_save_checkpoint = bool(cfg.get("always_save_checkpoint", False))
    eval_only = bool(cfg.get("eval_only", False))

    n_inputs = int(cfg["n_inputs"])
    hidden_dim = int(cfg.get("hidden_dim", 1024))
    dropout = float(cfg.get("dropout", 0.1))
    learning_rate = float(cfg.get("learning_rate", 1e-6))
    min_lr = float(cfg.get("min_lr", 1e-7))
    max_iters = int(cfg.get("max_iters", 300))
    batch_size = int(cfg.get("batch_size", 128))

    global warmup_iters, lr_decay_iters
    warmup_iters = int(cfg.get("warmup_iters", 2))
    lr_decay_iters = int(cfg.get("lr_decay_iters", max_iters))
    decay_lr = bool(cfg.get("decay_lr", True))

    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.95))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    backend = str(cfg.get("ddp_backend", "nccl"))
    gradient_accumulation_steps = int(cfg.get("grad_accum_steps", 8))

    global device
    device = str(cfg.get("device", "cuda"))
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    compile = bool(cfg.get("torch_compile", False))

    config = {
        "out_dir": out_dir,
        "init_name": init_name,
        "inputs": inputs,
        "task": task,
        "train_file": train_file,
        "val_file": val_file,
        "n_inputs": n_inputs,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "min_lr": min_lr,
        "max_iters": max_iters,
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "decay_lr": decay_lr,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "backend": backend,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "device": device,
        "dtype": dtype,
        "compile": compile,
        "always_save_checkpoint": always_save_checkpoint,
        "eval_only": eval_only,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "wandb_log": bool(cfg.get("wandb_log", True)),
        "config_path": args.config,
        "cli_overrides": args.set,
    }

    checkpoint_name_pre = (
        f"ckpt_{time.strftime('%Y%m%d_%H%M%S')}_ddp__token_sae_{inputs}_{task}_hidden_dim_{hidden_dim}_dropout_{dropout}"
    )
    checkpoint_name = checkpoint_name_pre + ".pt"
    checkpoint_name_final = checkpoint_name_pre + "_final.pt"

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(out_dir, f"{checkpoint_name_pre}.log")),
                logging.StreamHandler(),
            ],
        )
        run_config_path = os.path.join(out_dir, f"{checkpoint_name_pre}_run_config.json")
        with open(run_config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)

    torch.manual_seed(int(cfg.get("seed", 1337)) + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    num_token = None
    if task == "token":
        tokenizer = AutoTokenizer.from_pretrained(init_name)
        num_token = tokenizer.vocab_size

    model = Classifier(
        input_dim=n_inputs,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task=task,
        num_token=num_token,
    ).to(device)

    scaler = GradScaler("cuda", enabled=(dtype == "float16"))
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    if compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=CollateFn(),
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=CollateFn(),
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=CollateFn(),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    wandb_log = bool(cfg.get("wandb_log", True))
    if wandb_log and master_process:
        import wandb

        wandb.init(
            project=f"token-sae-baseline-{inputs}-predict-{task}",
            name=f"token-sae-{inputs}-{task}-batch{batch_size}-lr{learning_rate}",
            config=config,
        )

    best_val_loss = 1e9
    t0 = time.time()
    global_step = 0

    for iter_num in range(max_iters):
        if ddp:
            train_sampler.set_epoch(iter_num)

        lr = get_lr(iter_num, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        checkpoint_payload = None
        if iter_num % eval_interval == 0 and master_process:
            model.eval()
            with torch.no_grad():
                with ctx:
                    val_total_loss, batch_num = 0.0, 0
                    for batch in val_loader:
                        val_total_loss += compute_loss(model, batch).item()
                        batch_num += 1
            val_loss = val_total_loss / batch_num
            logging.info(f"step {iter_num}: val loss {val_loss:.4f}")
            if wandb_log:
                wandb.log({"iter": iter_num, "val/loss": val_loss})
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                checkpoint_payload = {
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "model": model.module.state_dict() if ddp else model.state_dict(),
                }
                torch.save(checkpoint_payload, os.path.join(out_dir, checkpoint_name))
            if checkpoint_payload is not None:
                torch.save(checkpoint_payload, os.path.join(out_dir, checkpoint_name_final))

        if eval_only:
            break

        model.train()
        micro_step = 0
        train_total_loss, batch_num = 0.0, 0

        for batch in train_loader:
            if micro_step == 0:
                optimizer.zero_grad(set_to_none=True)
            if ddp:
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

            with ctx:
                train_loss_batch = compute_loss(model, batch)
            scaler.scale(train_loss_batch).backward()

            train_total_loss += train_loss_batch.item()
            batch_num += 1
            micro_step += 1
            global_step += 1

            if master_process and wandb_log:
                wandb.log({"batch/loss": train_loss_batch.item()}, step=global_step)

            if micro_step == gradient_accumulation_steps:
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                micro_step = 0

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0 and master_process:
            train_loss = train_total_loss / batch_num
            logging.info(f"iter {iter_num}: train loss {train_loss:.4f}, time {dt * 1000:.2f}ms")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
