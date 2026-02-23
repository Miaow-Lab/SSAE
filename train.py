import argparse
import importlib
import json
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from config_utils import load_yaml_or_json_config as _load_yaml_or_json_config
from model_qwen import MyModel


def _phase_defaults(phase):
    if phase == 1:
        return {
            "MAX_ITERS": 30,
            "EVAL_INTERVAL": 1,
            "LEARNING_RATE": 1e-6,
            "MIN_LR": 1e-7,
        }
    return {
        "MAX_ITERS": 101,
        "EVAL_INTERVAL": 1,
        "LEARNING_RATE": 1e-4,
        "MIN_LR": 1e-5,
    }


def _parse_scalar(raw):
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        if text.startswith("0") and text not in {"0", "0.0"} and not text.startswith("0."):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_args():
    parser = argparse.ArgumentParser(description="SSAE training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values from CLI. Example: --set BATCH_SIZE=8 --set LEARNING_RATE=5e-7",
    )
    return parser.parse_args()


def _build_config(args):
    config = {
        "LOG_INTERVAL": 1,
        "ALWAYS_SAVE_CHECKPOINT": True,
        "EVAL_ONLY": False,
        "CONTINUE_TRAIN": False,
        "TASK": "gsm8k-385k",
        "N_INPUTS": 896,
        "SPARSITY_FACTOR": 1,
        "L1_WEIGHT": 1e-4,
        "L1_TARGET": 3.0,
        "PHASE": 1,
        "ENCODER_INIT": "./Qwen2.5-0.5B",
        "DECODER_INIT": "./Qwen2.5-0.5B",
        "BATCH_SIZE": 16,
        "WARMUP_ITERS": 2,
        "LR_DECAY_ITERS": None,
        "DECAY_LR": True,
        "BETA1": 0.9,
        "BETA2": 0.95,
        "WEIGHT_DECAY": 0.01,
        "GRAD_CLIP": 1.0,
        "DDP_BACKEND": "nccl",
        "GRAD_ACCUM_STEPS": 8,
        "DEVICE": "cuda",
        "TORCH_COMPILE": False,
        "TRAIN_DATA_PATH": "data/gsm8k_385K_train.json",
        "VAL_DATA_PATH": "data/gsm8k_385K_valid.json",
        "USE_NUMINA_DATALOADER": False,
        "WANDB_LOG": True,
        "RESUME_CKPT_NAME": None,
    }

    if args.config:
        file_cfg = _load_yaml_or_json_config(args.config)
        config.update(file_cfg)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set key in item: {item}")
        config[key] = _parse_scalar(value)

    phase = int(config["PHASE"])
    phase_defaults = _phase_defaults(phase)
    for key, value in phase_defaults.items():
        if key not in config or config[key] is None:
            config[key] = value

    if config.get("LR_DECAY_ITERS") is None:
        config["LR_DECAY_ITERS"] = int(config["MAX_ITERS"])

    return config


args = _parse_args()
runtime_config = _build_config(args)

log_interval = int(runtime_config["LOG_INTERVAL"])
always_save_checkpoint = bool(runtime_config["ALWAYS_SAVE_CHECKPOINT"])
eval_only = bool(runtime_config["EVAL_ONLY"])
continue_train = bool(runtime_config["CONTINUE_TRAIN"])
task = str(runtime_config["TASK"])

n_inputs = int(runtime_config["N_INPUTS"])
sparsity_factor = int(runtime_config["SPARSITY_FACTOR"])
n_latents = n_inputs * sparsity_factor
L1_WEIGHT = float(runtime_config["L1_WEIGHT"])
L1_TARGET = float(runtime_config["L1_TARGET"])
phase = int(runtime_config["PHASE"])

if phase == 1 and not continue_train:
    init_from = (
        str(runtime_config["ENCODER_INIT"]),
        str(runtime_config["DECODER_INIT"]),
    )
else:
    init_from = ("resume", "resume")

max_iters = int(runtime_config["MAX_ITERS"])
eval_interval = int(runtime_config["EVAL_INTERVAL"])
learning_rate = float(runtime_config["LEARNING_RATE"])
min_lr = float(runtime_config["MIN_LR"])

batch_size = int(runtime_config["BATCH_SIZE"])
warmup_iters = int(runtime_config["WARMUP_ITERS"])
lr_decay_iters = int(runtime_config["LR_DECAY_ITERS"])
decay_lr = bool(runtime_config["DECAY_LR"])

beta1 = float(runtime_config["BETA1"])
beta2 = float(runtime_config["BETA2"])
weight_decay = float(runtime_config["WEIGHT_DECAY"])
grad_clip = float(runtime_config["GRAD_CLIP"])

backend = str(runtime_config["DDP_BACKEND"])
gradient_accumulation_steps = int(runtime_config["GRAD_ACCUM_STEPS"])

device = str(runtime_config["DEVICE"])
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = bool(runtime_config["TORCH_COMPILE"])

train_data_path = str(runtime_config["TRAIN_DATA_PATH"])
val_data_path = str(runtime_config["VAL_DATA_PATH"])
wandb_log = bool(runtime_config["WANDB_LOG"])
resume_ckpt_name = runtime_config.get("RESUME_CKPT_NAME")
if resume_ckpt_name is not None:
    resume_ckpt_name = str(resume_ckpt_name).strip()
    if resume_ckpt_name == "":
        resume_ckpt_name = None

if 'numina' in task.lower():
    dataloader_module = importlib.import_module("dataloader_numina")
elif 'opencodeinstruct' in task.lower():
    dataloader_module = importlib.import_module("dataloader_opencodeinstruct")
else:
    dataloader_module = importlib.import_module("dataloader")
ProblemAnswerDataset = dataloader_module.ProblemAnswerDataset
CollateFn = dataloader_module.CollateFn

config = {
    "log_interval": log_interval,
    "always_save_checkpoint": always_save_checkpoint,
    "eval_only": eval_only,
    "continue_train": continue_train,
    "task": task,
    "n_inputs": n_inputs,
    "sparsity_factor": sparsity_factor,
    "n_latents": n_latents,
    "L1_WEIGHT": L1_WEIGHT,
    "L1_TARGET": L1_TARGET,
    "phase": phase,
    "init_from": init_from,
    "max_iters": max_iters,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "min_lr": min_lr,
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
    "train_data_path": train_data_path,
    "val_data_path": val_data_path,
    "wandb_log": wandb_log,
    "resume_ckpt_name": resume_ckpt_name,
    "config_path": args.config,
    "cli_overrides": args.set,
}

checkpoint_name_pre = (
    f"ckpt_{time.strftime('%Y%m%d_%H%M%S')}_ddp_"
    + task
    + "_qwen2.5-0.5b"
    + "_dwa_L1_target_"
    + str(L1_TARGET)
    + "_phase_"
    + str(phase)
    + "_mask_token_0.1"
)

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
    print(
        f"Using DDP with {ddp_world_size} processes, master process: {master_process}, current device {device}"
    )
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    print(f"Using single GPU, current device {device}")

out_dir = f"out_{task}/dwa_{L1_TARGET}"

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
    logging.info(f"Saved run config to {run_config_path}")
    logging.info(f"n_latents: {n_latents}")
    logging.info(f"L1_weight: {L1_WEIGHT}")
    logging.info(f"L1_target: {L1_TARGET}")
    logging.info(f"phase: {phase}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"max iters: {max_iters}")
    logging.info(f"learning rate: {learning_rate}")
    logging.info(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    logging.info(f"continue_train: {continue_train}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

if init_from[0] == "resume":
    if not resume_ckpt_name:
        raise ValueError(
            "RESUME_CKPT_NAME is required when entering resume branch "
            "(phase != 1 or CONTINUE_TRAIN = true)."
        )
    ckpt_path = os.path.join(out_dir, resume_ckpt_name)
    logging.info(f"Resuming training from {ckpt_path}, phase = {phase}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    logging.info(checkpoint["best_val_loss"])
    if continue_train:
        logging.info(
            f"Continuing training from {ckpt_path}, phase = {phase}, start from iter {checkpoint['iter_num']}"
        )
    encoder_name, decoder_name = checkpoint["encoder_name"], checkpoint["decoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = MyModel(
        tokenizer, sparsity_factor, init_from=(encoder_name, decoder_name), phase=phase
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
else:
    logging.info(
        f"Initializing from Weights: encoder-{init_from[0]}, decoder-{init_from[1]}, phase = {phase}"
    )
    tokenizer = AutoTokenizer.from_pretrained(init_from[0])
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")
    model = MyModel(tokenizer, sparsity_factor, init_from=init_from, phase=phase).to(
        device
    )

scaler = GradScaler("cuda", enabled=(dtype == "float16"))
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(
    trainable_params, lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay
)

if continue_train:
    optimizer.load_state_dict(checkpoint["optimizer"])
    best_val_loss = checkpoint["best_val_loss"]
    start_iter = checkpoint["iter_num"] + 1
else:
    best_val_loss = 1e9
    start_iter = 0

checkpoint = None

if compile:
    logging.info("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

train_dataset = ProblemAnswerDataset(train_data_path, tokenizer)
val_dataset = ProblemAnswerDataset(val_data_path, tokenizer)

if ddp:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=CollateFn(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
        ),
        num_workers=4,
        pin_memory=True,
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=CollateFn(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
        ),
        num_workers=4,
        pin_memory=True,
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=CollateFn(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
    ),
    num_workers=4,
    pin_memory=True,
)


class DWAController:
    def __init__(
        self, target=5.0, update_freq=50, init_weight=1e-3, min_w=1e-6, max_w=0.1
    ):
        self.target = target
        self.weight = init_weight
        self.update_freq = update_freq
        self.loss_accumulator = 0.0
        self.steps = 0
        self.min_w = min_w
        self.max_w = max_w

    def step(self, batch_sparse_loss):
        self.loss_accumulator += batch_sparse_loss
        self.steps += 1
        if self.steps < self.update_freq:
            return self.weight
        avg_loss = self.loss_accumulator / self.steps
        if avg_loss > self.target:
            self.weight *= 1.01
        else:
            self.weight *= 0.99
        self.weight = max(self.min_w, min(self.max_w, self.weight))
        self.loss_accumulator = 0.0
        self.steps = 0
        return self.weight

    def get_weight(self):
        return self.weight


dwa_controller = DWAController(
    target=L1_TARGET, update_freq=100, init_weight=L1_WEIGHT, min_w=1e-6, max_w=0.1
)


def get_lr(it, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def compute_loss(model, batch, L1_weight=L1_WEIGHT, sample=False, mask=False):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    sep_pos = batch["sep_pos"]
    val_len = batch["val_len"]
    sep_pos = torch.as_tensor(sep_pos, device=device)
    val_len = torch.as_tensor(val_len, device=device)
    hints_sep_ids = batch["hints_sep_ids"].to(device)
    hints_sep_attention_masks = batch["hints_sep_attention_masks"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    if phase == 1:
        if mask:
            bsz, seq_len = attention_mask.shape
            mask_ratio = 0.1
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            interval_mask = (pos >= sep_pos.unsqueeze(1)) & (pos < val_len.unsqueeze(1))
            rand_mask = torch.rand(bsz, seq_len, device=device) >= mask_ratio
            attention_mask = attention_mask & (~interval_mask | rand_mask)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        _, loss_sparsity, logits = model(
            input_ids, attention_mask, hints_sep_ids, hints_sep_attention_masks
        )

        logits = logits[:, sparsity_factor - 1 :, :]
        loss_predict = loss_fn(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1))
        loss_predict = loss_predict.view(input_ids.size()) * loss_mask

        loss_predict_sum = loss_predict.sum()
        no_mask_counts = loss_mask.sum()
        loss_nll_all = loss_predict_sum / no_mask_counts

        loss_sparsity_sum = loss_sparsity
        current_batch_size = input_ids.size(0)
        loss_sparsity_all = loss_sparsity_sum / current_batch_size
        loss_all = loss_nll_all + L1_weight * loss_sparsity_all
        return loss_nll_all, loss_sparsity_all, loss_all, no_mask_counts

    if phase == 2 or phase == 3:
        mse_nll_loss, mean_error = model(
            input_ids, attention_mask, hints_sep_ids, hints_sep_attention_masks
        )
        no_mask_counts = torch.tensor(1)
        return mse_nll_loss, mean_error, mse_nll_loss, no_mask_counts


if master_process:
    logging.info("Finish loading data, begin training...")

wandb_project = f"SSAE_phase{phase}_{task}_{init_from[0].split('/')[-1]}"
wandb_run_name = f"Qwen_Tr_{L1_TARGET}_sparsity_{sparsity_factor}_batchsize_{batch_size}"
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


t0 = time.time()
global_step = 0

num_batches_per_iter = len(train_loader) // gradient_accumulation_steps
print(f"num_batches_per_iter: {num_batches_per_iter}")
eval_batch_interval = num_batches_per_iter / 2 * eval_interval

for iter_num in range(start_iter, max_iters):
    if ddp:
        train_sampler.set_epoch(iter_num)

    lr = get_lr(iter_num, learning_rate, min_lr) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if eval_only:
        break

    model.train()
    micro_step = 0
    total_nll, total_spa, total_loss, total_tokens = 0, 0, 0, 0

    for batch in tqdm(train_loader, desc=f"iter{iter_num}", dynamic_ncols=True):
        l1_weight = dwa_controller.get_weight()

        if micro_step == 0:
            optimizer.zero_grad(set_to_none=True)

        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )

        with ctx:
            _nll, _spa, _loss, _tokens = compute_loss(
                model, batch, l1_weight, sample=False, mask=True
            )

        scaler.scale(_loss).backward()

        total_nll += _nll.item()
        total_spa += _spa.item()
        total_loss += _loss.item()
        total_tokens += _tokens.item()

        micro_step += 1
        global_step += 1
        dwa_controller.step(_spa.item())

        if master_process:
            print(
                f"Batch Loss: {_loss.item():.4f} | "
                f"Nll Loss: {_nll.item():.4f} | "
                f"Spa Loss {_spa.item():.4f} | "
                f"L1 Weight: {l1_weight}"
            )
            if wandb_log:
                wandb.log(
                    {
                        "batch/loss": _loss.item(),
                        "batch/loss_nll": _nll.item(),
                        "batch/loss_spa": _spa.item(),
                    },
                    step=global_step,
                )

        if micro_step == gradient_accumulation_steps:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            micro_step = 0

        if global_step % eval_batch_interval == 0 and master_process:
            model.eval()
            l1_weight = dwa_controller.get_weight()
            logging.info(f"global_step {global_step}: L1 weight {l1_weight}")

            total_v_nll, total_v_spa, total_v_loss, total_v_tokens = 0, 0, 0, 0

            with torch.no_grad():
                with ctx:
                    for idx, v_batch in enumerate(val_loader):
                        if idx == 0:
                            v_nll, v_spa, v_loss, v_tokens = compute_loss(
                                model, v_batch, l1_weight, sample=True, mask=False
                            )
                        else:
                            v_nll, v_spa, v_loss, v_tokens = compute_loss(
                                model, v_batch, l1_weight, sample=False, mask=False
                            )

                        total_v_nll += v_nll.item()
                        total_v_spa += v_spa.item()
                        total_v_loss += v_loss.item()
                        total_v_tokens += v_tokens.item()

            val_loss = total_v_loss / len(val_loader)
            val_loss_nll = total_v_nll / len(val_loader)
            val_loss_spa = total_v_spa / len(val_loader)

            if phase == 1:
                logging.info(
                    f"global_step {global_step}: "
                    f"val loss {val_loss:.4f}, "
                    f"val loss nll {val_loss_nll:.4f}, "
                    f"val loss spa {val_loss_spa:.4f}"
                )
                if wandb_log:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/loss_nll": val_loss_nll,
                            "val/loss_spa": val_loss_spa,
                        },
                        step=global_step,
                    )
            elif phase == 2:
                logging.info(
                    f"global_step {global_step}: "
                    f"val loss mse {val_loss:.4e}, "
                    f"val mean error {val_loss_spa:.4e}"
                )
            elif phase == 3:
                logging.info(
                    f"global_step {global_step}: "
                    f"val loss nll {val_loss:.4e}, "
                    f"val mean error {val_loss_spa:.4e}"
                )

            if val_loss_nll < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss_nll
                checkpoint = {
                    "encoder_name": init_from[0],
                    "decoder_name": init_from[1],
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                checkpoint["model"] = (
                    model.module.state_dict() if ddp else model.state_dict()
                )

                logging.info(f"saving checkpoint at global_step {global_step}")
                checkpoint_name = checkpoint_name_pre + f"_global_step_{global_step}.pt"
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))

            model.train()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        if phase == 1:
            logging.info(
                f"iter {iter_num}: "
                f"train loss {total_loss / len(train_loader):.4f}, "
                f"train loss predict {total_nll / len(train_loader):.4f}, "
                f"train loss sparsity {total_spa / len(train_loader):.4f}, "
                f"time {dt * 1000:.2f}ms"
            )
        elif phase == 2:
            logging.info(
                f"iter {iter_num}: "
                f"train loss mse {total_loss / len(train_loader):.4e}, "
                f"train mean error {total_spa / len(train_loader):.4e}, "
                f"time {dt * 1000:.2f}ms"
            )
        elif phase == 3:
            logging.info(
                f"iter {iter_num}: "
                f"train loss nll {total_loss / len(train_loader):.4e}, "
                f"train mean error {total_spa / len(train_loader):.4e}, "
                f"time {dt * 1000:.2f}ms"
            )

if ddp:
    destroy_process_group()
