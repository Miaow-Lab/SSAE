import argparse
import importlib
import json
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config_utils import load_section_config as _load_section_config
from model_qwen import MyModel


def get_top_act_latents(model, dataloader, device, top_k=20):
    n_latents = model.n_latents
    accum_latents_sum = torch.zeros(n_latents, device=device)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)  # shape: (batch_size, seq_len)
        attention_mask = batch["attention_mask"].to(
            device
        )  # shape: (batch_size, seq_len)
        latents = model.sample_Tr(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).squeeze(
            1
        )  # (B,n_latents)
        batch_sum = latents.sum(dim=0)  # (n_latents)
        accum_latents_sum += batch_sum
    topk_vals, topk_indices = torch.topk(accum_latents_sum, k=top_k)
    return topk_indices.tolist()


def get_latent_activation(model, input_ids, attention_mask, latent_index, device):
    latents = model.sample_Tr(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    latents = latents.squeeze(1).detach().cpu().to(torch.float16).numpy()
    val = latents[0, latent_index]
    return float(val)


class N2G:
    def __init__(self, model, tokenizer, activation_fn, device, suffix_ratio=0.5, latent_active_threshold=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.get_activation = activation_fn
        self.device = device
        self.suffix_ratio = float(suffix_ratio)
        self.latent_active_threshold = float(latent_active_threshold)
        self.last_patterns_info = []
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def find_shortest_suffix(self, input_ids, attention_mask, latent_idx, original_act):
        seq_len = input_ids.shape[1]
        for length in range(1, seq_len + 1):
            suffix_ids = input_ids[:, -length:]
            suffix_attention_mask = (
                attention_mask[:, -length:] if attention_mask is not None else None
            )
            current_act = self.get_activation(
                self.model, suffix_ids, suffix_attention_mask, latent_idx, self.device
            )
            if current_act >= self.suffix_ratio * original_act:
                return suffix_ids, suffix_attention_mask, current_act
        return input_ids, attention_mask, original_act

    def apply_wildcards(self, input_ids, attention_mask, latent_idx, baseline_act):
        tokens = input_ids[0].tolist()
        pattern = []
        for i in range(len(tokens)):
            original_token = tokens[i]
            masked_tokens = tokens.copy()
            masked_tokens[i] = self.pad_id
            masked_input = torch.tensor([masked_tokens], device=input_ids.device)
            masked_act = self.get_activation(
                self.model, masked_input, attention_mask, latent_idx, self.device
            )
            if masked_act >= self.suffix_ratio * baseline_act:
                pattern.append("*")
            else:
                pattern.append(self.tokenizer.decode([original_token]))

        return pattern

    def explain_top_k(self, test_loader, latent_idx, k=10):
        self.model.eval()
        candidates = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                latents = self.model.sample_Tr(input_ids, attention_mask)
                acts = latents[:, :, latent_idx].float()
                max_act = acts[0].max().item()
                if max_act > self.latent_active_threshold:
                    candidates.append({
                        "act": max_act,
                        "input_ids": input_ids.cpu(), 
                        "attention_mask": attention_mask.cpu()
                    })
        candidates.sort(key=lambda x: x["act"], reverse=True)
        patterns = []
        patterns_info = []
        collected_count = 0
        for item in candidates:
            if collected_count >= k:
                break
            curr_input_ids = item["input_ids"].to(self.device)
            curr_attention_mask = item["attention_mask"].to(self.device)
            max_act = item["act"]
            short_ids, short_mask, suffix_act = self.find_shortest_suffix(
                curr_input_ids,
                curr_attention_mask,
                latent_idx,
                max_act,
            )
            # pattern generation
            p = self.apply_wildcards(
                short_ids, short_mask, latent_idx, max_act
            )
            # remove consecutive duplicates
            if p not in patterns:
                patterns.append(p)
                collected_count += 1
                original_text = self.tokenizer.decode(
                    curr_input_ids[0], skip_special_tokens=True
                )
                short_text = self.tokenizer.decode(
                    short_ids[0], skip_special_tokens=True
                )
                patterns_info.append(
                    {
                        "pattern": p,
                        "act": float(max_act),
                        "original_text": original_text,
                        "short_text": short_text,
                    }
                )
                print(
                    f"[{collected_count}/{k}] new pattern: {p} (Act: {max_act:.4f})"
                )
            else:
                pass
        self.last_patterns_info = patterns_info
        return patterns


class MultiPatternMatcher:
    def __init__(self, patterns, tokenizer):
        self.tokenizer = tokenizer
        self.matchers = []
        for p in patterns:
            p_ids = []
            for token in p:
                if token == "*":
                    p_ids.append("*")
                else:
                    ids = tokenizer.encode(token, add_special_tokens=False)
                    if ids:
                        p_ids.append(ids[0])
            self.matchers.append(p_ids)

    def predict(self, input_ids):
        seq_len = len(input_ids)
        final_prediction = False
        input_ids_list = input_ids.tolist()

        for p_ids in self.matchers:
            p_len = len(p_ids)
            if seq_len < p_len:
                continue

            for i in range(p_len - 1, seq_len):
                if final_prediction:
                    continue 

                window = input_ids_list[i - p_len + 1 : i + 1]
                match = True
                for j, p_token in enumerate(p_ids):
                    if p_token == "*":
                        continue
                    if p_token != window[j]:
                        match = False
                        break
                if match:
                    final_prediction = True

        return final_prediction


def evaluate_multi_patterns(
    model, tokenizer, dataloader, latent_idx, patterns, device, num_batches=100, latent_active_threshold=0.0
):
    matcher = MultiPatternMatcher(patterns, tokenizer)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    print(f"Evaluation: Latent {latent_idx} Pattern: {patterns}")
    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            if batch_i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            latents = (
                model.sample_Tr(input_ids, attention_mask).squeeze(1).detach().cpu()
            )  # [B, Latent]
            acts = latents[:, latent_idx].cpu().float()
            for i in range(input_ids.shape[0]):
                seq_ids = input_ids[i]
                real_acts = acts[i]
                valid_len = attention_mask[i].sum().item()
                seq_ids = seq_ids[:valid_len]
                pred_mask = matcher.predict(seq_ids)  # bool
                is_active_real = real_acts > latent_active_threshold
                is_active_pred = pred_mask
                if is_active_pred and is_active_real:
                    true_positives += 1
                elif is_active_pred and not is_active_real:
                    false_positives += 1
                elif not is_active_pred and is_active_real:
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "stats": (true_positives, false_positives, false_negatives),
    }


def main():
    parser = argparse.ArgumentParser(description="Run N2G latent pattern mining and evaluation")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "n2g", args.set)

    required_keys = ["input_file", "eval_file"]
    for key in required_keys:
        if not cfg.get(key):
            raise ValueError(f"n2g.{key} is required")

    checkpoint_name = cfg.get("checkpoint_name")
    if checkpoint_name is None or str(checkpoint_name).strip() == "":
        raise ValueError("n2g.checkpoint_name is required")

    model_dir = cfg.get("model_dir")
    if model_dir is None or str(model_dir).strip() == "":
        raise ValueError("n2g.model_dir is required")

    runtime_config = {
        "init_name": str(cfg.get("init_name", "Qwen/Qwen2.5-0.5B")),
        "checkpoint_name": str(checkpoint_name).strip(),
        "model_dir": str(model_dir).strip(),
        "input_file": str(cfg["input_file"]),
        "eval_file": str(cfg["eval_file"]),
        "batch_size": int(cfg.get("batch_size", 1)),
        "sparsity_factor": int(cfg.get("sparsity_factor", 1)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "device": str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        "top_latents_k": int(cfg.get("top_latents_k", 50)),
        "pattern_k": int(cfg.get("pattern_k", 20)),
        "eval_batches": int(cfg.get("eval_batches", 500)),
        "suffix_ratio": float(cfg.get("suffix_ratio", 0.5)),
        "latent_active_threshold": float(cfg.get("latent_active_threshold", 0.0)),
        "out_dir": str(cfg.get("out_dir", "n2g_results")),
        "use_numina_dataloader": bool(cfg.get("use_numina_dataloader", False)),
        "config_path": args.config,
        "cli_overrides": args.set,
    }

    init_name = runtime_config["init_name"]
    checkpoint_name = runtime_config["checkpoint_name"]
    model_dir = runtime_config["model_dir"]
    input_file = runtime_config["input_file"]
    eval_file = runtime_config["eval_file"]
    batch_size = runtime_config["batch_size"]
    sparsity_factor = runtime_config["sparsity_factor"]
    device = runtime_config["device"]

    tokenizer = AutoTokenizer.from_pretrained(init_name)
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
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

    if 'numina' in runtime_config["dataset"].lower():
        dataloader_module = importlib.import_module("dataloader_numina")
    elif 'gsm8k' in runtime_config["dataset"].lower():
        dataloader_module = importlib.import_module("dataloader")
    else:
        dataloader_module = importlib.import_module("dataloader_opencodeinstruct")
    problem_dataset_cls = dataloader_module.ProblemAnswerDataset
    collate_cls = dataloader_module.CollateFn

    print("Loading model...")
    ckpt_path = os.path.join(model_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_name = checkpoint.get("encoder_name", init_name)
    decoder_name = checkpoint.get("decoder_name", init_name)
    model = MyModel(
        tokenizer,
        sparsity_factor,
        init_from=(encoder_name, decoder_name),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(checkpoint["best_val_loss"])
    model.eval()
    input_dataset = problem_dataset_cls(input_file, tokenizer)
    input_loader = DataLoader(
        input_dataset,
        batch_size=batch_size,
        collate_fn=collate_cls(
            tokenizer.eos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id
        ),
    )
    eval_dataset = problem_dataset_cls(eval_file, tokenizer)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_cls(
            tokenizer.eos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id
        ),
    )
    target_latents = get_top_act_latents(
        model,
        input_loader,
        device=device,
        top_k=runtime_config["top_latents_k"],
    )
    n2g = N2G(
        model,
        tokenizer,
        get_latent_activation,
        device=device,
        suffix_ratio=runtime_config["suffix_ratio"],
        latent_active_threshold=runtime_config["latent_active_threshold"],
    )

    # Accumulate metrics to compute averages later
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    results = []

    out_dir = runtime_config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    run_config_path = os.path.join(
        out_dir,
        f"{os.path.splitext(checkpoint_name)[0]}_run_config.json",
    )
    with open(run_config_path, "w", encoding="utf-8") as config_file:
        json.dump(runtime_config, config_file, indent=2, ensure_ascii=False)
    print(f"Saved run config to {run_config_path}")

    result_jsonl_path = os.path.join(out_dir, f"{init_name.split('/')[-1]}.jsonl")
    if os.path.exists(result_jsonl_path):
        os.remove(result_jsonl_path)

    for target_latent in target_latents:
        print(f"Latent {target_latent}...")
        pattern = n2g.explain_top_k(input_loader, target_latent, k=runtime_config["pattern_k"])
        print("\n=== Begin N2G Evaluation ===")

        metrics = evaluate_multi_patterns(
            model=model,
            tokenizer=tokenizer,
            dataloader=eval_loader,
            latent_idx=target_latent,
            patterns=pattern,
            device=device,
            num_batches=runtime_config["eval_batches"],
            latent_active_threshold=runtime_config["latent_active_threshold"],
        )
        print(f"Latent {target_latent} evaluation results:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        # Save results to jsonl
        results.append({
            "latent_idx": target_latent,
            "pattern": pattern,
            "pattern_info": n2g.last_patterns_info,
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "tp": int(metrics["stats"][0]),
            "fp": int(metrics["stats"][1]),
            "fn": int(metrics["stats"][2]),
            "num_latents": len(target_latents),
        })

        precision_sum += float(metrics["precision"])
        recall_sum += float(metrics["recall"])
        f1_sum += float(metrics["f1"])
        tp_sum += int(metrics["stats"][0])
        fp_sum += int(metrics["stats"][1])
        fn_sum += int(metrics["stats"][2])

    # Save results to jsonl
    # Use micro-averaged metrics based on total TP/FP/FN
    micro_precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    micro_recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    micro_f1 = (
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
    )
    results.append(
        {
            "latent_idx": "average",
            "pattern": None,
            "pattern_info": None,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum,
            "num_latents": len(target_latents),
        }
    )
        
    with open(result_jsonl_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
