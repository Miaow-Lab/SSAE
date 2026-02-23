import argparse
import datetime
import importlib
import json
import os
import re

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sae_lens import SAE, HookedSAETransformer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_utils import (
    load_section_config as _load_section_config,
    validate_required_nonempty as _validate_required_nonempty,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=180),
    )


def cleanup():
    dist.destroy_process_group()


def gsm8k_acc_judge(ground_step, pred):
    ground_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", ground_step)
    pred_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
    if not ground_numbers or not pred_numbers:
        return 0
    ground_answer = ground_numbers[-1]
    pred_answer = pred_numbers[-1]
    try:
        if abs(float(ground_answer) - float(pred_answer)) < 1e-3:
            return 1
        return 0
    except Exception:
        return 0


def math_acc_judge(query, ground_step, pred):
    prompt = f"""
### Role
Act as an expert mathematics evaluator. Your task is to verify both the correctness and the logical soundness of a student's step-by-step reasoning.

### Instructions
Compare the **Student's Proposed Step** with the **Reference Next Step** based on two independent dimensions:

1. **Conclusion (Correctness)**:
   - Mark as "Correct" if the mathematical result or statement is accurate and leads towards the solution.
   - Mark as "Incorrect" if there are calculation errors, sign errors, or if the step is mathematically false.

2. **Logicality (Reasoning Path)**:
   - **Independent of Correctness**: A step can be "Logical" even if the final calculation is "Incorrect."
   - Mark as "Logical" if the student's intent aligns with the problem's goal, follows valid mathematical rules/properties, and is contextually relevant.
   - Mark as "Illogical" if the step is a non-sequitur, violates fundamental logical rules, or ignores the problem's constraints.

3. **Flexibility**: Note that the student may use a different method than the reference. If the logic is sound, it should be marked as "Logical."

### Output Format
You must follow this exact format:

Conclusion: <Correct/Incorrect>
Logicality: <Logical/Illogical>
Analysis: <Your analysis here, explaining why the step is correct/incorrect and why it is logical/illogical>

### Examples
Input:
- **Problem Statement and existing steps**: There are 1 apples and 1 oranges in a basket. How many pieces of fruit are there in total?
- **Reference Next Step:**: There are 2 pieces of fruit in total because $1 + 1 = 2$.
- **Student's Proposed Step:** : Apples and oranges are both fruits.
Output:
Conclusion: Correct
Logicality: Logical
Analysis: The student correctly identifies the category of the items, which is a necessary logical precursor to summing them as "fruit."

Input:
- **Problem Statement and existing steps**: If $x + 2 = 5$, what is the value of $x$?
- **Reference Next Step:**: Subtract 2 from both sides to get $x = 3$.
- **Student's Proposed Step:** : Add 2 to both sides to get $x = 7$.
Output:
Conclusion: Incorrect
Logicality: Illogical
Analysis: The step is mathematically incorrect. Furthermore, it is illogical because adding 2 to both sides moves away from the goal of isolating $x$, violating the logic of solving linear equations.

Input:
- **Problem Statement and existing steps**: A triangle has two angles of $60^\circ$ and $60^\circ$. Find the third angle.
- **Reference Next Step:**: The sum of angles is $180^\circ$, so $180 - 60 - 60 = 60^\circ$.
- **Student's Proposed Step:** : Since the sum of angles in a triangle is $180^\circ$, the third angle is $180 - 60 - 60 = 80^\circ$.
Output:
Conclusion: Incorrect
Logicality: Logical
Analysis: The conclusion is incorrect due to a simple subtraction error ($180 - 120$ is not $80$). However, the step is logical because the student correctly identified and applied the "triangle angle sum theorem," which is the appropriate logical path for this problem.

### Your Turn
Input:
- **Problem Statement and existing steps:** {query}
- **Reference Next Step:** {ground_step}
- **Student's Proposed Step:** {pred}
Output:
"""
    return prompt


def run_sampling(rank, world_size, args):
    setup(rank, world_size)
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"Loading SAE: {args['sae_path']}...")
    sae = SAE.load_from_disk(args["sae_path"], device=device)

    if rank == 0:
        print(f"Loading Model: {args['init_name']}...")
    model = HookedSAETransformer.from_pretrained_no_processing(
        args["init_name"],
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()

    hook_point_layer = str(sae.cfg.metadata.hook_name).strip()
    hook_point_sae = f"{hook_point_layer}.hook_sae_acts_post"

    tokenizer = AutoTokenizer.from_pretrained(args["init_name"])
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_model = AutoModelForCausalLM.from_pretrained(
        args["init_name"],
        torch_dtype=torch.bfloat16,
    ).to(device)
    hf_model.eval()

    if args.get("use_numina_dataloader", False):
        dataloader_module = importlib.import_module("dataloader_numina")
    else:
        dataloader_module = importlib.import_module("dataloader")
    problem_dataset_cls = dataloader_module.ProblemAnswerDataset
    collate_cls = dataloader_module.CollateFn

    dataset = problem_dataset_cls(args["input_file"], tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        sampler=sampler,
        collate_fn=collate_cls(
            tokenizer.eos_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
        ),
    )

    local_sae_acts, local_layer_acts = [], []
    local_correctness, local_ids = [], []
    local_step_length, local_begin_token_id = [], []

    jsonl_filename = f"{args['output_file']}_judge_part_{rank}.jsonl"
    os.makedirs(os.path.dirname(jsonl_filename) or ".", exist_ok=True)
    f_jsonl = open(jsonl_filename, "w", encoding="utf-8")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"GPU {rank}", disable=(rank != 0)):
            hints_sep_ids = batch["hints_sep_ids"].to(device)
            hints_sep_attention_masks = batch["hints_sep_attention_masks"].to(device)
            sep_pos = batch["sep_pos"]

            batch_size, seq_len = hints_sep_ids.shape
            hints_ids = torch.zeros((batch_size, seq_len - 1), dtype=torch.long, device=device)
            hints_masks = torch.zeros((batch_size, seq_len - 1), dtype=torch.long, device=device)

            for idx in range(batch_size):
                curr_sep = sep_pos[idx]
                hints_ids[idx] = torch.cat(
                    [hints_sep_ids[idx, : curr_sep - 1], hints_sep_ids[idx, curr_sep:]]
                )
                hints_masks[idx] = torch.cat(
                    [
                        hints_sep_attention_masks[idx, : curr_sep - 1],
                        hints_sep_attention_masks[idx, curr_sep:],
                    ]
                )

            last_token_indices = hints_masks.sum(dim=1) - 1
            batch_indices = torch.arange(batch_size, device=device)

            _, sae_cache = model.run_with_cache_with_saes(
                hints_ids,
                saes=[sae],
                names_filter=[hook_point_layer, hook_point_sae],
            )
            curr_sae_acts = sae_cache[hook_point_sae][batch_indices, last_token_indices]
            del sae_cache

            _, layer_cache = model.run_with_cache(
                hints_ids,
                names_filter=[hook_point_layer],
            )
            curr_layer_acts = layer_cache[hook_point_layer][batch_indices, last_token_indices]
            del layer_cache

            local_layer_acts.append(curr_layer_acts.cpu().to(torch.float16).numpy())
            local_sae_acts.append(curr_sae_acts.cpu().to(torch.float16).numpy())

            seq_lens = hints_masks.sum(dim=1)
            max_len = hints_ids.shape[1]
            pad_len = max_len - seq_lens
            left_padded_ids = torch.full_like(hints_ids, tokenizer.pad_token_id)
            left_padded_masks = torch.zeros_like(hints_masks)
            for idx in range(batch_size):
                valid_len = int(seq_lens[idx].item())
                left_padded_ids[idx, pad_len[idx] :] = hints_ids[idx, :valid_len]
                left_padded_masks[idx, pad_len[idx] :] = hints_masks[idx, :valid_len]

            gen_tokens = hf_model.generate(
                input_ids=left_padded_ids,
                attention_mask=left_padded_masks,
                max_new_tokens=args["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_only = gen_tokens[:, left_padded_ids.shape[1] :]
            response = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            steps = batch["steps"]
            hints = batch["hints"]

            for step in steps:
                step_tokens = torch.tensor(
                    tokenizer.encode(step, max_length=128, truncation=True),
                    dtype=torch.long,
                )
                local_step_length.append(len(step_tokens))
                begin_token_id = step_tokens[0].item() if len(step_tokens) > 0 else 0
                local_begin_token_id.append(begin_token_id)

            for idx in range(len(response)):
                unique_id = f"{rank}_{len(local_ids)}"
                local_ids.append(unique_id)

                ground_step = steps[idx]
                pred = response[idx]
                hint = hints[idx]

                if args["task"] == "gsm8k":
                    acc = gsm8k_acc_judge(ground_step, pred)
                elif args["task"] == "math":
                    prompt = math_acc_judge(hint, ground_step, pred)
                    f_jsonl.write(
                        json.dumps({"id": unique_id, "query": prompt}, ensure_ascii=False)
                        + "\n"
                    )
                    acc = -1
                else:
                    acc = 0
                local_correctness.append(acc)

    f_jsonl.close()

    if local_layer_acts:
        np.savez_compressed(
            f"{args['output_file']}.part_{rank}.npz",
            layer_acts=np.vstack(local_layer_acts),
            sae_acts=np.vstack(local_sae_acts),
            correctness=np.array(local_correctness),
            step_length=np.array(local_step_length),
            begin_token_id=np.array(local_begin_token_id),
            ids=np.array(local_ids),
        )

    dist.barrier()
    cleanup()


def _infer_task(input_file, explicit_task=None):
    if explicit_task:
        return str(explicit_task)
    lowered = input_file.lower()
    if "gsm8k" in lowered:
        return "gsm8k"
    if "math" in lowered:
        return "math"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Generate token-SAE baseline classifier data")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "token_sae_data", args.set)

    sae_path = _validate_required_nonempty(cfg, "sae_path", "token_sae_data")
    input_file = _validate_required_nonempty(cfg, "input_file", "token_sae_data")
    output_file = _validate_required_nonempty(cfg, "output_file", "token_sae_data")

    config = {
        "init_name": str(cfg.get("init_name", "Qwen/Qwen2.5-0.5B")),
        "sae_path": sae_path,
        "input_file": input_file,
        "output_file": output_file.removesuffix(".pt"),
        "batch_size": int(cfg.get("batch_size", 8)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "use_numina_dataloader": bool(cfg.get("use_numina_dataloader", False)),
        "task": _infer_task(input_file, cfg.get("task")),
        "config_path": args.config,
        "cli_overrides": args.set,
    }

    os.makedirs(os.path.dirname(config["output_file"]) or ".", exist_ok=True)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("ERROR: No GPU detected.")
        return

    mp.spawn(run_sampling, args=(world_size, config), nprocs=world_size, join=True)

    final_layer_parts = []
    final_sae_parts = []
    final_correctness_parts = []
    final_step_length_parts = []
    final_begin_token_id_parts = []
    final_ids_parts = []

    for idx in range(world_size):
        part_file = f"{config['output_file']}.part_{idx}.npz"
        try:
            data = np.load(part_file)
            final_layer_parts.append(data["layer_acts"])
            final_sae_parts.append(data["sae_acts"])
            final_correctness_parts.append(data["correctness"])
            final_step_length_parts.append(data["step_length"])
            final_begin_token_id_parts.append(data["begin_token_id"])
            final_ids_parts.append(data["ids"])
            os.remove(part_file)
        except FileNotFoundError:
            print(f"Warning: Missing part {idx}")

    if final_layer_parts:
        full_layer = np.vstack(final_layer_parts)
        full_sae = np.vstack(final_sae_parts)
        full_ids = np.concatenate(final_ids_parts)
        full_correctness = np.concatenate(final_correctness_parts)
        full_step_length = np.concatenate(final_step_length_parts)
        full_begin_token_id = np.concatenate(final_begin_token_id_parts)

        final_path = config["output_file"] + ".npz"
        np.savez_compressed(
            final_path,
            layer_acts=full_layer,
            sae_acts=full_sae,
            correctness=full_correctness,
            step_length=full_step_length,
            begin_token_id=full_begin_token_id,
            ids=full_ids,
        )
        print(f"Saved merged file to {final_path}")

    judge_parts = []
    for idx in range(world_size):
        part_jsonl = f"{config['output_file']}_judge_part_{idx}.jsonl"
        if os.path.exists(part_jsonl):
            judge_parts.append(part_jsonl)

    if judge_parts:
        merged_jsonl = f"{config['output_file']}_judge.jsonl"
        with open(merged_jsonl, "w", encoding="utf-8") as out_file:
            for part_jsonl in judge_parts:
                with open(part_jsonl, "r", encoding="utf-8") as in_file:
                    for line in in_file:
                        out_file.write(line)
                os.remove(part_jsonl)
        print(f"Merged judge file to {merged_jsonl}")


if __name__ == "__main__":
    main()
