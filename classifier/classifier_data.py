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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from config_utils import (
    load_section_config as _load_section_config,
)
from model_qwen import MyModel


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=rank,
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
    print(f"Begin sampling on GPU {rank} ...")
    setup(rank, world_size)

    device = rank
    torch.cuda.set_device(device)
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    tokenizer = AutoTokenizer.from_pretrained(args["init_name"])
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

    if args.get("use_numina_dataloader", False):
        dataloader_module = importlib.import_module("dataloader_numina")
    else:
        dataloader_module = importlib.import_module("dataloader")
    problem_dataset_cls = dataloader_module.ProblemAnswerDataset
    collate_cls = dataloader_module.CollateFn

    dataset = problem_dataset_cls(args["input_file"], tokenizer)
    print(f"GPU {rank} load dataset, data size: {len(dataset)}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        sampler=sampler,
        collate_fn=collate_cls(tokenizer.eos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id),
    )

    if rank == 0:
        print("loading model...")
    map_location = f"cuda:{rank}"
    ckpt_path = os.path.join(args["out_dir"], args["checkpoint_name"])
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    encoder_name, decoder_name = args["init_name"], args["init_name"]
    model = MyModel(tokenizer, int(args["sparsity_factor"]), init_from=(encoder_name, decoder_name))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    if rank == 0:
        print(f"ckpt best validation loss: {checkpoint['best_val_loss']}")

    model.eval()
    if rank == 0:
        print("model loaded")

    local_latents_list = []
    local_hints_list = []
    local_step_length_list = []
    local_begin_token_id_list = []
    local_correctness_list = []
    local_ids_list = []

    temp_jsonl_file = f"{args['output_file']}_judge_part_{rank}.jsonl"
    os.makedirs(os.path.dirname(temp_jsonl_file), exist_ok=True)

    jsonl_file = open(temp_jsonl_file, "w", encoding="utf-8")

    with torch.no_grad(), ctx:
        progress_bar = tqdm(dataloader, desc=f"GPU {rank}", disable=(rank != 0))
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hints_sep_ids = batch["hints_sep_ids"].to(device)
            hints_sep_attention_masks = batch["hints_sep_attention_masks"].to(device)
            sep_pos = batch["sep_pos"]

            hints_ids = torch.zeros(
                (hints_sep_ids.size(0), hints_sep_ids.size(1) - 1), dtype=torch.long
            ).to(device)
            hints_attention_masks = torch.zeros(
                (hints_sep_attention_masks.size(0), hints_sep_attention_masks.size(1) - 1), dtype=torch.long
            ).to(device)
            for i in range(hints_sep_ids.size(0)):
                hints_ids[i] = torch.cat([hints_sep_ids[i, : sep_pos[i] - 1], hints_sep_ids[i, sep_pos[i] :]])
                hints_attention_masks[i] = torch.cat(
                    [
                        hints_sep_attention_masks[i, : sep_pos[i] - 1],
                        hints_sep_attention_masks[i, sep_pos[i] :],
                    ]
                )

            steps = batch["steps"]
            hints = batch["hints"]
            for step in steps:
                step_tokens = torch.tensor(
                    tokenizer.encode(step, max_length=128, truncation=True), dtype=torch.long
                )
                local_step_length_list.append(len(step_tokens))
                if "Llama" in args["init_name"] or "Gemma" in args["init_name"]:
                    begin_token_id = step_tokens[1].item()
                else:
                    begin_token_id = step_tokens[0].item()
                local_begin_token_id_list.append(begin_token_id)

            hints_last_token_embeddings = model.sample_hint_emb(hints_ids, hints_attention_masks)
            hints_emb_fp16 = hints_last_token_embeddings.squeeze(1).detach().cpu().to(torch.float16).numpy()
            for i in range(hints_emb_fp16.shape[0]):
                local_hints_list.append(hints_emb_fp16[i])

            latents, decode_text_ids, _ = model.manual_generate_sentence(
                input_ids,
                attention_mask,
                hints_sep_ids,
                hints_sep_attention_masks,
                args["temperature"],
                args["top_k"],
                args["top_p"],
                args["max_new_tokens"],
            )
            response = tokenizer.batch_decode(decode_text_ids, skip_special_tokens=True)

            correctness_list = []
            for idx in range(len(steps)):
                hint = hints[idx]
                ground_step = steps[idx]
                pred = response[idx]
                unique_id = f"{rank}_{len(local_ids_list)}"
                if "gsm8k_385k" in args["task"].lower():
                    acc = gsm8k_acc_judge(ground_step, pred)
                    local_ids_list.append(unique_id)
                elif "math" in args["task"].lower():
                    prompt = math_acc_judge(hint, ground_step, pred)
                    jsonl_file.write(json.dumps({"id": unique_id, "query": prompt}, ensure_ascii=False) + "\n")
                    local_ids_list.append(unique_id)
                    acc = -1
                else:
                    print("Unknown task for accuracy judgement.")
                    acc = 0
                    local_ids_list.append(unique_id)
                correctness_list.append(acc)
            local_correctness_list.extend(correctness_list)

            latents_fp16 = latents.squeeze(1).detach().cpu().to(torch.float16).numpy()
            for i in range(latents_fp16.shape[0]):
                local_latents_list.append(latents_fp16[i])

    jsonl_file.close()

    if local_latents_list:
        all_latents = np.vstack(local_latents_list)
        all_hints = np.vstack(local_hints_list)
        all_step_length = np.array(local_step_length_list, dtype=int)
        all_begin_token_id = np.array(local_begin_token_id_list, dtype=int)
        all_correctness = np.array(local_correctness_list, dtype=int)
        all_ids = np.array(local_ids_list)

        temp_output_file = f"{args['output_file']}.part_{rank}.npz"
        np.savez_compressed(
            temp_output_file,
            latents=all_latents,
            hints=all_hints,
            step_length=all_step_length,
            begin_token_id=all_begin_token_id,
            correctness=all_correctness,
            ids=all_ids,
        )
        print(f"GPU {rank} finish sampling, data save to {temp_output_file}")

    dist.barrier()

    if rank == 0:
        print("All GPU finish sampling, begin merging files...")

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Generate classifier data")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "classifier_data", args.set)

    checkpoint_name = cfg.get("checkpoint_name")
    if checkpoint_name is None or str(checkpoint_name).strip() == "":
        raise ValueError("classifier_data.checkpoint_name is required")

    model_dir = cfg.get("model_dir")
    if model_dir is None or str(model_dir).strip() == "":
        raise ValueError("classifier_data.model_dir is required")

    config = {
        "init_name": str(cfg.get("init_name", "Qwen/Qwen2.5-0.5B")),
        "checkpoint_name": str(checkpoint_name).strip(),
        "out_dir": str(model_dir).strip(),
        "sparsity_factor": float(cfg.get("sparsity_factor", 1.0)),
        "seed": int(cfg.get("seed", 1337)),
        "batch_size": int(cfg.get("batch_size", 64)),
        "input_file": str(cfg.get("input_file", "data/gsm8k_385K_valid.json")),
        "output_file": str(cfg.get("output_file", "data/gsm8k_385k_valid_classifier_data")).removesuffix(".pt"),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0)),
        "top_k": int(cfg.get("top_k", 200)),
        "top_p": float(cfg.get("top_p", 0.95)),
        "use_numina_dataloader": bool(cfg.get("use_numina_dataloader", False)),
    }

    config["task"] = config["input_file"].split("/")[-1].replace(".json", "")
    print(f"task: {config['task']}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("ERROR: No GPU detected.")
        return

    print(f"Using {world_size} GPU")

    mp.spawn(run_sampling, args=(world_size, config), nprocs=world_size, join=True)

    print("\nAll GPU processes finished sampling, begin merging.")
    final_output_file_path = config["output_file"] + ".npz"
    final_output_jsonl_path = config["output_file"] + "_judge.jsonl"

    final_latents_parts = []
    final_hints_parts = []
    final_step_length_parts = []
    final_begin_token_id_parts = []
    final_correctness_parts = []
    final_ids_parts = []

    with open(final_output_jsonl_path, "w", encoding="utf-8") as outfile:
        for i in range(world_size):
            part_file = f"{config['output_file']}.part_{i}.npz"
            try:
                data = np.load(part_file)
                final_latents_parts.append(data["latents"])
                final_hints_parts.append(data["hints"])
                final_step_length_parts.append(data["step_length"])
                final_begin_token_id_parts.append(data["begin_token_id"])
                final_correctness_parts.append(data["correctness"])
                final_ids_parts.append(data["ids"])
                os.remove(part_file)
            except FileNotFoundError:
                print(f"WARNING: No temp JSONL file {part_file}")

            part_jsonl = f"{config['output_file']}_judge_part_{i}.jsonl"
            if os.path.exists(part_jsonl):
                with open(part_jsonl, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(part_jsonl)
            else:
                print(f"WARNING: No temp JSONL file {part_jsonl}")

    if final_latents_parts:
        full_latents = np.vstack(final_latents_parts)
        full_hints = np.vstack(final_hints_parts)
        full_step_length = np.concatenate(final_step_length_parts)
        full_begin_token_id = np.concatenate(final_begin_token_id_parts)
        full_correctness = np.concatenate(final_correctness_parts)
        full_ids = np.concatenate(final_ids_parts)

        print(f"Saving merge files to {final_output_file_path} ...")
        np.savez_compressed(
            final_output_file_path,
            latents=full_latents,
            hints=full_hints,
            step_length=full_step_length,
            begin_token_id=full_begin_token_id,
            correctness=full_correctness,
            ids=full_ids,
        )
        print("Merge complete.")
        print(f"File path: '{final_output_file_path}'")
        print(f"Data size: {len(full_hints)}")
    else:
        print("WARNING: No data found to merge.")


if __name__ == "__main__":
    main()
