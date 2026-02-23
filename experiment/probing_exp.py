import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer

from classifier.classifier import Classifier
from config_utils import (
    load_section_config as _load_section_config,
    validate_required_nonempty as _validate_required_nonempty,
)
from model_qwen import MyModel


def _load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            data.append(json.loads(line))
    return data


def _build_runtime_config(cfg, args):
    checkpoint_name = _validate_required_nonempty(cfg, "checkpoint_name", "probing_exp")
    model_dir = _validate_required_nonempty(cfg, "model_dir", "probing_exp")
    input_file = _validate_required_nonempty(cfg, "input_file", "probing_exp")
    classifier_ckpt_path = _validate_required_nonempty(cfg, "classifier_ckpt_path", "probing_exp")

    task = input_file.split("/")[-1].replace(".json", "")
    return {
        "init_name": str(cfg.get("init_name", "Qwen/Qwen2.5-0.5B")),
        "input_file": input_file,
        "checkpoint_name": checkpoint_name,
        "classifier_ckpt_path": classifier_ckpt_path,
        "task": str(cfg.get("task", task)),
        "model_dir": model_dir,
        "sparsity_factor": int(cfg.get("sparsity_factor", 1)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "device": str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        "n_latents": int(cfg.get("n_latents", 896)),
        "hidden_dim": int(cfg.get("hidden_dim", 1024)),
        "k": int(cfg.get("k", 16)),
        "tau": float(cfg.get("tau", 0.8)),
        "classifier_dropout": float(cfg.get("classifier_dropout", 0.1)),
        "greedy_top_k": int(cfg.get("greedy_top_k", 200)),
        "majority_temperature": float(cfg.get("majority_temperature", 0.7)),
        "majority_top_p": float(cfg.get("majority_top_p", 0.9)),
        "majority_top_k": int(cfg.get("majority_top_k", 200)),
        "out_dir": str(cfg.get("out_dir", "probe_results")),
        "run_greedy": bool(cfg.get("run_greedy", True)),
        "run_majority": bool(cfg.get("run_majority", True)),
        "run_probe_guided": bool(cfg.get("run_probe_guided", True)),
        "config_path": args.config,
        "cli_overrides": args.set,
    }


def _build_tokenizer(init_name):
    tokenizer = AutoTokenizer.from_pretrained(init_name)
    # tokenizer.add_special_tokens({"sep_token": "<sep>"})
    # tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")
    return tokenizer


def _load_main_model(runtime_config, tokenizer):
    print("Loading model...")
    ckpt_path = os.path.join(runtime_config["model_dir"], runtime_config["checkpoint_name"])
    checkpoint = torch.load(ckpt_path, map_location=runtime_config["device"])
    # encoder_name = checkpoint.get("encoder_name", runtime_config["init_name"])
    # decoder_name = checkpoint.get("decoder_name", runtime_config["init_name"])
    encoder_name = runtime_config["init_name"]
    decoder_name = runtime_config["init_name"]
    model = MyModel(
        tokenizer,
        runtime_config["sparsity_factor"],
        init_from=(encoder_name, decoder_name),
    ).to(runtime_config["device"])
    model.load_state_dict(checkpoint["model"], strict=True)
    print(checkpoint.get("best_val_loss", "N/A"))
    model.eval()
    return model


def _load_classifier(runtime_config, tokenizer):
    classifier_ckpt = torch.load(runtime_config["classifier_ckpt_path"], map_location=runtime_config["device"])
    num_token = tokenizer.vocab_size
    classifier = Classifier(
        input_dim=runtime_config["n_latents"],
        hidden_dim=runtime_config["hidden_dim"],
        dropout=runtime_config["classifier_dropout"],
        task=runtime_config["task"],
        num_token=num_token,
    )
    print("Loading classifier...")
    state_dict = classifier_ckpt["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    classifier.load_state_dict(new_state_dict)
    classifier.to(runtime_config["device"])
    classifier.eval()
    return classifier


def trim_repeated_last_char(text, min_repeats=3, allowed_chars=None):
    if not text:
        return text
    last_char = text[-1]
    if allowed_chars is not None and last_char not in allowed_chars:
        return text
    idx = len(text) - 1
    while idx >= 0 and text[idx] == last_char:
        idx -= 1
    repeat_count = len(text) - 1 - idx
    if repeat_count >= min_repeats:
        return text[: idx + 2]
    return text


def trim_repeated_last_token(text, tokenizer, min_repeats=3):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return text
    last_id = token_ids[-1]
    idx = len(token_ids) - 1
    while idx >= 0 and token_ids[idx] == last_id:
        idx -= 1
    repeat_count = len(token_ids) - 1 - idx
    if repeat_count >= min_repeats:
        trimmed_ids = token_ids[: idx + 2]
        text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    text = trim_repeated_last_char(
        text,
        min_repeats=min_repeats,
        allowed_chars=set("#.'\"`!? ,:;)]}-_*/~".replace(" ", "")),
    )
    return text


def sample_responses(
    model,
    tokenizer,
    device,
    query,
    num_return_sequences,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
):
    model_inputs = tokenizer([query], return_tensors="pt").to(device)
    input_len = len(model_inputs.input_ids[0])

    do_sample = True if temperature > 0 else False

    generated_ids_tensor = model.decoder.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens_ids_lists = []

    for output_ids in generated_ids_tensor:
        new_tokens = output_ids[input_len:]
        new_tokens_list = new_tokens.tolist()
        new_tokens_ids_lists.append(new_tokens_list)

    responses = tokenizer.batch_decode(new_tokens_ids_lists, skip_special_tokens=True)
    return responses


def gsm8k_acc_judge(ground_step, pred):
    '''
    extract the last number from ground_step and pred, and compare them
    '''
    ground_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", ground_step)
    pred_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
    if not ground_numbers or not pred_numbers:
        return 0
    ground_answer = ground_numbers[-1]
    pred_answer = pred_numbers[-1]
    try:
        if abs(float(ground_answer) - float(pred_answer)) < 1e-3:
            return 1
        else:
            return 0
    except:
        return 0


def extract_answer_from_response(response):
    '''
    extract the last number from response
    '''
    pred_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    if pred_numbers:
        return pred_numbers[-1]
    return None


def greedy(runtime_config, model, tokenizer, input_data):
    temperature = 0.0
    top_p = 1.0
    top_k = runtime_config["greedy_top_k"]
    num_return_sequences = 1
    correct_count = 0
    total_count = 0
    for q_a in input_data:
        query = q_a["question"]
        answer = q_a["answer"]
        pred_answers = sample_responses(
            model,
            tokenizer,
            runtime_config["device"],
            query,
            num_return_sequences=num_return_sequences,
            max_new_tokens=runtime_config["max_new_tokens"],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        print(f"Question: {query}")
        print(f"Ground answer: {answer}")
        print(f"Pred answers: {pred_answers}")
        correct_count += sum([gsm8k_acc_judge(answer, pred) for pred in pred_answers])
        total_count += len(pred_answers)
        print(f"Current accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
    final_acc = correct_count / total_count if total_count else 0.0
    print(f"Final accuracy: {correct_count}/{total_count} = {final_acc:.4f}")
    return final_acc
        

def major_voting(runtime_config, model, tokenizer, input_data):
    temperature = runtime_config["majority_temperature"]
    top_p = runtime_config["majority_top_p"]
    top_k = runtime_config["majority_top_k"]
    k = runtime_config["k"]
    num_return_sequences = k 
    correct_count = 0
    total_count = 0
    majority_file = os.path.join(
        runtime_config["out_dir"],
        f"{runtime_config['task']}_{runtime_config['checkpoint_name']}_majority_voting.json",
    )
    if os.path.exists(majority_file):
        os.remove(majority_file)

    for q_a in input_data:
        query = q_a["question"]
        answer = q_a["answer"]
        pred_answers = sample_responses(
            model,
            tokenizer,
            runtime_config["device"],
            query,
            num_return_sequences=num_return_sequences,
            max_new_tokens=runtime_config["max_new_tokens"],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        results = {
            "question": query,
            "ground_answer": answer,
            "pred_answers": pred_answers
        }
        with open(majority_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(results) + "\n")
        answer_count = {}
        for pred in pred_answers:
            extracted_answer = extract_answer_from_response(pred)
            if extracted_answer is not None:
                if extracted_answer not in answer_count:
                    answer_count[extracted_answer] = 0
                answer_count[extracted_answer] += 1
        if answer_count:
            majority_answer = max(answer_count.items(), key=lambda x: x[1])[0]
        else:
            majority_answer = None
        ground_extracted_answer = extract_answer_from_response(answer)
        if majority_answer is not None and ground_extracted_answer is not None:
            try:
                if abs(float(majority_answer) - float(ground_extracted_answer)) < 1e-3:
                    correct_count += 1
            except:
                pass
        total_count += 1
        print(f"Question: {query}")
        print(f"Ground answer: {answer}")
        print(f"Majority answer: {majority_answer}")
        print(f"Current accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
    final_acc = correct_count / total_count if total_count else 0.0
    print(f"Final accuracy: {correct_count}/{total_count} = {final_acc:.4f}")
    return final_acc, majority_file


def answer_split(answer):
    answer = answer.replace('\n\n', '\n')
    sentences = answer.split('\n')
    try:
        sentences = sentences[:-2] + [sentences[-2]+'\n'+sentences[-1]]
    except:
        sentences = answer.split('. ')
    return sentences


def probe_guided(runtime_config, model, tokenizer, mv_file):
    mv_data = []
    cf_model = _load_classifier(runtime_config, tokenizer)

    with open(mv_file, 'r', encoding='utf-8') as f:
        for line in f:
            mv_data.append(json.loads(line))
    correct_count = 0
    total_count = 0
    for item in mv_data:
        question = item['question']
        ground_answer = item['ground_answer']
        pred_answers = item['pred_answers']
        answer_count = {}
        for pred in pred_answers:
            pred_answer = extract_answer_from_response(pred)
            if pred_answer is None:
                continue
            pred_steps = answer_split(pred)
            pred_score = 0
            for i in range(len(pred_steps)):
                hint = question + " " + "".join(pred_steps[:i])
                step = pred_steps[i]
                hint_tokens = torch.tensor(tokenizer.encode(hint, max_length=256, truncation=True), dtype=torch.long)
                step_tokens = torch.tensor(tokenizer.encode(step, max_length=256, truncation=True), dtype=torch.long)
                input_tokens = torch.cat([hint_tokens, torch.tensor([tokenizer.sep_token_id], dtype=torch.long)])
                input_tokens = torch.cat([input_tokens, step_tokens])
                input_tokens = torch.cat([input_tokens, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)])
                attention_mask = torch.ones(len(input_tokens), dtype=torch.long)
                input_tokens = input_tokens.unsqueeze(0).to(runtime_config["device"])
                attention_mask = attention_mask.unsqueeze(0).to(runtime_config["device"])
                with torch.no_grad():
                    Tr = model.sample_Tr(input_tokens, attention_mask)
                logits = cf_model(Tr)
                probs = torch.sigmoid(logits)
                pred_score += probs.item() ** runtime_config["tau"]
            pred_score /= len(pred_steps)
            if pred_answer not in answer_count:
                answer_count[pred_answer] = 0
            answer_count[pred_answer] += pred_score
        if answer_count:
            best_answer = max(answer_count.items(), key=lambda x: x[1])[0]
        else:
            best_answer = None
        ground_extracted_answer = extract_answer_from_response(ground_answer)
        if best_answer is not None and ground_extracted_answer is not None:
            try:
                if abs(float(best_answer) - float(ground_extracted_answer)) < 1e-3:
                    correct_count += 1
            except:
                pass
        total_count += 1
        print(f"Current accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
    final_acc = correct_count / total_count if total_count else 0.0
    print(f"Final accuracy: {correct_count}/{total_count} = {final_acc:.4f}")
    return final_acc
            

def main():
    parser = argparse.ArgumentParser(description="Run probing experiment pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "probing_exp", args.set)
    runtime_config = _build_runtime_config(cfg, args)

    os.makedirs(runtime_config["out_dir"], exist_ok=True)
    run_config_path = os.path.join(
        runtime_config["out_dir"],
        f"{os.path.splitext(runtime_config['checkpoint_name'])[0]}_probing_run_config.json",
    )
    with open(run_config_path, "w", encoding="utf-8") as config_file:
        json.dump(runtime_config, config_file, indent=2, ensure_ascii=False)
    print(f"Saved run config to {run_config_path}")

    tokenizer = _build_tokenizer(runtime_config["init_name"])
    model = _load_main_model(runtime_config, tokenizer)
    input_data = _load_jsonl(runtime_config["input_file"])

    majority_file = os.path.join(
        runtime_config["out_dir"],
        f"{runtime_config['task']}_{runtime_config['checkpoint_name']}_majority_voting.json",
    )

    if runtime_config["run_greedy"]:
        greedy(runtime_config, model, tokenizer, input_data)

    if runtime_config["run_majority"]:
        _, majority_file = major_voting(runtime_config, model, tokenizer, input_data)

    if runtime_config["run_probe_guided"]:
        if not os.path.exists(majority_file):
            raise FileNotFoundError(
                f"Majority voting file not found: {majority_file}. "
                "Run with run_majority=true or provide matching settings."
            )
        probe_guided(runtime_config, model, tokenizer, majority_file)


if __name__ == "__main__":
    main()
    