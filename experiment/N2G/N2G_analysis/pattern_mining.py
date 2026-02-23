import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from config_utils import (
    load_section_config as _load_section_config,
    validate_required_nonempty as _validate_required_nonempty,
)
from utils import parse_latent_to_prompt


def _load_jsonl(path, drop_last_row=True):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    if drop_last_row and data:
        data = data[:-1]
    return data


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_sleep(seconds):
    if seconds and seconds > 0:
        time.sleep(seconds)


def _parse_prompt_template(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    if "<SYSTEM_PROMPT>" not in template or "</SYSTEM_PROMPT>" not in template:
        raise ValueError("Prompt template missing <SYSTEM_PROMPT> tags")
    if "<USER_PROMPT>" not in template or "</USER_PROMPT>" not in template:
        raise ValueError("Prompt template missing <USER_PROMPT> tags")
    system_prompt = (
        template.split("<SYSTEM_PROMPT>")[1].split("</SYSTEM_PROMPT>")[0].strip()
    )
    user_prompt = template.split("<USER_PROMPT>")[1].split("</USER_PROMPT>")[0].strip()
    return system_prompt, user_prompt


def _build_messages(dp, prompt_template_path):
    system_prompt, user_prompt = _parse_prompt_template(prompt_template_path)
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt.replace(
                "{{PASTE_YOUR_DATA_HERE}}",
                parse_latent_to_prompt(dp),
            ),
        },
    ]


def _chat_generate(
    client,
    model,
    messages,
    n,
    temperature,
    top_p,
    max_tokens,
    retries,
    retry_sleep,
):
    last_error = None
    for _ in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            for choice in response.choices:
                json.loads(choice.message.content)  # validate JSON
            return [choice.message.content for choice in response.choices]
        except Exception as exc:
            last_error = exc
            _safe_sleep(retry_sleep)
    return [json.dumps({"error": str(last_error)}) for _ in range(n)]


def _process_one(dp, client, runtime_config):
    messages = _build_messages(dp, prompt_template_path=runtime_config["prompt_template_path"])
    responses = _chat_generate(
        client=client,
        model=runtime_config["model"],
        messages=messages,
        n=runtime_config["n"],
        temperature=runtime_config["temperature"],
        top_p=runtime_config["top_p"],
        max_tokens=runtime_config["max_tokens"],
        retries=runtime_config["retries"],
        retry_sleep=runtime_config["retry_sleep"],
    )
    return [json.loads(resp) for resp in responses]


def main():
    parser = argparse.ArgumentParser(description="Run N2G pattern labeling analysis")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    args = parser.parse_args()

    cfg = _load_section_config(args.config, "n2g_analysis", args.set)

    input_path = _validate_required_nonempty(cfg, "input_path", "n2g_analysis")
    output_path = _validate_required_nonempty(cfg, "output_path", "n2g_analysis")
    prompt_template_path = _validate_required_nonempty(
        cfg,
        "prompt_template_path",
        "n2g_analysis",
    )
    model = _validate_required_nonempty(cfg, "model", "n2g_analysis")

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    api_base = cfg.get("api_base") or os.environ.get("OPENAI_API_BASE")
    if api_key is None or str(api_key).strip() == "":
        raise RuntimeError("n2g_analysis.api_key is required (or set OPENAI_API_KEY)")
    api_key = str(api_key).strip()
    if api_base is not None and str(api_base).strip() == "":
        api_base = None

    runtime_config = {
        "model": model,
        "n": int(cfg.get("n", 1)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "max_tokens": int(cfg.get("max_tokens", 32768)),
        "input_path": input_path,
        "output_path": output_path,
        "prompt_template_path": prompt_template_path,
        "api_base": api_base,
        "retries": int(cfg.get("retries", 2)),
        "retry_sleep": float(cfg.get("retry_sleep", 1.0)),
        "workers": int(cfg.get("workers", 16)),
        "drop_last_row": bool(cfg.get("drop_last_row", True)),
        "config_path": args.config,
        "cli_overrides": args.set,
    }

    if not os.path.exists(runtime_config["input_path"]):
        raise FileNotFoundError(f"Input file not found: {runtime_config['input_path']}")
    if not os.path.exists(runtime_config["prompt_template_path"]):
        raise FileNotFoundError(
            f"Prompt template not found: {runtime_config['prompt_template_path']}"
        )

    output_dir = os.path.dirname(runtime_config["output_path"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_config_path = os.path.join(
        output_dir if output_dir else ".",
        f"{os.path.splitext(os.path.basename(runtime_config['output_path']))[0]}_run_config.json",
    )
    with open(run_config_path, "w", encoding="utf-8") as config_file:
        json.dump(runtime_config, config_file, indent=2, ensure_ascii=False)
    print(f"Saved run config to {run_config_path}")

    client = (
        OpenAI(api_key=api_key, base_url=api_base)
        if api_base
        else OpenAI(api_key=api_key)
    )

    data = _load_jsonl(
        runtime_config["input_path"],
        drop_last_row=runtime_config["drop_last_row"],
    )

    data_points = []
    with ThreadPoolExecutor(max_workers=runtime_config["workers"]) as executor:
        futures = [executor.submit(_process_one, dp, client, runtime_config) for dp in data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                data_points.extend(future.result())
            except Exception as exc:
                data_points.append({"error": str(exc)})

    _write_jsonl(runtime_config["output_path"], data_points)


if __name__ == "__main__":
    main()
