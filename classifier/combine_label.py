import argparse
import json
import os
import re

import numpy as np
from config_utils import (
    load_section_config as _load_section_config,
)


def parse_correctness_response(response_text):
    if not response_text:
        return 0

    match = re.search(r"Conclusion:\s*(Correct|Incorrect)", response_text, re.IGNORECASE)
    if match:
        result = match.group(1).lower()
        return 1 if result == "correct" else 0

    if "correct" in response_text.lower() and "incorrect" not in response_text.lower():
        return 1
    if "incorrect" in response_text.lower():
        return 0

    return 0


def parse_logicality_response(response_text):
    if not response_text:
        return 0

    match = re.search(r"Logicality:\s*(Logical|Illogical)", response_text, re.IGNORECASE)
    if match:
        result = match.group(1).lower()
        return 1 if result == "logical" else 0

    if "logical" in response_text.lower() and "illogical" not in response_text.lower():
        return 1
    if "illogical" in response_text.lower():
        return 0

    return 0


def main():
    parser = argparse.ArgumentParser(description="Merge LLM labels into npz")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_section_config(args.config, "combine_label", args.set)

    npz_file = cfg.get("npz_file")
    llm_response_file = cfg.get("llm_response_file")
    output_file = cfg.get("output_file")
    response_key = cfg.get("response_key", "response")

    if npz_file is None or str(npz_file).strip() == "":
        raise ValueError("combine_label.npz_file is required")
    if llm_response_file is None or str(llm_response_file).strip() == "":
        raise ValueError("combine_label.llm_response_file is required")

    npz_file = str(npz_file).strip()
    llm_response_file = str(llm_response_file).strip()

    if output_file is None:
        output_file = str(npz_file).replace(".npz", "_labeled.npz")

    print(f"Loading data from {npz_file}...")
    data = np.load(npz_file)

    data_dict = dict(data)
    ids = data_dict["ids"]
    original_correctness = data_dict["correctness"]

    print(f"Loading LLM responses from {llm_response_file}...")
    llm_results = {}
    with open(llm_response_file, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            item = json.loads(line)
            if response_key not in item:
                raise KeyError(
                    f"Missing response key '{response_key}' in item with id={item.get('id')}"
                )
            llm_results[str(item["id"])] = item[response_key]

    print("Merging labels...")
    new_correctness = original_correctness.copy()
    new_logicality = np.full_like(original_correctness, -1)

    hit_count = 0
    miss_count = 0
    for idx, uid in enumerate(ids):
        uid_str = str(uid)
        if uid_str in llm_results:
            response = llm_results[uid_str]
            new_correctness[idx] = parse_correctness_response(response)
            new_logicality[idx] = parse_logicality_response(response)
            hit_count += 1
        else:
            miss_count += 1

    print(f"Matched {hit_count} items. {miss_count} items unchanged (or missing).")

    data_dict["correctness"] = new_correctness
    data_dict["logicality"] = new_logicality

    print(f"Saving to {output_file}...")
    np.savez_compressed(output_file, **data_dict)

    print("Verification:")
    print(f"Correctness shape: {new_correctness.shape}")
    print(f"Logicality shape: {new_logicality.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
