import json


def parse_latent_to_prompt(data: dict) -> str:
    """
    Parses a single line of JSONL containing latent feature data.
    Treats patterns as independent triggers rather than a sequence.
    """

    # 1. Extract Header Info
    idx = data.get("latent_idx", "N/A")

    # 2. Extract Statistics
    precision = data.get("precision", 0)
    recall = data.get("recall", 0)
    f1 = data.get("f1", 0)

    # 3. Format the Global Pattern List (Independent Patterns)
    # The 'pattern' key is a list of lists. We format each sub-list as a string.
    raw_patterns = data.get("pattern") or []
    formatted_patterns = []

    if isinstance(raw_patterns, list):
        for p in raw_patterns:
            if isinstance(p, list):
                # Join tokens in the pattern (e.g., [" True", "*"] -> " True *")
                # We strip extra whitespace for cleanliness but keep internal spaces
                p_str = " ".join([str(token) for token in p]).strip()
                formatted_patterns.append(f"`{p_str}`")

    # Join them with commas or display as a set
    patterns_display = (
        ", ".join(formatted_patterns) if formatted_patterns else "(No patterns found)"
    )

    # 4. Build the Output String
    output = []
    output.append(f"## Latent Feature Analysis: Index {idx}")
    output.append("-" * 40)
    output.append(
        f"**Statistics:** Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
    )

    output.append(f"\n### Associated Patterns (Independent)")
    output.append(
        f"This feature activates on the following independent patterns:\n{patterns_display}"
    )

    output.append("\n### Representative Examples")
    output.append("Specific instances where these patterns triggered an activation:\n")

    # 5. Format the Pattern Info (The Examples)
    examples = data.get("pattern_info") or []

    if not examples:
        output.append("_No examples provided._")

    for i, ex in enumerate(examples, 1):
        if not isinstance(ex, dict):
            continue

        act_score = ex.get("act", 0)
        short_text = str(ex.get("short_text", "")).strip()
        original_text = ex.get("original_text", "")

        # specific pattern for this example
        ex_pattern_raw = ex.get("pattern", [])
        ex_pattern_str = (
            " ".join([str(t) for t in ex_pattern_raw]).strip()
            if isinstance(ex_pattern_raw, list)
            else "N/A"
        )

        output.append(f"**Example {i}**")
        output.append(f"- **Specific Pattern:** `{ex_pattern_str}`")
        output.append(f"- **Trigger Text:** `{short_text}`")
        output.append(f"- **Activation Score:** {act_score:.4f}")
        output.append(f"**Context:**\n```\n{original_text}\n```")
        output.append("-" * 20)

    return "\n".join(output)


if __name__ == "__main__":
    # --- Usage Example ---

    N2G_pattern_path = "/home/ningmiao/yhlai/dyco/out_gsm8k-385k/dwa_10.0/n2g_results_Qwen2.5-0.5B.jsonl"

    with open(N2G_pattern_path, "r") as f:
        for line in f:
            raw_json = line.strip()
            formatted_output = parse_latent_to_prompt(json.loads(raw_json))
            print(formatted_output)
