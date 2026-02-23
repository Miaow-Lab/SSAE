import matplotlib.pyplot as plt
import numpy as np

# Data Setup
data_raw = {
    "opencodeinstruct-qwen": {
        "categories": [
            "Documentation & Docstrings",
            "Function Logic & Returns",
            "General Syntax & Structure",
            "Variables & Data Flow",
            "Control Flow (Loops/Cond)",
            "Unclassified / Mixed",
            "Classes & OOP",
            "String & Types",
        ],
        "percentages": [25.5, 19.6, 19.6, 11.8, 9.8, 5.9, 3.9, 3.9],
    },
    "numina-qwen": {
        "categories": [
            "Derivation & Calculation",
            "Reasoning & Logic",
            "Narrative & Explanation",
            "Syntax & Formatting",
        ],
        "percentages": [37, 30, 17, 16],
    },
    "gsm8k-qwen": {
        "categories": [
            "Final Answer Resolution",
            "Intermediate Calculation",
            "Reasoning & Context",
            "Structural & Syntax",
        ],
        "percentages": [36, 26, 20, 18],
    },
    "gsm8k-llama": {
        "categories": ["Reasoning Process", "Results & Units", "Final Resolution"],
        "percentages": [40.4, 34.6, 25.0],
    },
}

# Unification Mapping
mapping = {
    "Reasoning & Logic": "Reasoning",
    "Reasoning & Context": "Reasoning",
    "Reasoning Process": "Reasoning",
    "Final Answer Resolution": "Final Resolution",
    "General Syntax & Structure": "Syntax & Structure",
    "Syntax & Formatting": "Syntax & Structure",
    "Structural & Syntax": "Syntax & Structure",
    "Derivation & Calculation": "Calculation & Derivation",
    "Intermediate Calculation": "Calculation & Derivation",
    "Results & Units": "Calculation & Derivation",
}

# --- Data Preparation Logic ---
plot_data = []

# 1. Opencode
cats = [mapping.get(c, c) for c in data_raw["opencodeinstruct-qwen"]["categories"]]
vals = data_raw["opencodeinstruct-qwen"]["percentages"]
plot_data.append(
    {
        "title": "Opencode-Qwen",
        "categories": cats,
        "series": [{"name": "Qwen", "values": vals, "color": "#8DA0CB"}],
    }
)

# 2. Numina
cats = [mapping.get(c, c) for c in data_raw["numina-qwen"]["categories"]]
vals = data_raw["numina-qwen"]["percentages"]
plot_data.append(
    {
        "title": "Numina-Qwen",
        "categories": cats,
        "series": [{"name": "Qwen", "values": vals, "color": "#8DA0CB"}],
    }
)

# 3. GSM8K (Grouped)
# Unite categories
qwen_map = {
    mapping.get(c, c): v
    for c, v in zip(
        data_raw["gsm8k-qwen"]["categories"], data_raw["gsm8k-qwen"]["percentages"]
    )
}
llama_map = {
    mapping.get(c, c): v
    for c, v in zip(
        data_raw["gsm8k-llama"]["categories"], data_raw["gsm8k-llama"]["percentages"]
    )
}
all_cats = list(set(qwen_map.keys()) | set(llama_map.keys()))
all_cats.sort(key=lambda x: qwen_map.get(x, 0), reverse=True)  # Sort by Qwen value

plot_data.append(
    {
        "title": "GSM8K (Qwen vs Llama)",
        "categories": all_cats,
        "series": [
            {
                "name": "Qwen",
                "values": [qwen_map.get(c, 0) for c in all_cats],
                "color": "#8DA0CB",
            },
            {
                "name": "Llama",
                "values": [llama_map.get(c, 0) for c in all_cats],
                "color": "#E1974C",
            },
        ],
    }
)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes = axes.flatten()

for i, data in enumerate(plot_data):
    ax = axes[i]
    y_pos = np.arange(len(data["categories"]))

    # Handle Grouped Bars (GSM8K)
    if len(data["series"]) > 1:
        height = 0.35
        # Qwen (Top offset)
        bars1 = ax.barh(
            y_pos - height / 2,
            data["series"][0]["values"],
            height,
            label=data["series"][0]["name"],
            color=data["series"][0]["color"],
        )
        # Llama (Bottom offset)
        bars2 = ax.barh(
            y_pos + height / 2,
            data["series"][1]["values"],
            height,
            label=data["series"][1]["name"],
            color=data["series"][1]["color"],
        )

        # Add Values
        for bars in [bars1, bars2]:
            for bar in bars:
                if bar.get_width() > 0:
                    ax.text(
                        bar.get_width() + 1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width()}%",
                        va="center",
                        ha="left",
                        fontsize=12,
                        fontweight="bold",
                        color=bar.get_facecolor(),
                    )
        ax.legend(fontsize=14)

    # Handle Single Bars
    else:
        bars = ax.barh(
            y_pos,
            data["series"][0]["values"],
            align="center",
            color=data["series"][0]["color"],
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width()}%",
                va="center",
                ha="left",
                fontsize=14,
            )

    # Styling
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["categories"], fontsize=16)
    ax.set_xlabel("Percentage (%)", fontsize=16)
    ax.set_title(data["title"], fontsize=20, fontweight="bold")

    # Adjust X-Limits
    all_vals = [v for s in data["series"] for v in s["values"]]
    ax.set_xlim(0, max(all_vals) * 1.25)

plt.tight_layout()
plt.savefig("n2g_patterns_label_stats.pdf", format="pdf", bbox_inches="tight")
plt.show()
