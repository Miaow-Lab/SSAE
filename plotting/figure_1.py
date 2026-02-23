import numpy as np
import matplotlib.pyplot as plt

# Increase font sizes for better readability
plt.rcParams.update(
    {
        "font.size": 14,  # General font size
        "axes.titlesize": 16,  # Title font size
        "axes.labelsize": 14,  # Axis label font size
        "xtick.labelsize": 12,  # X-axis tick label size
        "ytick.labelsize": 12,  # Y-axis tick label size
        "legend.fontsize": 14,  # Legend font size
    }
)

# 数据准备
methods = ["SSAE", "Token-SAE", "Statistic"]
tasks = ["GSM8K (SL)", "GSM8K (PPL)", "MATH-500 (SL)", "MATH-500 (PPL)"]

SSAE_Qwen_NSL = [2.10, 1.94]
SSAE_Qwen_PPL = [4.09, 1.46]
TSAE_Qwen_NSL = [29.06, 31.58]
TSAE_Qwen_PPL = [103.54, 49.17]
SBL_NSL = [28.04, 33.30]
SBL_PPL = [61.01, 74.96]

nsl = np.array([SSAE_Qwen_NSL, TSAE_Qwen_NSL, SBL_NSL])
ppl = np.array([SSAE_Qwen_PPL, TSAE_Qwen_PPL, SBL_PPL])

# 相对性能：以 SBL 为 naive baseline (1.0)
baseline_nsl = np.array(SBL_NSL)
baseline_ppl = np.array(SBL_PPL)

rel = np.zeros((len(methods), 4))
rel[:, 0] = nsl[:, 0] / baseline_nsl[0]  # GSM8K SL
rel[:, 1] = ppl[:, 0] / baseline_ppl[0]  # GSM8K PPL
rel[:, 2] = nsl[:, 1] / baseline_nsl[1]  # MATH-500 SL
rel[:, 3] = ppl[:, 1] / baseline_ppl[1]  # MATH-500 PPL

# 画图参数
bar_width = 0.18
group_gap = 0.35
mid_cluster_shift = 0.18
last_cluster_shift = 0.25

fig, ax = plt.subplots(figsize=(8.8, 5.2))

# 颜色：按方法区分
method_colors = {
    "SSAE": "#F28E2B",
    "Token-SAE": "#B07AA1",
    "Statistic": "#8CD17D",
}

# 分簇绘制（任务为簇）
x_centers = []
current = 0.0
for t_idx, task in enumerate(tasks):
    offsets = np.linspace(-bar_width, bar_width, len(methods))
    for m_idx, m_name in enumerate(methods):
        ax.bar(
            current + offsets[m_idx],
            rel[m_idx, t_idx],
            width=bar_width,
            color=method_colors[m_name],
            label=m_name if t_idx == 0 else None,
        )
    x_centers.append(current)
    current += (len(methods) * bar_width) + group_gap
    if t_idx == 1:
        current += mid_cluster_shift
    if t_idx == len(tasks) - 2:
        current += last_cluster_shift

# 基线参考线
ax.axhline(1.0, color="#999999", linewidth=1, linestyle="--")

# 轴与美化
ax.set_xticks(x_centers)
ax.set_xticklabels(tasks, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
ax.set_ylabel("Relative Performance", fontweight="bold")

# 图例：左上角三行（稍微放大）
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    frameon=False,
    ncol=1,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.98),
    borderaxespad=0.0,
    fontsize=16,
)

plt.tight_layout()
# save pdf
plt.savefig("figure_1_relative_performance.pdf")
