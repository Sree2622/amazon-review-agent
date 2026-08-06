from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    PALETTE_ML = sns.color_palette("Blues_d", 3) + sns.color_palette("light:teal", 0)
    # Soft, muted seaborn palette (5 tones) for the 5 models
    MODEL_COLORS = sns.color_palette("pastel", 5)
    METRIC_COLORS = sns.color_palette("muted", 3)
    STAGE_COLORS = sns.color_palette("pastel", 2)
except ImportError:
    sns = None
    # Fallback: manually chosen soft/light hex tones mimicking seaborn "pastel"
    MODEL_COLORS = ["#a1c9f4", "#8de5a1", "#ffb482", "#d0bbff", "#fab0e4"]
    METRIC_COLORS = ["#4c72b0", "#55a868", "#c44e52"]
    STAGE_COLORS = ["#a1c9f4", "#8de5a1"]


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = Path(__file__).resolve().parent / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["Logistic\nRegression", "Random\nForest", "XGBoost", "Qwen 2.5\nBase", "Qwen 2.5\nLoRA"]
ACCURACY = [0.521275, 0.833667, 0.818629, 0.94, 0.97]
METRICS = {
    "Precision": [0.736805, 0.814840, 0.798757, 0.956364, 0.974737],
    "Recall": [0.521275, 0.833667, 0.818629, 0.94, 0.97],
    "F1 Score": [0.583243, 0.817283, 0.785476, 0.922237, 0.967195],
}

BAR_WIDTH_SINGLE = 0.42   # thin single-series bars
BAR_WIDTH_GROUPED = 0.18  # thin grouped bars


def style_chart(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def create_accuracy_chart() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(
        MODELS, ACCURACY,
        color=MODEL_COLORS, width=BAR_WIDTH_SINGLE,
        edgecolor="white", linewidth=1.2,
    )
    style_chart(ax, "Accuracy across ML and LLM models")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter("{x:.0%}")
    for bar, value in zip(bars, ACCURACY):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}",
                 ha="center", fontweight="bold", color="#334155")
    save_figure("model_accuracy_comparison.png")


def create_metrics_chart() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    x = np.arange(len(MODELS))
    width = BAR_WIDTH_GROUPED
    for index, (metric, values) in enumerate(METRICS.items()):
        ax.bar(
            x + (index - 1) * width, values, width,
            label=metric, color=METRIC_COLORS[index],
            edgecolor="white", linewidth=1.0,
        )
    style_chart(ax, "Weighted precision, recall, and F1 score")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    save_figure("model_metrics_comparison.png")


def create_preprocessing_chart() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    stages = ["Raw reviews", "After validation"]
    values = [701_528, 693_547]
    bars = ax.bar(
        stages, values, color=STAGE_COLORS, width=BAR_WIDTH_SINGLE,
        edgecolor="white", linewidth=1.2,
    )
    style_chart(ax, "Preprocessing retention after validation")
    ax.set_ylim(0, 800_000)
    ax.set_ylabel("Review count")
    ax.yaxis.set_major_formatter("{x:,.0f}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 20_000, f"{value:,}",
                 ha="center", fontweight="bold", color="#334155")
    ax.text(0.5, 90_000, "7,981 records removed: missing reviews and duplicates",
             ha="center", color="#64748b")
    save_figure("preprocessing_retention.png")


def copy_architecture() -> None:
    source = ROOT_DIR / "Architecture.png"
    if source.exists():
        shutil.copy2(source, IMAGE_DIR / "architecture.png")


if __name__ == "__main__":
    create_accuracy_chart()
    create_metrics_chart()
    create_preprocessing_chart()
    copy_architecture()