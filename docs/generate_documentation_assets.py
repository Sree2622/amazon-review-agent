from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


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

COLORS = ["#94a3b8", "#0f766e", "#0284c7", "#7c3aed", "#db2777"]


def style_chart(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def create_accuracy_chart() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(MODELS, ACCURACY, color=COLORS, width=0.62)
    style_chart(ax, "Accuracy across ML and LLM models")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter("{x:.0%}")
    for bar, value in zip(bars, ACCURACY):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", fontweight="bold")
    save_figure("model_accuracy_comparison.png")


def create_metrics_chart() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    x = np.arange(len(MODELS))
    width = 0.23
    metric_colors = ["#0f766e", "#0284c7", "#7c3aed"]
    for index, (metric, values) in enumerate(METRICS.items()):
        ax.bar(x + (index - 1) * width, values, width, label=metric, color=metric_colors[index])
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
    bars = ax.bar(stages, values, color=["#94a3b8", "#0f766e"], width=0.52)
    style_chart(ax, "Preprocessing retention after validation")
    ax.set_ylim(0, 800_000)
    ax.set_ylabel("Review count")
    ax.yaxis.set_major_formatter("{x:,.0f}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 20_000, f"{value:,}", ha="center", fontweight="bold")
    ax.text(0.5, 90_000, "7,981 records removed: missing reviews and duplicates", ha="center", color="#475569")
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
