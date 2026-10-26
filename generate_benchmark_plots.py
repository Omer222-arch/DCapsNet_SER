"""Generate publication-grade benchmark plots for DCapsNet and baseline models on RAVDESS."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300
})

OUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Load Data
with open(os.path.join("outputs", "bench_dcapsnet_protocol_a.json"), "r") as f:
    dcapsnet_data = json.load(f)

with open(os.path.join("outputs", "bench_baselines_protocol_a.json"), "r") as f:
    baselines_data = json.load(f)

cnn_data = next(r for r in baselines_data if r["model"] == "1d_cnn")
bilstm_data = next((r for r in baselines_data if r["model"] == "bilstm"), None)

classes = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
cap_classes = [c.capitalize() for c in classes]

# -----------------------------------------------------------------------------
# 1. Confusion Matrix Heatmaps
# -----------------------------------------------------------------------------
def plot_cm(cm_data, title, filename, cmap="Blues"):
    cm = np.array(cm_data)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap=cmap, cbar=True,
                xticklabels=cap_classes, yticklabels=cap_classes, ax=ax,
                vmin=0, vmax=1.0, linewidths=0.5, linecolor="#eeeeee")

    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Emotion", labelpad=10, fontweight="semibold")
    ax.set_ylabel("True Emotion", labelpad=10, fontweight="semibold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

plot_cm(dcapsnet_data["confusion_matrix"], "DCapsNet (Proposed) Confusion Matrix\n10-Fold CV on RAVDESS (Mean Acc: 63.54%)", "confusion_matrix_dcapsnet.png", "Blues")
plot_cm(cnn_data["confusion_matrix"], "1D-CNN Baseline Confusion Matrix\n10-Fold CV on RAVDESS (Mean Acc: 51.81%)", "confusion_matrix_1dcnn.png", "Purples")

# -----------------------------------------------------------------------------
# 2. Overall Model Metrics Comparison Bar Chart
# -----------------------------------------------------------------------------
def plot_model_comparison():
    models = ["DCapsNet\n(Proposed)", "1D-CNN\n(Baseline)", "BiLSTM\n(Baseline Fold 1)"]
    mean_acc = [dcapsnet_data["mean_acc"] * 100, cnn_data["mean_acc"] * 100, 51.39]
    peak_acc = [max(dcapsnet_data["fold_acc"]) * 100, max(cnn_data["fold_acc"]) * 100, 51.39]
    macro_f1 = [dcapsnet_data["macro_f1"] * 100, cnn_data["macro_f1"] * 100, 50.12]
    weighted_f1 = [dcapsnet_data["weighted_f1"] * 100, cnn_data["weighted_f1"] * 100, 51.08]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 6))
    r1 = ax.bar(x - 1.5 * width, mean_acc, width, label="Mean Accuracy (%)", color="#1f77b4", edgecolor="black", linewidth=0.6)
    r2 = ax.bar(x - 0.5 * width, peak_acc, width, label="Peak Accuracy (%)", color="#2ca02c", edgecolor="black", linewidth=0.6)
    r3 = ax.bar(x + 0.5 * width, macro_f1, width, label="Macro F1-Score (%)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    r4 = ax.bar(x + 1.5 * width, weighted_f1, width, label="Weighted F1 (%)", color="#9467bd", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Percentage (%)", fontweight="semibold")
    ax.set_title("Performance Comparison: DCapsNet vs. Baselines on RAVDESS", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0, 85)
    ax.legend(frameon=True, loc="upper right")

    # Add data labels
    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison_bar.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

plot_model_comparison()

# -----------------------------------------------------------------------------
# 3. Class-by-Class F1-Score Grouped Bar Chart
# -----------------------------------------------------------------------------
def plot_class_f1():
    dcaps_f1 = [dcapsnet_data["class_report"][c]["f1-score"] * 100 for c in classes]
    cnn_f1 = [cnn_data["class_report"][c]["f1-score"] * 100 for c in classes]

    y_pos = np.arange(len(classes))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6.5))
    b1 = ax.barh(y_pos - height/2, dcaps_f1, height, label="DCapsNet (Proposed)", color="#1f77b4", edgecolor="black", linewidth=0.6)
    b2 = ax.barh(y_pos + height/2, cnn_f1, height, label="1D-CNN (Baseline)", color="#e377c2", edgecolor="black", linewidth=0.6)

    ax.set_xlabel("F1-Score (%)", fontweight="semibold")
    ax.set_title("Per-Class F1-Score: DCapsNet vs. 1D-CNN on RAVDESS", pad=15, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cap_classes, fontweight="semibold")
    ax.set_xlim(0, 90)
    ax.legend(frameon=True, loc="lower right")

    # Value annotations and delta highlights
    for i, (v1, v2) in enumerate(zip(dcaps_f1, cnn_f1)):
        delta = v1 - v2
        sign = "+" if delta >= 0 else ""
        ax.text(v1 + 1.2, i - height/2, f"{v1:.1f}%", va="center", fontsize=9, fontweight="bold", color="#0b4074")
        ax.text(v2 + 1.2, i + height/2, f"{v2:.1f}%", va="center", fontsize=9, color="#6b2b57")
        ax.text(max(v1, v2) + 11.0, i, f"({sign}{delta:.1f}%)", va="center", fontsize=8.5, fontweight="bold",
                color="#008000" if delta > 0 else "#cc0000")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "class_f1_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

plot_class_f1()

# -----------------------------------------------------------------------------
# 4. Fold-by-Fold Cross-Validation Trend
# -----------------------------------------------------------------------------
def plot_fold_trend():
    folds = list(range(1, 11))
    dcaps_acc = [a * 100 for a in dcapsnet_data["fold_acc"]]
    cnn_acc = [a * 100 for a in cnn_data["fold_acc"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(folds, dcaps_acc, marker="o", linewidth=2.5, color="#1f77b4", label="DCapsNet (Mean: 63.5%)", markersize=7)
    ax.plot(folds, cnn_acc, marker="s", linewidth=2.2, color="#e377c2", linestyle="--", label="1D-CNN (Mean: 51.8%)", markersize=6)

    ax.axhline(np.mean(dcaps_acc), color="#1f77b4", linestyle=":", alpha=0.6, label="DCapsNet Mean Line")
    ax.axhline(np.mean(cnn_acc), color="#e377c2", linestyle=":", alpha=0.6, label="1D-CNN Mean Line")

    ax.set_xlabel("Cross-Validation Fold Index", fontweight="semibold")
    ax.set_ylabel("Test Accuracy (%)", fontweight="semibold")
    ax.set_title("10-Fold Stratified Cross-Validation Accuracy Trajectory", pad=15, fontweight="bold")
    ax.set_xticks(folds)
    ax.set_ylim(35, 80)
    ax.legend(frameon=True, loc="upper right")

    # Annotate peak fold
    peak_idx = np.argmax(dcaps_acc)
    ax.annotate(f"Peak Fold 4: {dcaps_acc[peak_idx]:.2f}%",
                xy=(peak_idx + 1, dcaps_acc[peak_idx]),
                xytext=(peak_idx + 0.3, dcaps_acc[peak_idx] + 3.5),
                arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="#0b4074", fontsize=9.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fold_accuracy_trend.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

plot_fold_trend()

# -----------------------------------------------------------------------------
# 5. Latency vs Accuracy Scatter Profile
# -----------------------------------------------------------------------------
def plot_latency_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ["1D-CNN", "DCapsNet", "BiLSTM"]
    latencies = [2.86, 8.24, 24.67]
    accuracies = [51.81, 63.54, 51.39]
    colors = ["#e377c2", "#1f77b4", "#ff7f0e"]
    sizes = [180, 260, 220]

    for m, lat, acc, col, sz in zip(models, latencies, accuracies, colors, sizes):
        ax.scatter(lat, acc, s=sz, color=col, edgecolor="black", linewidth=1.2, label=m, zorder=3)
        ax.text(lat + 0.6, acc + 0.4, f"{m}\n({acc:.1f}%, {lat:.2f}ms)", fontsize=9.5, fontweight="bold", color=col)

    ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
    ax.text(48, 48, "Real-Time Threshold (< 50 ms)", rotation=90, color="gray", fontsize=8.5)

    ax.set_xlabel("Inference Latency (ms / sample)", fontweight="semibold")
    ax.set_ylabel("Mean Accuracy (%)", fontweight="semibold")
    ax.set_title("Accuracy vs. Inference Latency Trade-Off", pad=15, fontweight="bold")
    ax.set_xlim(0, 32)
    ax.set_ylim(45, 70)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "latency_accuracy_tradeoff.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

plot_latency_accuracy()
print("\nAll publication-grade benchmark plots generated successfully in outputs/plots/!")

