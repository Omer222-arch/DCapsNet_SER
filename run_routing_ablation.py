"""Dynamic Routing Iteration Ablation Study for DCapsNet (r = 1 vs. r = 2 vs. r = 3)."""
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import model as _model
from model import DigitCaps, CapsuleLength, margin_loss

OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def build_dcapsnet_ablation(n_classes, n_features=217, dropout=0.1, routings=3):
    inp = keras.Input(shape=(n_features,), name="input")
    x = layers.Reshape((n_features, 1))(inp)
    for i, (f, k, s) in enumerate([(32, 9, 1), (64, 9, 1), (128, 7, 1), (256, 7, 2)], 1):
        x = layers.Conv1D(f, k, strides=s, padding="valid", name=f"conv{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(128, 5, strides=2, padding="valid", activation="relu",
                      name="primary_caps")(x)
    x = layers.Reshape((-1, 8))(x)
    v = DigitCaps(n_classes, 16, routings=routings, name="digit_caps")(x)
    out = CapsuleLength(name="capsule_lengths")(v)
    return keras.Model(inp, out, name=f"DCapsNet_r{routings}")


def run_ablation():
    tf.keras.utils.set_random_seed(42)

    cache_file = os.path.join(OUT_DIR, "ravdess_features.npz")
    data = np.load(cache_file, allow_pickle=True)
    X, X_aug, y, classes = data["X"], data["X_aug"], data["y"], list(data["classes"])
    n_classes = len(classes)

    splits_file = os.path.join(OUT_DIR, "splits_ravdess.json")
    with open(splits_file, "r") as f:
        splits = json.load(f)["protocol_a_10fold"][:3]  # Folds 1, 2, 3 for fair ablation

    routing_configs = [1, 2, 3]
    ablation_results = []

    print("=" * 70)
    print(" STARTING DYNAMIC ROUTING ABLATION STUDY (r = 1 vs. r = 2 vs. r = 3)")
    print("=" * 70)

    for r in routing_configs:
        print(f"\n--- Testing Routing Iterations: r = {r} ---")
        accs, fit_times, latencies = [], [], []
        y_true_all, y_pred_all = [], []

        for fold_idx, split in enumerate(splits, 1):
            tr = np.array(split["train"])
            te = np.array(split["test"])

            tr_sub, val_sub = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=42)
            scaler = StandardScaler().fit(X[tr_sub])

            n_aug = X_aug.shape[1]
            Xt = np.concatenate([X[tr_sub], X_aug[tr_sub].reshape(-1, X.shape[1])])
            yt = np.concatenate([y[tr_sub], np.repeat(y[tr_sub], n_aug)])
            Xv, yv = X[val_sub], y[val_sub]

            Xt, Xv = scaler.transform(Xt), scaler.transform(Xv)
            X_test_scaled = scaler.transform(X[te])

            net = build_dcapsnet_ablation(n_classes, X.shape[1], dropout=0.1, routings=r)
            net.compile(optimizer=keras.optimizers.Adam(3e-4), loss=margin_loss,
                        metrics=[keras.metrics.CategoricalAccuracy(name="acc")])

            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0)
            ]

            t0 = time.time()
            net.fit(
                Xt, keras.utils.to_categorical(yt, n_classes),
                validation_data=(Xv, keras.utils.to_categorical(yv, n_classes)),
                epochs=25,
                batch_size=16,
                callbacks=callbacks,
                verbose=0
            )
            fit_time = time.time() - t0

            # Measure latency
            t_inf0 = time.time()
            raw_preds = net.predict(X_test_scaled, verbose=0)
            latency_ms = ((time.time() - t_inf0) / len(te)) * 1000.0

            preds = raw_preds.argmax(axis=1)
            acc = float(accuracy_score(y[te], preds))

            accs.append(acc)
            fit_times.append(fit_time)
            latencies.append(latency_ms)
            y_true_all.extend(y[te])
            y_pred_all.extend(preds)

            print(f"  Fold {fold_idx}/3 | Acc: {acc*100:.2f}% | Train Time: {fit_time:.1f}s | Latency: {latency_ms:.2f}ms")

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro"))
        weighted_f1 = float(f1_score(y_true_all, y_pred_all, average="weighted"))
        mean_lat = float(np.mean(latencies))
        total_time = float(np.sum(fit_times))

        rep = classification_report(y_true_all, y_pred_all, target_names=classes, output_dict=True)

        res = {
            "routings": r,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mean_latency_ms": mean_lat,
            "total_train_time_sec": total_time,
            "fold_acc": accs,
            "class_f1": {c: round(rep[c]["f1-score"], 4) for c in classes}
        }
        ablation_results.append(res)
        print(f"  --> r={r} Summary: Mean Acc = {mean_acc*100:.2f}%, Macro F1 = {macro_f1:.4f}, Latency = {mean_lat:.2f}ms")

    # Save JSON results
    res_file = os.path.join(OUT_DIR, "bench_routing_ablation.json")
    with open(res_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved ablation results to {res_file}")

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    r_vals = [res["routings"] for res in ablation_results]
    accs = [res["mean_acc"] * 100 for res in ablation_results]
    f1s = [res["macro_f1"] * 100 for res in ablation_results]
    lats = [res["mean_latency_ms"] for res in ablation_results]

    # Plot 1: Accuracy & Macro F1
    ax1.plot(r_vals, accs, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Mean Accuracy (%)")
    ax1.plot(r_vals, f1s, marker="s", color="#2ca02c", linewidth=2.2, linestyle="--", markersize=7, label="Macro F1-Score (%)")
    for r, acc in zip(r_vals, accs):
        ax1.annotate(f"{acc:.2f}%", (r, acc), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold", color="#1f77b4")
    ax1.set_title("Recognition Accuracy vs. Routing Iterations (r)", pad=12, fontweight="bold")
    ax1.set_xlabel("Dynamic Routing Iterations (r)", fontweight="semibold")
    ax1.set_ylabel("Percentage (%)", fontweight="semibold")
    ax1.set_xticks(r_vals)
    ax1.set_ylim(min(f1s) - 5, max(accs) + 5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Plot 2: Latency vs Routing Iterations
    bars = ax2.bar(r_vals, lats, color="#ff7f0e", width=0.4, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f} ms", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontweight="bold", color="#b35400")
    ax2.set_title("Inference Latency vs. Routing Iterations (r)", pad=12, fontweight="bold")
    ax2.set_xlabel("Dynamic Routing Iterations (r)", fontweight="semibold")
    ax2.set_ylabel("Latency (ms / sample)", fontweight="semibold")
    ax2.set_xticks(r_vals)
    ax2.set_ylim(0, max(lats) * 1.3)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "routing_ablation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved ablation figure to {plot_path}")


if __name__ == "__main__":
    run_ablation()

