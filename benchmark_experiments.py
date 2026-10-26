"""Benchmark 3 different parameter configurations against the baseline DCapsNet on RAVDESS.

Configurations:
1. Config 1 (Batch-16 + Dynamic LR): Batch size 16, LR 3e-4 with ReduceLROnPlateau, EarlyStopping patience 8
2. Config 2 (Class-Balanced + Soft Margin): Batch size 16, LR 3e-4, balanced class weights, Soft Margin (0.80 / 0.20)
3. Config 3 (Regularized + 2-Step Routing): Batch size 32, LR 4e-4, Dropout 0.25, 2 routing iterations, Margin (0.85 / 0.15)
"""
import json
import os
import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import model as _model  # registers DigitCaps and CapsuleLength
from model import DigitCaps, CapsuleLength

PKG = "dcapsnet"


def make_margin_loss(m_pos=0.9, m_neg=0.1, lam=0.5):
    def loss_fn(y_true, y_pred):
        left = y_true * tf.square(tf.maximum(0.0, m_pos - y_pred))
        right = lam * (1.0 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_neg))
        return tf.reduce_mean(tf.reduce_sum(left + right, axis=-1))
    return loss_fn


def build_custom_dcapsnet(n_classes, n_features=217, dropout=0.1, routings=3):
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
    return keras.Model(inp, out, name="DCapsNet_Bench")


def evaluate_config(cfg_name, cfg_params, X, X_aug, y, classes, folds=3):
    print(f"\n=======================================================")
    print(f" Running Benchmark: {cfg_name}")
    print(f" Params: {cfg_params}")
    print(f"=======================================================")

    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    n_classes = len(classes)
    y_true_all, y_pred_all, accs = [], [], []
    start_time = time.time()

    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        tr_idx, val_idx = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=42)
        scaler = StandardScaler().fit(X[tr_idx])

        n_aug = X_aug.shape[1]
        Xt = np.concatenate([X[tr_idx], X_aug[tr_idx].reshape(-1, X.shape[1])])
        yt = np.concatenate([y[tr_idx], np.repeat(y[tr_idx], n_aug)])
        Xv, yv = X[val_idx], y[val_idx]
        Xt, Xv = scaler.transform(Xt), scaler.transform(Xv)
        X_test_scaled = scaler.transform(X[te])

        # Class weights if specified
        cw_dict = None
        if cfg_params.get("use_class_weights", False):
            cw = compute_class_weight("balanced", classes=np.unique(yt), y=yt)
            cw_dict = dict(enumerate(cw))

        # Build model
        dropout = cfg_params.get("dropout", 0.1)
        routings = cfg_params.get("routings", 3)
        net = build_custom_dcapsnet(n_classes, X.shape[1], dropout=dropout, routings=routings)

        # Loss function
        loss_fn = make_margin_loss(
            m_pos=cfg_params.get("m_pos", 0.9),
            m_neg=cfg_params.get("m_neg", 0.1),
            lam=cfg_params.get("lam", 0.5)
        )

        lr = cfg_params.get("lr", 1e-4)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        net.compile(optimizer=optimizer, loss=loss_fn,
                    metrics=[keras.metrics.CategoricalAccuracy(name="acc")])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg_params.get("patience", 8),
                restore_best_weights=True
            )
        ]
        if cfg_params.get("use_reduce_lr", True):
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-5,
                    verbose=0
                )
            )

        batch_size = cfg_params.get("batch_size", 16)
        epochs = cfg_params.get("epochs", 40)

        net.fit(
            Xt, keras.utils.to_categorical(yt, n_classes),
            validation_data=(Xv, keras.utils.to_categorical(yv, n_classes)),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=0
        )

        preds = net.predict(X_test_scaled, verbose=0).argmax(axis=1)
        acc = float(accuracy_score(y[te], preds))
        accs.append(acc)
        y_true_all.extend(y[te])
        y_pred_all.extend(preds)
        print(f"  Fold {k}/{folds} Accuracy: {acc:.4f}")

    total_time = time.time() - start_time
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro"))
    weighted_f1 = float(f1_score(y_true_all, y_pred_all, average="weighted"))
    rep = classification_report(y_true_all, y_pred_all, target_names=classes, output_dict=True)

    print(f"  --> Mean Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%, Macro F1: {macro_f1:.4f}, Time: {total_time:.1f}s")

    return {
        "name": cfg_name,
        "params": cfg_params,
        "fold_acc": accs,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "time_seconds": round(total_time, 1),
        "class_f1": {c: round(rep[c]["f1-score"], 4) for c in classes}
    }


def main():
    tf.keras.utils.set_random_seed(42)
    cache_path = os.path.join("outputs", "ravdess_features.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found.")

    data = np.load(cache_path, allow_pickle=True)
    X, X_aug, y, classes = data["X"], data["X_aug"], data["y"], list(data["classes"])
    print(f"Loaded {len(y)} samples for RAVDESS with classes: {classes}")

    # Load baseline results from outputs/ravdess_cv_results.json
    baseline_path = os.path.join("outputs", "ravdess_cv_results.json")
    baseline_summary = None
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            base_data = json.load(f)
            baseline_summary = {
                "name": "Baseline (Current Executed)",
                "params": {
                    "batch_size": 8,
                    "lr": 1e-4,
                    "patience": 10,
                    "dropout": 0.1,
                    "routings": 3,
                    "m_pos": 0.9,
                    "m_neg": 0.1,
                    "use_class_weights": False,
                    "use_reduce_lr": False
                },
                "fold_acc": base_data.get("fold_acc", []),
                "mean_acc": base_data.get("mean_acc", 0.5410),
                "std_acc": base_data.get("std_acc", 0.0128),
                "macro_f1": 0.5282,
                "weighted_f1": 0.5355,
                "time_seconds": None,
                "class_f1": {
                    "angry": 0.6282, "calm": 0.6621, "disgust": 0.5169, "fear": 0.5330,
                    "happy": 0.3879, "neutral": 0.4181, "sad": 0.4930, "surprise": 0.5863
                }
            }

    # Define 3 distinct configurations
    configs = [
        (
            "Config 1: Batch-16 + Dynamic LR Decay",
            {
                "batch_size": 16,
                "lr": 3e-4,
                "epochs": 30,
                "patience": 8,
                "dropout": 0.1,
                "routings": 3,
                "m_pos": 0.9,
                "m_neg": 0.1,
                "use_class_weights": False,
                "use_reduce_lr": True
            }
        ),
        (
            "Config 2: Class-Balanced + Soft Margin (0.80/0.20)",
            {
                "batch_size": 16,
                "lr": 3e-4,
                "epochs": 30,
                "patience": 8,
                "dropout": 0.1,
                "routings": 3,
                "m_pos": 0.80,
                "m_neg": 0.20,
                "use_class_weights": True,
                "use_reduce_lr": True
            }
        ),
        (
            "Config 3: High Regularization (0.25 Dropout) + 2-Step Routing",
            {
                "batch_size": 32,
                "lr": 4e-4,
                "epochs": 35,
                "patience": 8,
                "dropout": 0.25,
                "routings": 2,
                "m_pos": 0.85,
                "m_neg": 0.15,
                "use_class_weights": False,
                "use_reduce_lr": True
            }
        )
    ]

    results = []
    if baseline_summary:
        results.append(baseline_summary)

    for cfg_name, cfg_params in configs:
        res = evaluate_config(cfg_name, cfg_params, X, X_aug, y, classes, folds=3)
        results.append(res)

    # Save benchmark comparison
    out_file = os.path.join("outputs", "benchmark_comparison.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n" + "=" * 70)
    print("                     FINAL BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Configuration':<35} | {'Mean Acc':<10} | {'Macro F1':<10} | {'Happy F1':<10} | {'Neutral F1':<10}")
    print("-" * 85)
    for r in results:
        name = r["name"][:35]
        acc_str = f"{r['mean_acc']*100:.2f}%"
        mf1_str = f"{r['macro_f1']:.4f}"
        hf1_str = f"{r['class_f1'].get('happy', 0):.4f}"
        nf1_str = f"{r['class_f1'].get('neutral', 0):.4f}"
        print(f"{name:<35} | {acc_str:<10} | {mf1_str:<10} | {hf1_str:<10} | {nf1_str:<10}")
    print("=" * 70)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()

