"""Unified Benchmarking Suite for DCapsNet and Baselines on RAVDESS.

Milestone 1: Split Generation (Protocol A: 10-fold Stratified, Protocol B: 6-fold GroupKFold by Actor)
Milestone 2: Paper Replication - DCapsNet on Protocol A (10-fold Stratified CV)
Milestone 3: Baseline Models (1D-CNN, LSTM, BiLSTM) on Protocol A
Milestone 4: Speaker-Independent Evaluation (Protocol B)
"""
import argparse
import json
import os
import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import model as _model
from model import DigitCaps, CapsuleLength, build_dcapsnet, margin_loss
import features

# -----------------------------------------------------------------------------
# Model Definitions
# -----------------------------------------------------------------------------
def build_1d_cnn(n_classes, n_features=217):
    """4-layer 1D CNN baseline per paper specification (Section 5)."""
    inp = keras.Input(shape=(n_features,), name="input")
    x = layers.Reshape((n_features, 1))(inp)
    for i, (f, k) in enumerate([(32, 9), (64, 9), (128, 7), (256, 7)], 1):
        x = layers.Conv1D(f, k, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling1D(2, name=f"pool{i}")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="Baseline_1D_CNN")


def build_lstm(n_classes, n_features=217):
    """4-layer LSTM baseline per paper specification (Section 5)."""
    inp = keras.Input(shape=(n_features,), name="input")
    x = layers.Reshape((n_features, 1))(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Baseline_LSTM")


def build_bilstm(n_classes, n_features=217):
    """4-layer BiLSTM baseline per paper specification (Section 5)."""
    inp = keras.Input(shape=(n_features,), name="input")
    x = layers.Reshape((n_features, 1))(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Baseline_BiLSTM")


# -----------------------------------------------------------------------------
# Split Preparation (Milestone 1)
# -----------------------------------------------------------------------------
def get_or_create_splits(data_dir, y, out_dir="outputs"):
    splits_file = os.path.join(out_dir, "splits_ravdess.json")
    if os.path.exists(splits_file):
        with open(splits_file, "r") as f:
            return json.load(f)

    # Extract Actor IDs from file names
    pairs = features.list_files("ravdess", data_dir)
    actors = []
    for path, _ in pairs:
        stem = os.path.splitext(os.path.basename(path))[0]
        actor_id = int(stem.split("-")[-1])
        actors.append(actor_id)
    actors = np.array(actors)

    # Protocol A: 10-Fold Stratified K-Fold (Sample-Independent)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    protocol_a_splits = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        protocol_a_splits.append({"train": tr.tolist(), "test": te.tolist()})

    # Protocol B: 6-Fold GroupKFold by Actor (4 actors per fold, Speaker-Independent)
    gkf = GroupKFold(n_splits=6)
    protocol_b_splits = []
    for tr, te in gkf.split(np.zeros(len(y)), y, groups=actors):
        test_actors = sorted(list(set(actors[te])))
        protocol_b_splits.append({
            "train": tr.tolist(),
            "test": te.tolist(),
            "test_actors": [int(a) for a in test_actors]
        })

    splits_data = {
        "protocol_a_10fold": protocol_a_splits,
        "protocol_b_6fold_speaker_independent": protocol_b_splits,
        "actors": actors.tolist()
    }
    with open(splits_file, "w") as f:
        json.dump(splits_data, f, indent=2)
    print(f"Generated and saved splits to {splits_file}")
    return splits_data


# -----------------------------------------------------------------------------
# Training & Evaluation Engine
# -----------------------------------------------------------------------------
def run_fold(model_type, X, X_aug, y, tr_idx, te_idx, n_classes, params):
    # Leakage-safe holdout validation split (10% of train)
    tr_sub, val_sub = train_test_split(tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=42)
    scaler = StandardScaler().fit(X[tr_sub])

    # Augment training partition only
    n_aug = X_aug.shape[1]
    Xt = np.concatenate([X[tr_sub], X_aug[tr_sub].reshape(-1, X.shape[1])])
    yt = np.concatenate([y[tr_sub], np.repeat(y[tr_sub], n_aug)])
    Xv, yv = X[val_sub], y[val_sub]

    Xt, Xv = scaler.transform(Xt), scaler.transform(Xv)
    X_test_scaled = scaler.transform(X[te_idx])

    # Build model
    if model_type == "dcapsnet":
        net = build_dcapsnet(n_classes, X.shape[1], dropout=params.get("dropout", 0.1))
        loss_fn = margin_loss
        metric_name = "acc"
    elif model_type == "1d_cnn":
        net = build_1d_cnn(n_classes, X.shape[1])
        loss_fn = "categorical_crossentropy"
        metric_name = "acc"
    elif model_type == "lstm":
        net = build_lstm(n_classes, X.shape[1])
        loss_fn = "categorical_crossentropy"
        metric_name = "acc"
    elif model_type == "bilstm":
        net = build_bilstm(n_classes, X.shape[1])
        loss_fn = "categorical_crossentropy"
        metric_name = "acc"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    lr = params.get("lr", 1e-4)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    net.compile(optimizer=optimizer, loss=loss_fn,
                metrics=[keras.metrics.CategoricalAccuracy(name=metric_name)])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params.get("patience", 10),
            restore_best_weights=True
        )
    ]
    if params.get("use_reduce_lr", False):
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0
            )
        )

    epochs = params.get("epochs", 100)
    batch_size = params.get("batch_size", 8)

    t0 = time.time()
    net.fit(
        Xt, keras.utils.to_categorical(yt, n_classes),
        validation_data=(Xv, keras.utils.to_categorical(yv, n_classes)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    fit_time = time.time() - t0

    # Inference benchmark
    t_inf0 = time.time()
    raw_preds = net.predict(X_test_scaled, verbose=0)
    inf_time_ms = ((time.time() - t_inf0) / len(te_idx)) * 1000.0

    preds = raw_preds.argmax(axis=1)
    acc = float(accuracy_score(y[te_idx], preds))
    return preds, acc, fit_time, inf_time_ms


def evaluate_protocol(model_name, splits, X, X_aug, y, classes, params, protocol_name="Protocol A (10-Fold CV)"):
    print(f"\n{'='*70}")
    print(f" EVALUATING: {model_name.upper()} on {protocol_name}")
    print(f" Hyperparameters: {params}")
    print(f"{'='*70}")

    n_classes = len(classes)
    accs, fit_times, inf_latencies = [], [], []
    y_true_all, y_pred_all = [], []

    for fold_i, split in enumerate(splits, 1):
        tr = np.array(split["train"])
        te = np.array(split["test"])
        preds, acc, fit_time, inf_lat = run_fold(model_name, X, X_aug, y, tr, te, n_classes, params)
        accs.append(acc)
        fit_times.append(fit_time)
        inf_latencies.append(inf_lat)
        y_true_all.extend(y[te])
        y_pred_all.extend(preds)

        extra_info = ""
        if "test_actors" in split:
            extra_info = f" (Held-out Actors: {split['test_actors']})"
        print(f"  Fold {fold_i:2d}/{len(splits)} | Test Acc: {acc*100:.2f}% | Train Time: {fit_time:.1f}s | Latency: {inf_lat:.2f}ms{extra_info}")

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro"))
    weighted_f1 = float(f1_score(y_true_all, y_pred_all, average="weighted"))
    rep = classification_report(y_true_all, y_pred_all, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true_all, y_pred_all).tolist()

    # Calculate RMSE per Eq. 23
    rmse = float(np.sqrt(np.mean((np.array(y_true_all) != np.array(y_pred_all))**2)))

    print(f"\n--> {model_name} RESULTS:")
    print(f"    Mean Accuracy : {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"    Macro F1-Score: {macro_f1:.4f}")
    print(f"    Weighted F1   : {weighted_f1:.4f}")
    print(f"    Mean Latency  : {np.mean(inf_latencies):.2f} ms/sample")
    print(f"    Total Time    : {np.sum(fit_times):.1f} s")

    return {
        "model": model_name,
        "protocol": protocol_name,
        "params": params,
        "fold_acc": accs,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rmse": rmse,
        "mean_inf_latency_ms": float(np.mean(inf_latencies)),
        "total_train_time_sec": float(np.sum(fit_times)),
        "class_report": rep,
        "confusion_matrix": cm
    }


def main():
    parser = argparse.ArgumentParser(description="Run DCapsNet & Baselines Benchmark on RAVDESS")
    parser.add_argument("--data_dir", default="data/audio_speech_actors_01-24")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--mode", default="dcapsnet_protocol_a",
                        choices=["dcapsnet_protocol_a", "baselines_protocol_a", "speaker_independent", "all"],
                        help="Benchmark milestone to execute")
    parser.add_argument("--baseline_models", nargs="+", default=["1d_cnn", "bilstm"],
                        help="Baseline models to evaluate (1d_cnn, lstm, bilstm)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    cache_file = os.path.join(args.out_dir, "ravdess_features.npz")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"{cache_file} not found. Run features.py first.")

    data = np.load(cache_file, allow_pickle=True)
    X, X_aug, y, classes = data["X"], data["X_aug"], data["y"], list(data["classes"])
    print(f"Loaded {len(y)} RAVDESS samples with 8 emotion classes.")

    # Milestone 1: Splits
    splits_data = get_or_create_splits(args.data_dir, y, args.out_dir)

    default_params = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "dropout": 0.1,
        "use_reduce_lr": True
    }

    # Milestone 2: DCapsNet on Protocol A (10-Fold CV)
    if args.mode in ["dcapsnet_protocol_a", "all"]:
        dcapsnet_res = evaluate_protocol(
            "dcapsnet", splits_data["protocol_a_10fold"], X, X_aug, y, classes, default_params,
            protocol_name="Protocol A (10-Fold Stratified CV)"
        )
        res_file = os.path.join(args.out_dir, "bench_dcapsnet_protocol_a.json")
        with open(res_file, "w") as f:
            json.dump(dcapsnet_res, f, indent=2)
        print(f"Saved DCapsNet Protocol A results to {res_file}")

    # Milestone 3: Baselines on Protocol A
    if args.mode in ["baselines_protocol_a", "all"]:
        res_file = os.path.join(args.out_dir, "bench_baselines_protocol_a.json")
        baselines_results = []
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    baselines_results = json.load(f)
            except Exception:
                baselines_results = []

        for b_name in args.baseline_models:
            b_res = evaluate_protocol(
                b_name, splits_data["protocol_a_10fold"], X, X_aug, y, classes, default_params,
                protocol_name="Protocol A (10-Fold Stratified CV)"
            )
            # Replace if already exists or append
            baselines_results = [r for r in baselines_results if r.get("model") != b_name]
            baselines_results.append(b_res)
            with open(res_file, "w") as f:
                json.dump(baselines_results, f, indent=2)
            print(f"Updated Baselines Protocol A results in {res_file}")

    # Milestone 4: Speaker-Independent on Protocol B (6-Fold GroupKFold)
    if args.mode in ["speaker_independent", "all"]:
        speaker_results = []
        for m_name in ["dcapsnet", "bilstm"]:
            s_res = evaluate_protocol(
                m_name, splits_data["protocol_b_6fold_speaker_independent"], X, X_aug, y, classes, default_params,
                protocol_name="Protocol B (6-Fold Speaker-Independent GroupKFold)"
            )
            speaker_results.append(s_res)
        res_file = os.path.join(args.out_dir, "bench_speaker_independent.json")
        with open(res_file, "w") as f:
            json.dump(speaker_results, f, indent=2)
        print(f"Saved Speaker-Independent results to {res_file}")


if __name__ == "__main__":
    main()

