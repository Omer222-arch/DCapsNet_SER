"""Train / evaluate DCapsNet with stratified k-fold cross-validation.

Example:
    python train.py --dataset ravdess --data_dir data/RAVDESS
    python train.py --dataset emodb   --data_dir data/EMODB --folds 10
Augmented samples are used for the training part of each fold only.
After CV, a final model is trained on all data and saved for predict.py.
"""
import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import keras

from features import build_feature_set
from model import build_dcapsnet, margin_loss


def train_one(X, X_aug, y, tr_idx, n_classes, args):
    """Fit a model on tr_idx (+ its augmented copies). Returns (model, scaler)."""
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=42)
    scaler = StandardScaler().fit(X[tr_idx])
    n_aug = X_aug.shape[1]
    Xt = np.concatenate([X[tr_idx], X_aug[tr_idx].reshape(-1, X.shape[1])])
    yt = np.concatenate([y[tr_idx], np.repeat(y[tr_idx], n_aug)])
    Xv, yv = X[val_idx], y[val_idx]                      # validation = original samples only
    Xt, Xv = scaler.transform(Xt), scaler.transform(Xv)

    model = build_dcapsnet(n_classes, X.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=margin_loss,
                  metrics=[keras.metrics.CategoricalAccuracy(name="acc")])
    model.fit(Xt, keras.utils.to_categorical(yt, n_classes),
              validation_data=(Xv, keras.utils.to_categorical(yv, n_classes)),
              epochs=args.epochs, batch_size=args.batch_size, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                       restore_best_weights=True)])
    return model, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ravdess", "emodb", "savee"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--skip_cv", action="store_true", help="only train the final model")
    ap.add_argument("--skip_final", action="store_true", help="only run cross-validation")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    X, X_aug, y, classes = build_feature_set(
        args.dataset, args.data_dir, cache=os.path.join(args.out_dir, f"{args.dataset}_features.npz"))
    n_classes = len(classes)
    print(f"{len(y)} samples, classes: {classes}")

    if not args.skip_cv:
        folds = min(args.folds, int(np.bincount(y).min()))
        skf = StratifiedKFold(folds, shuffle=True, random_state=42)
        y_true, y_pred, accs = [], [], []
        for k, (tr, te) in enumerate(skf.split(X, y), 1):
            print(f"\n=== Fold {k}/{folds} ===")
            model, scaler = train_one(X, X_aug, y, tr, n_classes, args)
            pred = model.predict(scaler.transform(X[te]), verbose=0).argmax(1)
            acc = float((pred == y[te]).mean())
            print(f"Fold {k} test accuracy: {acc:.4f}")
            accs.append(acc); y_true += list(y[te]); y_pred += list(pred)
        report = classification_report(y_true, y_pred, target_names=classes, digits=4)
        cm = confusion_matrix(y_true, y_pred)
        print("\n" + report); print("Confusion matrix (rows=true, cols=pred):\n", cm)
        print(f"Mean fold accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        with open(os.path.join(args.out_dir, f"{args.dataset}_cv_results.json"), "w") as f:
            json.dump({"fold_acc": accs, "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
                       "classes": classes, "confusion_matrix": cm.tolist(), "report": report}, f, indent=2)

    if not args.skip_final:
        print("\n=== Training final model on all data ===")
        model, scaler = train_one(X, X_aug, y, np.arange(len(y)), n_classes, args)
        model.save(os.path.join(args.out_dir, f"{args.dataset}_model.keras"))
        with open(os.path.join(args.out_dir, f"{args.dataset}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(args.out_dir, f"{args.dataset}_classes.json"), "w") as f:
            json.dump(classes, f)
        print("Saved model, scaler and class list to", args.out_dir)


if __name__ == "__main__":
    main()
