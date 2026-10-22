"""Predict the emotion of a single .wav file with a trained DCapsNet.

    python predict.py --dataset ravdess --audio some_clip.wav
"""
import argparse
import json
import os
import pickle

import librosa
import numpy as np
import keras

import model as _model  # noqa: F401  (registers custom layers)
from features import SR, extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ravdess", "emodb", "savee"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    p = lambda s: os.path.join(args.out_dir, f"{args.dataset}_{s}")
    net = keras.models.load_model(p("model.keras"), compile=False)
    scaler = pickle.load(open(p("scaler.pkl"), "rb"))
    classes = json.load(open(p("classes.json")))

    y, _ = librosa.load(args.audio, sr=SR)
    feat = scaler.transform(extract_features(y)[None, :])
    scores = net.predict(feat, verbose=0)[0]
    print(f"Predicted emotion: {classes[int(np.argmax(scores))]}")
    for c, s in sorted(zip(classes, scores), key=lambda t: -t[1]):
        print(f"  {c:<10s} {s:.3f}")


if __name__ == "__main__":
    main()
