"""Dataset loading, data augmentation and feature extraction for DCapsNet SER.

Each audio file becomes a single 217-dimensional feature vector:
    MFCC(60) + Chroma(12) + Mel(128) + ZCR(1) + Tonnetz(6) + Kurtosis(1)
    + Skewness(1) + RMS(1) + Spectral contrast(7) = 217
Augmentation (Gaussian noise, speed perturbation, time shift) produces 3 extra
feature vectors per file. They are kept separate so that train.py can use them
for the training folds only (no leakage into validation/test).
"""
import glob
import os

import librosa
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew

SR = 22050
N_FEATURES = 217
N_AUG = 3

RAVDESS_MAP = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
EMODB_MAP = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
             "F": "happiness", "T": "sadness", "N": "neutral"}
SAVEE_MAP = {"a": "anger", "d": "disgust", "f": "fear", "h": "happiness",
             "n": "neutral", "sa": "sadness", "su": "surprise"}


def list_files(dataset, data_dir):
    """Return [(wav_path, emotion_label)] parsed from the dataset's file names."""
    pairs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)):
        stem = os.path.splitext(os.path.basename(path))[0]
        label = None
        if dataset == "ravdess":            # 03-01-05-01-02-01-12.wav
            parts = stem.split("-")
            if len(parts) == 7 and parts[1] == "01":      # speech only (not song)
                label = RAVDESS_MAP.get(parts[2])
        elif dataset == "emodb":            # 03a01Fa.wav -> letter at index 5
            if len(stem) >= 6:
                label = EMODB_MAP.get(stem[5].upper())
        elif dataset == "savee":            # DC_sa01.wav -> 'sa'
            code = stem.split("_")[-1]
            label = SAVEE_MAP.get("".join(c for c in code if c.isalpha()).lower())
        if label:
            pairs.append((path, label))
    return pairs


# ---------------------------------------------------------------- augmentation
def add_noise(y, rng, factor=0.001):
    return (y + factor * rng.standard_normal(len(y))).astype(np.float32)


def speed_perturb(y, rng):
    rate = rng.uniform(0.9, 1.1)
    z = librosa.effects.time_stretch(y, rate=rate)
    return librosa.util.fix_length(z, size=len(y))      # truncate / zero-pad


def time_shift(y, rng, sr=SR):
    max_shift = min(int(0.10 * sr), len(y) - 1)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    out = np.zeros_like(y)
    if shift >= 0:
        out[shift:] = y[:len(y) - shift]
    else:
        out[:shift] = y[-shift:]
    return out


# ------------------------------------------------------------ feature vector
def extract_features(y, sr=SR):
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 2048:
        y = librosa.util.fix_length(y, size=2048)
    kw = dict(n_fft=2048, hop_length=512)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60, **kw).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, **kw).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, **kw).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512).mean()
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).mean()
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).mean(axis=1)
    # fmin=100 (paper: 200) because 200*2^6 Hz exceeds Nyquist at 22.05 kHz
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=100.0, **kw).mean(axis=1)
    feat = np.concatenate([mfcc, chroma, mel, [zcr], tonnetz,
                           [kurtosis(y)], [skew(y)], [rms], contrast])
    assert feat.shape[0] == N_FEATURES, feat.shape
    return np.nan_to_num(feat).astype(np.float32)


def _process(path, seed):
    rng = np.random.default_rng(seed)
    y, _ = librosa.load(path, sr=SR)
    orig = extract_features(y)
    augs = [extract_features(add_noise(y, rng)),
            extract_features(speed_perturb(y, rng)),
            extract_features(time_shift(y, rng))]
    return orig, np.stack(augs)


def build_feature_set(dataset, data_dir, cache=None, n_jobs=-1):
    """Returns X (N,217), X_aug (N,3,217), y (N,) ints, classes (list of names)."""
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["X_aug"], d["y"], list(d["classes"])
    pairs = list_files(dataset, data_dir)
    if not pairs:
        raise SystemExit(f"No usable .wav files found for '{dataset}' under {data_dir}")
    classes = sorted({lab for _, lab in pairs})
    print(f"Extracting features from {len(pairs)} files ({len(classes)} classes)...")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process)(p, i) for i, (p, _) in enumerate(pairs))
    X = np.stack([o[0] for o in out])
    X_aug = np.stack([o[1] for o in out])
    y = np.array([classes.index(lab) for _, lab in pairs])
    if cache:
        np.savez_compressed(cache, X=X, X_aug=X_aug, y=y, classes=np.array(classes))
    return X, X_aug, y, classes
