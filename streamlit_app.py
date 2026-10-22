"""Streamlit interface for single-file DCapsNet speech emotion prediction."""
import io
import json
import pickle
from pathlib import Path

import keras
import librosa
import numpy as np
import soundfile as sf
import streamlit as st

import model as _model  # noqa: F401  (registers custom layers)
from features import SR, extract_features


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASETS = ("ravdess", "emodb", "savee")

st.set_page_config(page_title="DCapsNet emotion recognition", page_icon=":material/graphic_eq:")
st.title("Speech emotion recognition")
st.caption("Upload a speech recording and classify its emotion with a trained DCapsNet model.")

available_datasets = [
    dataset
    for dataset in DATASETS
    if all(
        (OUTPUT_DIR / f"{dataset}_{artifact}").is_file()
        for artifact in ("model.keras", "scaler.pkl", "classes.json")
    )
]

if not available_datasets:
    st.error(
        "No complete trained model was found in the outputs folder. "
        "Train a model first with `python train.py --dataset ravdess --data_dir <dataset path>`."
    )
    st.stop()


@st.cache_resource(max_entries=3)
def load_artifacts(dataset: str):
    model = keras.models.load_model(OUTPUT_DIR / f"{dataset}_model.keras", compile=False)
    with (OUTPUT_DIR / f"{dataset}_scaler.pkl").open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with (OUTPUT_DIR / f"{dataset}_classes.json").open(encoding="utf-8") as classes_file:
        classes = json.load(classes_file)
    return model, scaler, classes


with st.form("prediction_form", border=False):
    dataset = st.selectbox("Trained dataset", available_datasets)
    uploaded_audio = st.file_uploader("Choose a WAV audio file", type=["wav"])
    submitted = st.form_submit_button(
        "Predict emotion", type="primary", icon=":material/graphic_eq:"
    )

if uploaded_audio is not None:
    st.audio(uploaded_audio.getvalue(), format="audio/wav", alt="Uploaded speech recording")

if submitted:
    if uploaded_audio is None:
        st.warning("Upload a WAV file before running a prediction.")
        st.stop()

    audio_bytes = uploaded_audio.getvalue()
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    except (sf.LibsndfileError, ValueError) as exc:
        st.error(f"Could not read this WAV file: {exc}")
        st.stop()

    try:
        model, scaler, classes = load_artifacts(dataset)
    except FileNotFoundError as exc:
        st.error(f"A required model artifact is missing: {exc.filename}")
        st.stop()

    features = extract_features(audio).reshape(1, -1)
    scaled_features = scaler.transform(features)
    scores = np.asarray(model.predict(scaled_features, verbose=0)[0])

    if scores.size != len(classes):
        st.error("The model output does not match the saved class list.")
        st.stop()

    ranked = sorted(zip(classes, scores.tolist()), key=lambda item: item[1], reverse=True)
    predicted_emotion, top_score = ranked[0]

    st.subheader("Prediction")
    with st.container(border=True):
        st.metric("Predicted emotion", predicted_emotion)
        st.caption(f"Top capsule score: {top_score:.3f}")

    st.subheader("Class scores")
    st.bar_chart(
        {emotion: score for emotion, score in ranked},
        horizontal=True,
        sort=False,
        alt="DCapsNet capsule scores by emotion",
    )
    st.caption("Capsule scores are model outputs, not calibrated probabilities.")
