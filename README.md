# DCapsNet – Speech Emotion Recognition (mini project)

Simple implementation of **DCapsNet** from *"DCapsNet: a robust multi-feature deep capsule network for
speech emotion recognition"* (Jahangir et al., Automatika 67(1), 2026), with a command-line interface
and a simple Streamlit app.

## Files
| File | What it does |
|---|---|
| `features.py` | Reads dataset labels from file names, augments audio (Gaussian noise 0.001, speed 0.9-1.1, time shift <=10%), extracts the 217-D feature vector |
| `model.py` | DCapsNet: 4x Conv1D(+BN+ELU+Dropout 0.1) -> primary capsules (736x8) -> digit capsules (dynamic routing, 3 iters) -> capsule lengths; margin loss (0.9 / 0.1 / 0.5) |
| `train.py` | Stratified 10-fold CV (Adam 1e-4, batch 8, <=100 epochs, early stopping patience 10), then trains and saves a final model |
| `predict.py` | Predicts the emotion of one `.wav` file |
| `streamlit_app.py` | Uploads a `.wav` file and displays the predicted emotion and class scores |

## Setup
```
pip install -r requirements.txt
```

## Datasets (download separately)
* RAVDESS (speech only, 1440 files): `Audio_Speech_Actors_01-24` -> `--dataset ravdess`
* EMO-DB (535 files) -> `--dataset emodb`
* SAVEE (480 files) -> `--dataset savee`

Labels are parsed from file names, so keep the original names. Subfolders are fine.

## Run
```
python train.py --dataset ravdess --data_dir path/to/RAVDESS
python train.py --dataset emodb   --data_dir path/to/EMODB
python train.py --dataset savee   --data_dir path/to/SAVEE
python predict.py --dataset ravdess --audio some_clip.wav
streamlit run streamlit_app.py
```
Useful flags: `--folds 5`, `--epochs 30` (faster runs), `--skip_cv`, `--skip_final`.
Features are cached in `outputs/<dataset>_features.npz`, so the second run skips extraction.
Results (per-fold accuracy, classification report, confusion matrix) go to `outputs/<dataset>_cv_results.json`.

## Verified
`python model.py` reproduces the paper's Table 5 exactly (e.g. conv4 229,632 / primary_caps 163,968 /
digit_caps 753,664 params; ~1.225 M total for 8 classes). The full pipeline was smoke-tested on synthetic audio.
Real-dataset accuracy has not been measured here, so expect your numbers to differ from the paper's 93/90/94%.

## Where this differs from the paper (mention in your report)
1. **Primary capsule padding**: the text says "same", but Table 5 (736 capsules, 163,968 params) only works with
   `valid` padding, so `valid` is used.
2. **Spectral contrast `fmin`**: paper lists 200 Hz with 6 bands, which exceeds Nyquist at 22,050 Hz and errors in
   librosa. `fmin=100` is used to keep the 7 contrast values (217 features total).
3. **Segmentation (128-sample windows)** is not used. The paper's Table 5 input is one 217-D vector per file,
   so features are averaged over time per file.
4. **Digit capsules** = number of classes in the dataset (7 for EMO-DB/SAVEE, 8 for RAVDESS); Table 5 shows 8.
5. **Validation**: 10% of each training fold (original samples only) is held out for early stopping, so the
   test fold is never used for model selection. Augmented copies exist only for training samples.
6. Feature scaling (StandardScaler) is fit on the training portion only; the paper does not specify this.
7. The paper itself notes its CV is not speaker-independent, so results may overestimate unseen-speaker performance.
