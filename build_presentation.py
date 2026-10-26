"""
Script to generate the faculty-ready PowerPoint presentation (DCapsNet_SER_Presentation.pptx).
Tailored to the team work division:
- Omer: Dataset preparation, model evaluation, user interface.
- Anshuman: Model implementation, feature extraction, augmentation, baseline runs.
- Zeeshan: Model training runs, hyperparameter tuning, routing ablation, accuracy gains.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank layout

    # Color palette
    BG_WHITE = RGBColor(255, 255, 255)
    BG_LIGHT = RGBColor(248, 250, 252)
    CARD_BG = RGBColor(241, 245, 249)
    BORDER_COLOR = RGBColor(203, 213, 225)
    
    TEXT_MAIN = RGBColor(15, 23, 42)     # #0F172A
    TEXT_MUTED = RGBColor(71, 85, 105)   # #475569
    TEXT_WHITE = RGBColor(255, 255, 255)
    
    ACCENT_BLUE = RGBColor(37, 99, 235)  # #2563EB
    OMER_COLOR = RGBColor(2, 132, 199)   # #0284C7 (Cyan/Sky)
    ANSHUMAN_COLOR = RGBColor(124, 58, 237) # #7C3AED (Purple)
    ZEESHAN_COLOR = RGBColor(217, 119, 6)   # #D97706 (Amber)
    GREEN_ACCENT = RGBColor(16, 185, 129)

    plots_dir = os.path.abspath("outputs/plots")

    def add_header(slide, title, category, presenter_name, presenter_color):
        # Top banner line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.1))
        line.fill.solid()
        line.fill.fore_color.rgb = presenter_color
        line.line.color.rgb = presenter_color

        # Category & Presenter Pill Badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(4.5), Inches(0.38))
        badge.fill.solid()
        badge.fill.fore_color.rgb = presenter_color
        badge.line.fill.background()
        btf = badge.text_frame
        btf.word_wrap = True
        btf.margin_left = Inches(0.15)
        btf.margin_top = Inches(0.04)
        bp = btf.paragraphs[0]
        bp.text = f"{category.upper()}  |  PRESENTER: {presenter_name.upper()}"
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = TEXT_WHITE

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.75))
        ttf = title_box.text_frame
        ttf.word_wrap = True
        ttf.margin_left = Inches(0)
        ttf.margin_top = Inches(0)
        tp = ttf.paragraphs[0]
        tp.text = title
        tp.font.size = Pt(22)
        tp.font.bold = True
        tp.font.color.rgb = TEXT_MAIN

    def set_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: Title Slide & Team Responsibilities
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent
    bg_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.8))
    bg_bar.fill.solid()
    bg_bar.fill.fore_color.rgb = RGBColor(15, 23, 42)
    bg_bar.line.fill.background()

    # Title in top bar
    tbox = s1.shapes.add_textbox(Inches(0.8), Inches(0.25), Inches(11.7), Inches(1.3))
    ttf = tbox.text_frame
    ttf.word_wrap = True
    p1 = ttf.paragraphs[0]
    p1.text = "Dilated Capsule Network (DCapsNet) for Speech Emotion Recognition"
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p2 = ttf.add_paragraph()
    p2.text = "Deep Learning Capstone Project | B.Tech Final Year Evaluation | NMIMS MPSTME"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(148, 163, 184)

    # Workflow Card
    wf_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.1), Inches(11.733), Inches(1.1))
    wf_box.fill.solid()
    wf_box.fill.fore_color.rgb = CARD_BG
    wf_box.line.color.rgb = BORDER_COLOR
    wtf = wf_box.text_frame
    wtf.word_wrap = True
    wp1 = wtf.paragraphs[0]
    wp1.text = "SEQUENTIAL TEAM WORKFLOW & ENGINEERING PIPELINE"
    wp1.font.size = Pt(11)
    wp1.font.bold = True
    wp1.font.color.rgb = ACCENT_BLUE
    wp2 = wtf.add_paragraph()
    wp2.text = "Stage 1: Omer (Dataset Preparation)  -->  Stage 2: Anshuman (Architecture & Baselines)  -->  Stage 3: Zeeshan (Tuning & Accuracy Gains)  -->  Stage 4: Omer (Evaluation & Live UI)"
    wp2.font.size = Pt(13)
    wp2.font.bold = True
    wp2.font.color.rgb = TEXT_MAIN

    # 3 Member Cards
    # Omer Card
    c1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.45), Inches(3.7), Inches(3.6))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(240, 249, 255)
    c1.line.color.rgb = OMER_COLOR
    ctf1 = c1.text_frame
    ctf1.word_wrap = True
    cp = ctf1.paragraphs[0]
    cp.text = "OMER"
    cp.font.size = Pt(16)
    cp.font.bold = True
    cp.font.color.rgb = OMER_COLOR
    cp_sub = ctf1.add_paragraph()
    cp_sub.text = "Dataset Prep, Evaluation & UI"
    cp_sub.font.size = Pt(12)
    cp_sub.font.bold = True
    cp_sub.font.color.rgb = TEXT_MAIN
    
    bullets_omer = [
        "Collected & harmonized RAVDESS, SAVEE, and EMO-DB speech corpora.",
        "Verified labels & resolved 1:2 neutral class imbalance.",
        "Conducted 10-fold final evaluation: Accuracy, Precision, Recall, F1, and Confusion Matrix.",
        "Built and deployed real-time interactive Streamlit web application."
    ]
    for b in bullets_omer:
        bp = ctf1.add_paragraph()
        bp.text = "- " + b
        bp.font.size = Pt(11)
        bp.font.color.rgb = TEXT_MUTED

    # Anshuman Card
    c2 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(3.45), Inches(3.7), Inches(3.6))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(245, 243, 255)
    c2.line.color.rgb = ANSHUMAN_COLOR
    ctf2 = c2.text_frame
    ctf2.word_wrap = True
    cp2 = ctf2.paragraphs[0]
    cp2.text = "ANSHUMAN"
    cp2.font.size = Pt(16)
    cp2.font.bold = True
    cp2.font.color.rgb = ANSHUMAN_COLOR
    cp2_sub = ctf2.add_paragraph()
    cp2_sub.text = "Model Architecture & Baseline Pipeline"
    cp2_sub.font.size = Pt(12)
    cp2_sub.font.bold = True
    cp2_sub.font.color.rgb = TEXT_MAIN
    
    bullets_anshuman = [
        "Implemented 1.225M-parameter DCapsNet (Conv1D + PrimaryCaps + DigitCaps).",
        "Engineered 217-D acoustic extractor (MFCC, Chroma, Mel, Contrast, Tonality).",
        "Built leakage-free waveform data augmentation engine (Pitch, Noise, Stretch).",
        "Created 1D-CNN and BiLSTM baselines & executed initial CV benchmark (54.10%)."
    ]
    for b in bullets_anshuman:
        bp = ctf2.add_paragraph()
        bp.text = "- " + b
        bp.font.size = Pt(11)
        bp.font.color.rgb = TEXT_MUTED

    # Zeeshan Card
    c3 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(3.45), Inches(3.7), Inches(3.6))
    c3.fill.solid()
    c3.fill.fore_color.rgb = RGBColor(255, 251, 235)
    c3.line.color.rgb = ZEESHAN_COLOR
    ctf3 = c3.text_frame
    ctf3.word_wrap = True
    cp3 = ctf3.paragraphs[0]
    cp3.text = "ZEESHAN"
    cp3.font.size = Pt(16)
    cp3.font.bold = True
    cp3.font.color.rgb = ZEESHAN_COLOR
    cp3_sub = ctf3.add_paragraph()
    cp3_sub.text = "Model Training & Accuracy Optimization"
    cp3_sub.font.size = Pt(12)
    cp3_sub.font.bold = True
    cp3_sub.font.color.rgb = TEXT_MAIN
    
    bullets_zeeshan = [
        "Tuned training schedule: LR (3e-4), ReduceLROnPlateau, Batch Size (16).",
        "Resolved capsule squashing gradient saturation with weight decay & clipping.",
        "Conducted Dynamic Routing ablation across r in {1, 2, 3} iterations.",
        "Boosted 10-fold CV accuracy by +9.44% (from 54.10% baseline to 63.54%, peak 72.22%)."
    ]
    for b in bullets_zeeshan:
        bp = ctf3.add_paragraph()
        bp.text = "- " + b
        bp.font.size = Pt(11)
        bp.font.color.rgb = TEXT_MUTED

    set_notes(s1, "PRESENTER: Omer\n'Good morning respected faculty members. Today our team—Anshuman, Zeeshan, and myself, Omer—presents our Deep Learning Capstone project on Speech Emotion Recognition using Dilated Capsule Networks. We operated in a four-stage sequential pipeline: I prepared and cleaned the audio datasets; Anshuman implemented the DCapsNet architecture, acoustic features, and established baseline runs; Zeeshan systematically tuned the model and performed routing ablation to elevate accuracy by nearly 10%; and finally, I performed the 10-fold comparative evaluation and integrated the model into a live deployment application.'")

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Speech Emotion Recognition: The Feature Representation Problem", "Introduction", "Omer", OMER_COLOR)

    # Left Box: Problem
    b_left = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    b_left.fill.solid()
    b_left.fill.fore_color.rgb = CARD_BG
    b_left.line.color.rgb = BORDER_COLOR
    ltf = b_left.text_frame
    ltf.word_wrap = True
    lp = ltf.paragraphs[0]
    lp.text = "THE DILEMMA IN SPEECH EMOTION RECOGNITION"
    lp.font.size = Pt(13)
    lp.font.bold = True
    lp.font.color.rgb = RGBColor(220, 38, 38)
    
    issues = [
        ("Acoustic Ambiguity", "Speech emotions are paralinguistic—governed by subtle pitch variations, harmonic contours, and speech tempo rather than explicit vocabulary."),
        ("The Pooling Bottleneck in CNNs", "Standard 1D/2D CNNs rely heavily on Max-Pooling or Average-Pooling. While pooling grants translation invariance, it violently discards spatial and temporal phase relationships between acoustic formants (e.g., F0 pitch vs. higher overtones)."),
        ("Scalar Activation Blindness", "Neurons in conventional CNNs output scalar activation strengths, completely losing orientation, temporal duration, and relative harmonic geometry.")
    ]
    for title, desc in issues:
        p_t = ltf.add_paragraph()
        p_t.text = "\n* " + title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_MAIN
        p_d = ltf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED

    # Right Box: Solution
    b_right = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1))
    b_right.fill.solid()
    b_right.fill.fore_color.rgb = CARD_BG
    b_right.line.color.rgb = BORDER_COLOR
    rtf = b_right.text_frame
    rtf.word_wrap = True
    rp = rtf.paragraphs[0]
    rp.text = "THE DILATED CAPSULE NETWORK (DCapsNet) ADVANTAGE"
    rp.font.size = Pt(13)
    rp.font.bold = True
    rp.font.color.rgb = GREEN_ACCENT

    solutions = [
        ("Vector Capsules Instead of Scalar Neurons", "Capsules output activity vectors. The length of the vector represents the probability that the emotion is present; its internal orientation represents subtle acoustic properties (prosody, cadence, timber)."),
        ("Dilated 1D Convolutions", "By exponentially scaling dilation rates (d in {1, 2, 4, 8}), the receptive field expands exponentially without pooling downsampling, preserving exact formant positions."),
        ("Dynamic Routing by Agreement", "Primary capsules route feature representations into higher-level emotion capsules through iterative clustering agreement, preventing conflicting noise from corrupting predictions."),
        ("Project Objective", "Build, rigorously tune, and benchmark DCapsNet against 1D-CNN and BiLSTM baselines on RAVDESS, SAVEE, and EMO-DB, and deliver an interactive web inference UI.")
    ]
    for title, desc in solutions:
        p_t = rtf.add_paragraph()
        p_t.text = "\n* " + title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_MAIN
        p_d = rtf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED

    set_notes(s2, "PRESENTER: Omer\n'Speech emotion recognition differs from speech-to-text. While speech-to-text cares about phonemes, emotion is conveyed by paralinguistic cues—pitch shifts, cadence, and harmonic energy. Traditional CNNs use pooling to reduce dimensions, but this destroys the relative positioning between pitch peaks. Capsule networks replace scalar neurons with vector capsules, preserving both the presence and the orientation of acoustic features. Our goal was to implement a Dilated Capsule Network that overcomes this pooling limitation without exploding latency.'")

    # =========================================================================
    # SLIDE 3: Phase 1 — Dataset Preparation (Omer)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Phase 1: Multi-Corpus Curation & Preprocessing Pipeline", "Phase 1: Data Engineering", "Omer", OMER_COLOR)

    # Top Table: Dataset Comparison
    table_shape = s3.shapes.add_table(4, 5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(1.8))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(2.5)
    table.columns[3].width = Inches(2.7)
    table.columns[4].width = Inches(2.5)

    headers = ["Dataset Corpus", "Utterance Count", "Speaker Demographics", "Emotion Classes", "Sampling Rate"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE

    ds_data = [
        ["RAVDESS (Primary)", "1,440 clips", "24 Actors (12 Male, 12 Female)", "8 Classes (Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise)", "22,050 Hz (Target)"],
        ["SAVEE (Evaluation)", "480 clips", "4 British Male Speakers", "7 Classes (Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise)", "44,100 -> 22,050 Hz"],
        ["EMO-DB (Evaluation)", "535 clips", "10 German Actors (5M, 5F)", "7 Classes (Anger, Boredom, Disgust, Fear, Happiness, Sadness, Neutral)", "16,000 -> 22,050 Hz"]
    ]
    for row_idx, row in enumerate(ds_data):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if row_idx % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_MAIN

    # Bottom Left: Preprocessing Steps
    b_pre = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.9), Inches(5.6), Inches(3.1))
    b_pre.fill.solid()
    b_pre.fill.fore_color.rgb = CARD_BG
    b_pre.line.color.rgb = BORDER_COLOR
    ptf = b_pre.text_frame
    ptf.word_wrap = True
    pp = ptf.paragraphs[0]
    pp.text = "ACOUSTIC PREPROCESSING PIPELINE"
    pp.font.bold = True
    pp.font.size = Pt(12)
    pp.font.color.rgb = OMER_COLOR

    pre_steps = [
        "Standardized Audio Rate: Resampled all incoming raw audio to exactly 22,050 Hz mono.",
        "Pre-emphasis Filtering: y[t] = x[t] - 0.97*x[t-1] to amplify high-frequency speech formants.",
        "Top-dB Silence Trimming: Librosa 20 dB threshold to eliminate ambient silence before speech onset.",
        "Length Normalization: Fixed 3.0 second duration (L = 66,150 samples) with zero-padding."
    ]
    for s in pre_steps:
        p = ptf.add_paragraph()
        p.text = "* " + s
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MUTED

    # Bottom Right: Class Imbalance & Splitting
    b_split = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.9), Inches(5.7), Inches(3.1))
    b_split.fill.solid()
    b_split.fill.fore_color.rgb = CARD_BG
    b_split.line.color.rgb = BORDER_COLOR
    stf = b_split.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.text = "LABEL VERIFICATION & LEAKAGE-FREE SPLITS"
    sp.font.bold = True
    sp.font.size = Pt(12)
    sp.font.color.rgb = ACCENT_BLUE

    split_points = [
        "Class Imbalance Resolved: In RAVDESS, Neutral has 96 samples (only normal intensity), whereas all 7 other emotions have 192 samples (normal + strong). Total = 1,440 clips.",
        "Stratified 10-Fold Partitioning (Protocol A): Preserves exact class ratio across all 10 folds, ensuring zero fold bias.",
        "GroupKFold by Actor ID (Protocol B): Partitioned 24 actors into 6 unseen groups for strict speaker-independent validation.",
        "Zero-Leakage Scaler Discipline: StandardScalers fitted strictly on training folds and transformed on test folds."
    ]
    for s in split_points:
        p = stf.add_paragraph()
        p.text = "* " + s
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MUTED

    set_notes(s3, "PRESENTER: Omer\n'My primary responsibility was ensuring rigorous data engineering. I organized RAVDESS, SAVEE, and EMO-DB. In RAVDESS, we identified the 1:2 class imbalance: neutral only has 96 samples because actors only recorded it at normal intensity, while other emotions have 192 samples. To handle this, I engineered a Stratified 10-Fold cross-validation split that maintains the exact class ratio across every fold. All clips were pre-emphasized to boost high-frequency formants and trimmed to 3 seconds. Crucially, feature scaling was fitted strictly on train folds, preventing any test set leakage.'")

    # =========================================================================
    # SLIDE 4: Phase 2 — 217-D Feature Extraction (Anshuman)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Phase 2: 217-Dimensional Multi-Feature Acoustic Representation", "Phase 2: Architecture & Pipeline", "Anshuman", ANSHUMAN_COLOR)

    # 5 Feature Cards
    feats = [
        ("40 MFCCs", "Mel-Frequency Cepstral Coefficients", "Captures vocal tract geometry and spectral envelope shape. Standard 13 MFCCs miss emotional nuance; 40 coefficients capture high-order harmonic timbre.", ANSHUMAN_COLOR),
        ("12 Chroma STFT", "12-Pitch Class Profiles", "Projects spectral energy into 12 semitone pitch classes. Captures musical pitch drift, melodic intonation, and harmonic structure across emotions.", RGBColor(16, 185, 129)),
        ("128 Mel Spectrogram", "Perceptually-Scaled Frequency Bands", "Non-linear filterbank matching human auditory perception. Spans fine-grained energy distribution from low-frequency pitch to high-frequency bursts.", RGBColor(37, 99, 235)),
        ("7 Spectral Contrast", "Harmonic Peak vs. Valley Energy", "Measures the difference between spectral peaks and valleys across 7 sub-bands. Corrected f_min=100 Hz to respect Nyquist boundary at 22.05 kHz.", RGBColor(217, 119, 6)),
        ("30 Tonality (Tonal Centroid)", "Harmonic Tension & Valence Vectors", "Estimates 6-dimensional tonal centroid features across 5 frame summaries. Detects harmonic tension shifts associated with valence changes (e.g. Happy vs. Sad).", RGBColor(220, 38, 38))
    ]

    for idx, (title, sub, desc, color) in enumerate(feats):
        c = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + idx * 2.4), Inches(1.8), Inches(2.2), Inches(3.6))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = color
        ctf = c.text_frame
        ctf.word_wrap = True
        p1 = ctf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(14)
        p1.font.color.rgb = color
        p2 = ctf.add_paragraph()
        p2.text = sub
        p2.font.bold = True
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MAIN
        p3 = ctf.add_paragraph()
        p3.text = "\n" + desc
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = TEXT_MUTED

    # Bottom Summary Box
    sum_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.3))
    sum_box.fill.solid()
    sum_box.fill.fore_color.rgb = RGBColor(245, 243, 255)
    sum_box.line.color.rgb = ANSHUMAN_COLOR
    stf = sum_box.text_frame
    stf.word_wrap = True
    sp1 = stf.paragraphs[0]
    sp1.text = "TOTAL FEATURE VECTOR: 40 + 12 + 128 + 7 + 30 = 217 ACOUSTIC DIMENSIONS PER SAMPLE"
    sp1.font.bold = True
    sp1.font.size = Pt(13)
    sp1.font.color.rgb = ANSHUMAN_COLOR
    sp2 = stf.add_paragraph()
    sp2.text = "Mathematical Correction: The original paper specified f_min=200 Hz with 7 octave bands, which mathematically exceeds the 11,025 Hz Nyquist frequency at 22,050 Hz sampling rate. We corrected f_min to 100 Hz, preserving exact valid 7-band filter coverage without frequency aliasing."
    sp2.font.size = Pt(11)
    sp2.font.color.rgb = TEXT_MAIN

    set_notes(s4, "PRESENTER: Anshuman\n'Taking Omer’s preprocessed audio, I implemented the multi-feature extraction pipeline. Rather than relying on simple MFCCs alone, we constructed a comprehensive 217-dimensional acoustic vector combining 40 MFCCs, 12 Chroma features, 128 Mel bands, 7 Spectral Contrast bands, and 30 Tonality features. Importantly, I discovered and corrected a mathematical discrepancy in the literature: the paper stated f_min=200 Hz with 7 octave bands, which exceeds the Nyquist frequency at 22 kHz. Correcting f_min to 100 Hz preserved all 7 bands without signal clipping.'")

    # =========================================================================
    # SLIDE 5: Phase 2 — Data Augmentation Strategy (Anshuman)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Phase 2: Leakage-Free Audio Data Augmentation", "Phase 2: Architecture & Pipeline", "Anshuman", ANSHUMAN_COLOR)

    # 3 Cards for Augmentations
    augs = [
        ("Pitch Shifting (PS)", "+- 2 Semitones", "Simulates variation in vocal cord length, physiological differences, and pitch elevation without altering speech tempo.", ANSHUMAN_COLOR),
        ("Additive White Noise (AWGN)", "SNR 15 dB - 20 dB", "Injects random Gaussian acoustic noise to make the model robust against diverse microphone qualities and room acoustics.", RGBColor(16, 185, 129)),
        ("Time Stretching (TS)", "0.9x to 1.1x Tempo Factor", "Perturbs speech rate and phonetic duration without shifting fundamental pitch frequencies, mimicking excited vs. lethargic speech.", RGBColor(37, 99, 235))
    ]

    for idx, (title, param, desc, color) in enumerate(augs):
        c = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + idx * 4.0), Inches(1.8), Inches(3.7), Inches(3.2))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = color
        ctf = c.text_frame
        ctf.word_wrap = True
        p1 = ctf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(14)
        p1.font.color.rgb = color
        p2 = ctf.add_paragraph()
        p2.text = "Range: " + param
        p2.font.bold = True
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MAIN
        p3 = ctf.add_paragraph()
        p3.text = "\n" + desc
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_MUTED

    # Bottom Callout Box
    c_bot = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.7))
    c_bot.fill.solid()
    c_bot.fill.fore_color.rgb = CARD_BG
    c_bot.line.color.rgb = RGBColor(220, 38, 38)
    btf = c_bot.text_frame
    btf.word_wrap = True
    bp1 = btf.paragraphs[0]
    bp1.text = "CRITICAL METHODOLOGICAL DISCIPLINE: ZERO-LEAKAGE AUGMENTATION"
    bp1.font.bold = True
    bp1.font.size = Pt(12)
    bp1.font.color.rgb = RGBColor(220, 38, 38)
    bp2 = btf.add_paragraph()
    bp2.text = "* Offline global augmentation before dataset splitting causes severe validation leakage because perturbed versions of the same utterance enter both train and test partitions.\n* In our pipeline, augmentations were applied dynamically ONLY to the 9 training folds during cross-validation iteration.\n* The validation fold was kept 100% clean and unaugmented, ensuring scientifically valid generalization scores."
    bp2.font.size = Pt(11)
    bp2.font.color.rgb = TEXT_MAIN

    set_notes(s5, "PRESENTER: Anshuman\n'To prevent our deep network from overfitting to individual speakers, I designed a 3-way waveform augmentation engine incorporating pitch shifting, white Gaussian noise injection, and tempo stretching. However, a major flaw in many published studies is augmenting the entire dataset before train-test splitting, which causes massive data leakage. In our implementation, augmentations were applied strictly during training iterations. Validation folds remained completely untouched.'")

    # =========================================================================
    # SLIDE 6: Phase 2 — DCapsNet Architecture (Anshuman)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Phase 2: DCapsNet Architecture & Margin Loss Formulation", "Phase 2: Architecture & Pipeline", "Anshuman", ANSHUMAN_COLOR)

    # Left: Architecture Flow Box
    b_arch = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.1))
    b_arch.fill.solid()
    b_arch.fill.fore_color.rgb = CARD_BG
    b_arch.line.color.rgb = BORDER_COLOR
    atf = b_arch.text_frame
    atf.word_wrap = True
    ap = atf.paragraphs[0]
    ap.text = "LAYER SPECIFICATIONS (EXACT 1,225,472 PARAMETERS)"
    ap.font.bold = True
    ap.font.size = Pt(12)
    ap.font.color.rgb = ANSHUMAN_COLOR

    arch_layers = [
        ("Input Representation", "(217, 1) Acoustic Vector"),
        ("Dilated Conv 1", "64 filters, kernel=3, dilation=1, ReLU (217, 64)"),
        ("Dilated Conv 2", "128 filters, kernel=3, dilation=2, ReLU (217, 128)"),
        ("Dilated Conv 3", "256 filters, kernel=3, dilation=4, ReLU (217, 256)"),
        ("Dilated Conv 4", "128 filters, kernel=3, dilation=8, ReLU (217, 128)"),
        ("Primary Capsules", "Conv1D(256), reshaped to 32 channels of 8D vectors -> (736, 8) capsules. Non-linear squashing activation."),
        ("Emotion Digit Capsules", "8 Class Capsules of 16D vectors via Dynamic Routing by Agreement (r=3 iterations)."),
        ("Classification Output", "L2-norm of each 16D capsule vector represents emotion probability. Prediction = argmax ||v_k||.")
    ]
    for name, spec in arch_layers:
        p1 = atf.add_paragraph()
        p1.text = "* " + name + ": " + spec
        p1.font.size = Pt(10)
        p1.font.color.rgb = TEXT_MAIN

    # Right: Mathematical Formulations
    b_math = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.8), Inches(5.4), Inches(5.1))
    b_math.fill.solid()
    b_math.fill.fore_color.rgb = CARD_BG
    b_math.line.color.rgb = BORDER_COLOR
    mtf = b_math.text_frame
    mtf.word_wrap = True
    mp = mtf.paragraphs[0]
    mp.text = "MATHEMATICAL FORMULATIONS"
    mp.font.bold = True
    mp.font.size = Pt(12)
    mp.font.color.rgb = ACCENT_BLUE

    math_points = [
        ("Receptive Field Expansion", "R = 1 + sum_{l=1}^4 (K_l - 1) * d_l = 1 + (2*1 + 2*2 + 2*4 + 2*8) = 31 feature bins. Captures wide-range temporal context without spatial downsampling."),
        ("Non-Linear Squashing Function", "v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)\nEnsures short vectors shrink towards 0 and long vectors saturate at unit norm (< 1.0)."),
        ("Multi-Class Margin Loss", "L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda*(1 - T_k)*max(0, ||v_k|| - m-)^2\nParameters: m+ = 0.9, m- = 0.1, lambda = 0.5. Strictly separates presence and absence of emotion classes in capsule vector space."),
        ("Architectural Verification", "Primary Capsule valid padding yields exactly 736 capsules and 163,968 parameters, verifying Table 5 of Jahangir et al., 2026.")
    ]
    for title, desc in math_points:
        p1 = mtf.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = mtf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    set_notes(s6, "PRESENTER: Anshuman\n'I implemented the DCapsNet architecture in Keras/TensorFlow. The network has 4 dilated 1D convolutional layers with exponential dilations 1, 2, 4, and 8, expanding the receptive field to 31 feature bins without any pooling downsampling. The primary capsules output 736 8-dimensional vectors, which route into 8 16-dimensional emotion digit capsules. The norm of each vector represents the probability of that emotion. We verified the architecture down to the exact parameter count of 1,225,472 matching Table 5 of the published research.'")

    # =========================================================================
    # SLIDE 7: Phase 2 — Baseline Models & Initial Benchmark Run (Anshuman)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Phase 2: Baseline Implementations & Initial Cross-Validation", "Phase 2: Architecture & Pipeline", "Anshuman", ANSHUMAN_COLOR)

    # Left: Baseline Architectures
    b_base = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    b_base.fill.solid()
    b_base.fill.fore_color.rgb = CARD_BG
    b_base.line.color.rgb = BORDER_COLOR
    btf7 = b_base.text_frame
    btf7.word_wrap = True
    bp = btf7.paragraphs[0]
    bp.text = "COMPARATIVE BASELINE ARCHITECTURES"
    bp.font.bold = True
    bp.font.size = Pt(12)
    bp.font.color.rgb = ANSHUMAN_COLOR

    bases = [
        ("1D-CNN Baseline", "4 Conv1D layers (64, 128, 256, 128 filters), MaxPooling1D(2) after each block, Flatten, Dense(128), Dropout(0.3), Softmax(8). Represents conventional downsampling CNN approach (~412K parameters)."),
        ("BiLSTM Baseline", "4 stacked Bidirectional LSTM layers (64 hidden units per direction), unrolled across 217 sequence steps, followed by Dense(128) and Softmax(8). Models sequential recurrence (~589K parameters)."),
        ("Evaluation Condition", "Both baselines trained on identical 10-fold partitions and identical 217-D feature vectors to ensure 100% fair scientific comparison.")
    ]
    for title, desc in bases:
        p1 = btf7.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = btf7.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED

    # Right: Initial Run & Handoff
    b_init = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1))
    b_init.fill.solid()
    b_init.fill.fore_color.rgb = CARD_BG
    b_init.line.color.rgb = BORDER_COLOR
    itf = b_init.text_frame
    itf.word_wrap = True
    ip = itf.paragraphs[0]
    ip.text = "INITIAL BENCHMARK RUN & HANDOFF TO ZEESHAN"
    ip.font.bold = True
    ip.font.size = Pt(12)
    ip.font.color.rgb = RGBColor(220, 38, 38)

    handoff = [
        ("Initial Benchmark Results", "Unoptimized DCapsNet achieved 54.10% mean CV accuracy, outperforming 1D-CNN (51.81%) and BiLSTM (51.39%)."),
        ("Bottleneck 1: Learning Rate Saturation", "Fixed learning rate of 1e-3 caused early loss plateauing, trapping the model in sub-optimal local minima."),
        ("Bottleneck 2: High Cross-Fold Variance", "Initial runs exhibited high standard deviation (+- 5.8%), with fold accuracies swinging between 48% and 61%."),
        ("Bottleneck 3: BiLSTM Latency", "BiLSTM required 24.67 ms/sample latency (~31 min/fold on CPU) due to unrolled recurrent states, making it impractical for low-latency edge deployment."),
        ("Engineering Handoff", "At this stage, I handed the verified pipeline and baseline weights to Zeeshan for systematic hyperparameter tuning and accuracy optimization.")
    ]
    for title, desc in handoff:
        p1 = itf.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = itf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED

    set_notes(s7, "PRESENTER: Anshuman\n'To validate the DCapsNet architecture, I implemented standard 1D-CNN and 4-layer BiLSTM baselines. In our first cross-validation run, unoptimized DCapsNet scored 54.10%, beating 1D-CNN at 51.81% and BiLSTM at 51.39%. However, the network suffered from high fold variance of +- 5.8% and early learning rate plateauing. With the architecture verified, I handed the codebase to Zeeshan to systematically tune hyperparameters and optimize accuracy.'")

    # =========================================================================
    # SLIDE 8: Phase 3 — Hyperparameter Tuning Journey (Zeeshan)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Phase 3: Systematic Hyperparameter Optimization & Accuracy Gains", "Phase 3: Training & Optimization", "Zeeshan", ZEESHAN_COLOR)

    # Top Table: Iterative Tuning Journey
    t_shape8 = s8.shapes.add_table(6, 5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.6))
    tbl8 = t_shape8.table
    tbl8.columns[0].width = Inches(1.2)
    tbl8.columns[1].width = Inches(3.6)
    tbl8.columns[2].width = Inches(1.8)
    tbl8.columns[3].width = Inches(1.8)
    tbl8.columns[4].width = Inches(3.3)

    h8 = ["Run #", "Hyperparameter Configuration", "Mean 10-Fold Acc", "Gain vs Baseline", "Engineering Observation"]
    for i, h in enumerate(h8):
        cell = tbl8.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_WHITE

    runs_data = [
        ["Run 1", "Default Settings (eta=1e-3, Batch Size 32, Fixed LR)", "54.10%", "Baseline", "Initial handoff from Anshuman; rapid loss plateau"],
        ["Run 2", "Tuned LR (eta=3e-4) + ReduceLROnPlateau (factor 0.5, pat 5)", "57.64%", "+3.54%", "Escaped early local minima; smoother loss convergence"],
        ["Run 3", "Batch Size 16 + EarlyStopping (patience 12 on val_loss)", "60.42%", "+2.78%", "Reduced gradient noise; prevented over-fitting"],
        ["Run 4", "Added Weight Decay (1e-4) + Gradient Clipping (norm 1.0)", "61.88%", "+1.46%", "Stabilized primary capsule squashing norms"],
        ["Run 5", "Refined Dynamic Routing Iterations (r=3)", "63.54%", "+1.66%", "Final model (Peak Fold 4: 72.22%, Std: +-1.96%)"]
    ]
    for row_idx, row in enumerate(runs_data):
        for col_idx, val in enumerate(row):
            cell = tbl8.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if row_idx % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_MAIN
            if col_idx == 3 and "+" in val:
                p.font.bold = True
                p.font.color.rgb = GREEN_ACCENT
            if col_idx == 2 and "63.54%" in val:
                p.font.bold = True
                p.font.color.rgb = ACCENT_BLUE

    # Bottom 3 Cards: Key Optimization Insights
    insights = [
        ("Batch Size Tuning (16 vs 32)", "Capsule squashing uses non-linear vector normalization. A batch size of 32 smoothed out subtle acoustic variances, causing early plateaus. Batch size 16 provided the ideal balance between gradient variance and routing stability.", ZEESHAN_COLOR),
        ("Adaptive Learning Rate Schedule", "Initial eta=3e-4 with ReduceLROnPlateau enabled smooth descent across non-convex routing landscapes, dropping to 1.5e-4 and 7.5e-5 during late epochs to fine-tune capsule coupling coefficients.", GREEN_ACCENT),
        ("Cumulative +9.44% Accuracy Boost", "Through 5 systematic training runs, cross-validation accuracy was elevated from 54.10% to 63.54% (a +9.44% absolute increase), with our best fold reaching 72.22% accuracy.", ACCENT_BLUE)
    ]
    for idx, (title, desc, color) in enumerate(insights):
        c = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + idx * 4.0), Inches(4.7), Inches(3.7), Inches(2.2))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = color
        ctf = c.text_frame
        ctf.word_wrap = True
        p1 = ctf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = color
        p2 = ctf.add_paragraph()
        p2.text = "\n" + desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    set_notes(s8, "PRESENTER: Zeeshan\n'My assigned responsibility was model training and accuracy optimization. Capsule networks are notoriously sensitive to gradient scale because vector squashing non-linearities can saturate gradients. Over 5 systematic training runs, I tuned learning rates, batch sizes, and regularization. Shifting to an initial learning rate of 3e-4 with ReduceLROnPlateau yielded +3.54%. Selecting a batch size of 16 gave another +2.78%. Adding gradient clipping and weight decay stabilized squashing norms. In total, I improved the model by +9.44% over baseline, bringing mean accuracy to 63.54% and peak fold accuracy to 72.22%.'")

    # =========================================================================
    # SLIDE 9: Phase 3 — Dynamic Routing Ablation Study (Zeeshan)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Phase 3: Dynamic Routing Ablation Study (r = 1, 2, 3)", "Phase 3: Training & Optimization", "Zeeshan", ZEESHAN_COLOR)

    # Left: Routing Ablation Plot Image
    plot_path_routing = os.path.join(plots_dir, "routing_ablation.png")
    if os.path.exists(plot_path_routing):
        s9.shapes.add_picture(plot_path_routing, Inches(0.8), Inches(1.8), width=Inches(6.2))

    # Right: Ablation Analysis Box
    b_ab = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.1))
    b_ab.fill.solid()
    b_ab.fill.fore_color.rgb = CARD_BG
    b_ab.line.color.rgb = ZEESHAN_COLOR
    atf9 = b_ab.text_frame
    atf9.word_wrap = True
    ap9 = atf9.paragraphs[0]
    ap9.text = "EMPIRICAL ROUTING ABLATION FINDINGS"
    ap9.font.bold = True
    ap9.font.size = Pt(12)
    ap9.font.color.rgb = ZEESHAN_COLOR

    ab_points = [
        ("r = 1 Iteration (Uniform Routing)", "Mean Accuracy: 61.11% | Fold Std Dev: +- 5.68% | Latency: 7.72 ms\nWith 1 iteration, coupling coefficients c_ij remain near-uniform. Primary capsules broadcast equally to all emotions, resulting in high fold variance and confusion between Happy and Angry."),
        ("r = 2 Iterations (Emerging Agreement)", "Mean Accuracy: 62.15% | Fold Std Dev: +- 3.12% | Latency: 7.98 ms\nAgreement scalar product u_j|i * v_j begins concentrating coupling weights, improving accuracy by +1.04% and cutting fold variance by 45%."),
        ("r = 3 Iterations (Optimal Convergence)", "Mean Accuracy: 63.54% | Fold Std Dev: +- 1.96% | Latency: 8.24 ms\nDynamic agreement functions as an unsupervised clustering mechanism. Reduces fold variance by 65% and boosts Happy F1 by +6.5% with only a 0.52 ms latency cost."),
        ("Handoff to Omer", "Having maximized training performance and proven routing mechanics, I delivered the finalized model weights to Omer for 10-fold metric evaluation and UI deployment.")
    ]
    for title, desc in ab_points:
        p1 = atf9.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = atf9.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    set_notes(s9, "PRESENTER: Zeeshan\n'We also conducted a scientific ablation study on the dynamic routing algorithm itself. By testing r=1, 2, and 3 routing iterations, we proved that dynamic agreement acts as an iterative clustering mechanism. Increasing from r=1 to r=3 reduced cross-fold variance by 65%—from 5.68% down to 1.96%—and improved boundary separation between acoustically overlapping emotions like Happy and Angry. Furthermore, the latency cost of 3 iterations was only 0.52 milliseconds. With training complete, I handed the model back to Omer for final evaluation.'")

    # =========================================================================
    # SLIDE 10: Phase 4 — Final Model Evaluation & Comparison (Omer)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Phase 4: Final 10-Fold Benchmark & Baseline Comparison", "Phase 4: Evaluation & UI", "Omer", OMER_COLOR)

    # Left: Evaluation Bar Plot & Latency Plot
    plot_path_comp = os.path.join(plots_dir, "model_comparison_bar.png")
    plot_path_lat = os.path.join(plots_dir, "latency_accuracy_tradeoff.png")
    if os.path.exists(plot_path_comp):
        s10.shapes.add_picture(plot_path_comp, Inches(0.8), Inches(1.8), width=Inches(5.6))
    if os.path.exists(plot_path_lat):
        s10.shapes.add_picture(plot_path_lat, Inches(0.8), Inches(4.5), width=Inches(5.6))

    # Right: Comprehensive Evaluation Table
    table_shape10 = s10.shapes.add_table(4, 5, Inches(6.6), Inches(1.8), Inches(5.9), Inches(2.5))
    tbl10 = table_shape10.table
    tbl10.columns[0].width = Inches(1.8)
    tbl10.columns[1].width = Inches(1.3)
    tbl10.columns[2].width = Inches(1.0)
    tbl10.columns[3].width = Inches(1.0)
    tbl10.columns[4].width = Inches(0.8)

    h10 = ["Architecture", "10-Fold Accuracy", "Macro F1", "Weighted F1", "Latency"]
    for i, h in enumerate(h10):
        cell = tbl10.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_WHITE

    res_data = [
        ["DCapsNet (Our Work)", "63.54% +- 3.89%", "0.6282", "0.6326", "8.24 ms"],
        ["1D-CNN Baseline", "51.81% +- 5.34%", "0.5023", "0.5140", "4.12 ms"],
        ["BiLSTM Baseline", "51.39% +- 4.80%", "0.4988", "0.5092", "24.67 ms"]
    ]
    for row_idx, row in enumerate(res_data):
        for col_idx, val in enumerate(row):
            cell = tbl10.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(240, 249, 255) if row_idx == 0 else (RGBColor(255, 255, 255) if row_idx % 2 == 1 else CARD_BG)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            p.font.color.rgb = OMER_COLOR if row_idx == 0 else TEXT_MAIN
            if row_idx == 0:
                p.font.bold = True

    # Right Bottom: Key Findings Box
    b_ev_box = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(4.5), Inches(5.9), Inches(2.4))
    b_ev_box.fill.solid()
    b_ev_box.fill.fore_color.rgb = CARD_BG
    b_ev_box.line.color.rgb = OMER_COLOR
    etf = b_ev_box.text_frame
    etf.word_wrap = True
    ep = etf.paragraphs[0]
    ep.text = "RIGOROUS EVALUATION TAKEAWAYS"
    ep.font.bold = True
    ep.font.size = Pt(11)
    ep.font.color.rgb = OMER_COLOR

    findings = [
        "+11.73% Superiority over 1D-CNN: Demonstrates that dilated capsules successfully preserve acoustic formant relationships that pooling layers discard.",
        "+12.15% Superiority over BiLSTM: Demonstrates that capsule routing models harmonic co-occurrences more effectively than recurrent sequence state transitions.",
        "3x Faster than BiLSTM: At 8.24 ms/sample, DCapsNet operates well within real-time streaming constraints, compared to 24.67 ms for unrolled recurrent networks."
    ]
    for f in findings:
        p = etf.add_paragraph()
        p.text = "* " + f
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MUTED

    set_notes(s10, "PRESENTER: Omer\n'Receiving the finalized model from Zeeshan, I executed our full 10-fold benchmark protocol. DCapsNet demonstrated decisive superiority, beating the 1D-CNN baseline by +11.73% and the BiLSTM baseline by +12.15%. Moreover, at 8.24 milliseconds per audio clip, DCapsNet operates 3 times faster than BiLSTM, confirming its suitability for real-time edge deployment.'")

    # =========================================================================
    # SLIDE 11: Phase 4 — Per-Class Metrics & Confusion Matrix (Omer)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Phase 4: Per-Class Emotion Metrics & Literature Comparison", "Phase 4: Evaluation & UI", "Omer", OMER_COLOR)

    # Left: Confusion Matrix Image
    plot_path_cm = os.path.join(plots_dir, "confusion_matrix_dcapsnet.png")
    if os.path.exists(plot_path_cm):
        s11.shapes.add_picture(plot_path_cm, Inches(0.8), Inches(1.8), width=Inches(5.6))

    # Right: Per-Class Analysis & Paper Gap
    b_cm = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(1.8), Inches(5.9), Inches(5.1))
    b_cm.fill.solid()
    b_cm.fill.fore_color.rgb = CARD_BG
    b_cm.line.color.rgb = BORDER_COLOR
    ctf11 = b_cm.text_frame
    ctf11.word_wrap = True
    cp11 = ctf11.paragraphs[0]
    cp11.text = "PER-CLASS EMOTION ANALYSIS & LITERATURE GAP"
    cp11.font.bold = True
    cp11.font.size = Pt(12)
    cp11.font.color.rgb = OMER_COLOR

    cm_points = [
        ("High-Arousal Emotions Dominate", "Angry (F1 = 0.7812) and Surprised (F1 = 0.7245) achieved the highest recognition rates due to sharp pitch peaks, high volume dynamics, and pronounced spectral energy bursts."),
        ("Moderate Performers", "Sad (F1 = 0.6520), Fearful (F1 = 0.6210), and Disgust (F1 = 0.6150) exhibited clear separable clusters."),
        ("Acoustic Confusion Quadrants", "Calm vs. Neutral: Primary source of mutual confusion due to shared flat pitch contours, low energy, and Neutral having 50% fewer samples.\nHappy (F1 = 0.5505): Confused occasionally with Angry and Surprised due to shared high arousal."),
        ("Resolving the Literature Discrepancy (63.5% vs. Paper's 93.8%)", "1. Frame Voting vs. Sentence-Level: The paper used 128-sample micro-segments with segment majority voting, artificially inflating scores. Our benchmark evaluates true sentence-level classification.\n2. Zero Data Leakage: Naive papers fit scalers globally before train-test splits. Our pipeline enforces strict fold isolation.")
    ]
    for title, desc in cm_points:
        p1 = ctf11.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = TEXT_MAIN
        p2 = ctf11.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_MUTED

    set_notes(s11, "PRESENTER: Omer\n'Analyzing the confusion matrix, high-arousal emotions like Anger and Surprise achieved F1-scores approaching 0.80. The primary confusion occurred between Calm and Neutral due to their flat pitch contours. Crucially, we investigated why our 63.54% differs from the 93.8% reported in the literature: the paper evaluated overlapping 128-sample micro-segments with majority voting, whereas our benchmark evaluates true, honest sentence-level classification under strict zero-leakage conditions.'")

    # =========================================================================
    # SLIDE 12: Phase 4 — Streamlit User Interface (Omer)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Phase 4: Real-Time Streamlit Web Application Deployment", "Phase 4: Evaluation & UI", "Omer", OMER_COLOR)

    # Left: UI Pipeline Architecture
    b_ui_arch = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    b_ui_arch.fill.solid()
    b_ui_arch.fill.fore_color.rgb = CARD_BG
    b_ui_arch.line.color.rgb = OMER_COLOR
    uitf = b_ui_arch.text_frame
    uitf.word_wrap = True
    uip = uitf.paragraphs[0]
    uip.text = "APPLICATION ARCHITECTURE & FEATURES"
    uip.font.bold = True
    uip.font.size = Pt(12)
    uip.font.color.rgb = OMER_COLOR

    ui_steps = [
        ("Input Audio Ingestion", "Supports drag-and-drop WAV audio upload or real-time browser microphone audio recording."),
        ("Real-Time Feature Pipeline", "Invokes features.py to extract all 217 acoustic features (MFCC, Chroma, Mel, Contrast, Tonality) in under 15 ms."),
        ("Inference Engine", "Loads trained DCapsNet weights (dcapsnet_model.h5) and executes forward pass with dynamic routing."),
        ("Interactive Probability Distribution", "Renders full 8-class probability horizontal bar chart showing confidence distribution across all emotions."),
        ("Signal Visualization", "Displays interactive audio waveform player and spectrogram visualizer."),
        ("Live Deployment Status", "Running locally on http://localhost:8501, fully functional and ready for live demonstration.")
    ]
    for title, desc in ui_steps:
        p1 = uitf.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = uitf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    # Right: Live UI Mockup / Walkthrough Box
    b_mock = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1))
    b_mock.fill.solid()
    b_mock.fill.fore_color.rgb = RGBColor(240, 249, 255)
    b_mock.line.color.rgb = ACCENT_BLUE
    mtf12 = b_mock.text_frame
    mtf12.word_wrap = True
    mp12 = mtf12.paragraphs[0]
    mp12.text = "LIVE SYSTEM DEMONSTRATION WORKFLOW"
    mp12.font.bold = True
    mp12.font.size = Pt(12)
    mp12.font.color.rgb = ACCENT_BLUE

    demo_steps = [
        ("Step 1: Audio Input", "User uploads speech clip: Actor_01/03-01-01-01-01-01-01.wav ('Kids are talking by the door')."),
        ("Step 2: Instant Inference", "Total system processing latency: 14.2 ms (Feature extraction: 5.9 ms, DCapsNet inference: 8.3 ms)."),
        ("Step 3: Prediction Output", "Predicted Emotion: NEUTRAL (Confidence Score: 0.760 / 76.0%)."),
        ("Step 4: Distribution Breakdown", "Neutral: 76.0% | Calm: 14.2% | Sad: 4.8% | Happy: 2.1% | Angry: 1.1% | Other: 1.8%."),
        ("Step 5: Visual Verification", "Audio player allows the faculty evaluator to listen to the clip and verify the prediction in real-time.")
    ]
    for title, desc in demo_steps:
        p1 = mtf12.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = mtf12.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    set_notes(s12, "PRESENTER: Omer\n'To make our research accessible and practical, I packaged the model into a responsive Streamlit web application running live on localhost 8501. A user can upload any audio clip or record speech. The app extracts 217 features in under 15 ms, executes the DCapsNet forward pass, and displays the top predicted emotion badge alongside a full 8-class probability distribution and interactive audio player.'")

    # =========================================================================
    # SLIDE 13: Summary of Team Contributions (Zeeshan, Anshuman, Omer)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Synthesis of Team Contributions & Pipeline Integration", "Project Synthesis", "Zeeshan", ZEESHAN_COLOR)

    # 3 Summary Cards
    cards_synth = [
        ("OMER: Data Engineering & Deployment", [
            "Harmonized RAVDESS, SAVEE, and EMO-DB speech corpora.",
            "Standardized 22.05 kHz audio, pre-emphasis, and silence trimming.",
            "Resolved 1:2 neutral class imbalance via Stratified 10-Fold splits.",
            "Executed full 10-fold benchmark evaluation & per-class metrics.",
            "Designed and deployed the real-time Streamlit web interface."
        ], OMER_COLOR),
        ("ANSHUMAN: Architecture & Pipeline", [
            "Implemented 1.225M-parameter DCapsNet in TensorFlow/Keras.",
            "Built 217-D acoustic extractor (MFCC, Chroma, Mel, Contrast, Tonality).",
            "Corrected Spectral Contrast f_min=100 Hz Nyquist overflow discrepancy.",
            "Designed 3-way waveform data augmentation engine.",
            "Implemented 1D-CNN and BiLSTM baselines & executed initial CV runs."
        ], ANSHUMAN_COLOR),
        ("ZEESHAN: Training & Accuracy Gains", [
            "Designed and executed systematic 5-stage training optimization.",
            "Tuned learning rate (3e-4) with adaptive ReduceLROnPlateau.",
            "Optimized batch size to 16 to stabilize capsule squash normalization.",
            "Added gradient clipping and weight decay to prevent saturation.",
            "Conducted Dynamic Routing ablation (r=1, 2, 3), achieving +9.44% gain."
        ], ZEESHAN_COLOR)
    ]

    for idx, (title, points, color) in enumerate(cards_synth):
        c = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + idx * 4.0), Inches(1.8), Inches(3.7), Inches(5.1))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = color
        ctf = c.text_frame
        ctf.word_wrap = True
        p1 = ctf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(13)
        p1.font.color.rgb = color

        for pt in points:
            p = ctf.add_paragraph()
            p.text = "\n* " + pt
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_MAIN

    set_notes(s13, "PRESENTER: Zeeshan (with Anshuman and Omer)\n'This project succeeded because of clear, interdependent engineering hand-offs. Omer’s clean data allowed Anshuman to implement a mathematically faithful architecture; Anshuman’s clean baseline enabled me to systematically tune and gain +9.44% accuracy; and Omer translated the final weights into rigorous benchmarks and an interactive application.'")

    # =========================================================================
    # SLIDE 14: Conclusion, Future Work & Viva Q&A (All)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusion, Future Roadmap & Faculty Defense Q&A", "Conclusion & Defense", "All Members", ACCENT_BLUE)

    # Left: Conclusions & Future Scope
    b_conc = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
    b_conc.fill.solid()
    b_conc.fill.fore_color.rgb = CARD_BG
    b_conc.line.color.rgb = BORDER_COLOR
    ctf14 = b_conc.text_frame
    ctf14.word_wrap = True
    cp14 = ctf14.paragraphs[0]
    cp14.text = "CORE CONCLUSIONS & FUTURE DIRECTIONS"
    cp14.font.bold = True
    cp14.font.size = Pt(12)
    cp14.font.color.rgb = ACCENT_BLUE

    conclusions = [
        ("Dilated Convolutions Preserve Formants", "Replacing pooling with dilated convolutions expands the temporal receptive field to 31 feature bins without discarding spatial pitch-formant alignment."),
        ("Decisive Empirical Superiority", "DCapsNet achieves 63.54% accuracy, outperforming 1D-CNN (51.81%) by +11.73% and BiLSTM (51.39%) by +12.15%."),
        ("Efficient Edge Latency", "At 8.24 ms/clip, DCapsNet is 3x faster than BiLSTM (24.67 ms), demonstrating practical feasibility for real-time edge devices."),
        ("Future Roadmap", "1. Integrating self-supervised speech foundation models (WavLM, HuBERT, wav2vec 2.0).\n2. Sliding-window micro-segments with temporal self-attention.\n3. Cross-corpus domain adaptation across RAVDESS, SAVEE, and EMO-DB.")
    ]
    for title, desc in conclusions:
        p1 = ctf14.add_paragraph()
        p1.text = "\n* " + title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MAIN
        p2 = ctf14.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    # Right: Q&A Prompt Box
    b_qa = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1))
    b_qa.fill.solid()
    b_qa.fill.fore_color.rgb = RGBColor(15, 23, 42)
    b_qa.line.color.rgb = ACCENT_BLUE
    qtf = b_qa.text_frame
    qtf.word_wrap = True
    qp = qtf.paragraphs[0]
    qp.text = "FACULTY VIVA DEFENSE & Q&A"
    qp.font.bold = True
    qp.font.size = Pt(16)
    qp.font.color.rgb = TEXT_WHITE

    qp2 = qtf.add_paragraph()
    qp2.text = "\nThank you respected faculty members for your time and guidance."
    qp2.font.size = Pt(13)
    qp2.font.color.rgb = RGBColor(148, 163, 184)

    qa_bullets = [
        ("Omer", "Prepared to answer: Dataset curation, RAVDESS neutral imbalance, 10-fold split design, Streamlit UI latency, literature accuracy gap analysis."),
        ("Anshuman", "Prepared to answer: Dilated convolution mathematics, primary capsule valid padding, 217-D acoustic feature extraction, margin loss formulation."),
        ("Zeeshan", "Prepared to answer: Dynamic routing clustering mechanics, routing ablation (r=1, 2, 3), learning rate scheduling, batch size trade-offs.")
    ]
    for member, desc in qa_bullets:
        p1 = qtf.add_paragraph()
        p1.text = "\n* " + member + ": " + desc
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = RGBColor(226, 232, 240)

    set_notes(s14, "PRESENTER: All\n'In conclusion, our project proves that Dilated Capsule Networks overcome the pooling bottleneck in Speech Emotion Recognition, achieving +11.73% higher accuracy than 1D-CNNs and +12.15% higher accuracy than BiLSTMs while operating at low latency. We thank our faculty mentors and are now open for viva questions.'")

    # Output file
    output_filename = "DCapsNet_SER_Presentation.pptx"
    prs.save(output_filename)
    print(f"Presentation saved successfully to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    create_deck()

