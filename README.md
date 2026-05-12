# OralSense 🦷

**Multimodal Oral Cancer Risk Stratification Using Smartphone Images and Areca Nut–Inclusive Behavioral Metadata**  
*A Cross-Population Explainable AI Study*

> ⚠️ **Disclaimer:** OralSense is intended for research and screening purposes only. It is **not** a substitute for clinical diagnosis. All predictions must be reviewed by a qualified medical professional.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Setup & Installation](#setup--installation)
- [Web App Features](#web-app-features)
- [Clinical Risk Levels](#clinical-risk-levels)
- [References](#references)
- [Authors](#authors)

---

## Overview

OralSense is a deep learning clinical decision support system for **early oral cancer risk stratification**. It combines smartphone oral cavity images with behavioral metadata (areca nut, tobacco, alcohol, age, sex) using two complementary AI models.

Key capabilities:
- Monte Carlo Dropout uncertainty estimation (20 inference passes)
- Explainable predictions via Grad-CAM heatmaps
- Multimodal fusion of visual + behavioral signals

The project addresses a critical healthcare gap in South Asia, where oral cancer rates are among the highest in the world due to widespread areca nut and tobacco use.

**Risk Classes:**

| Label | Meaning |
|-------|---------|
| ✅ Normal | No signs of malignancy |
| ⚠️ Variations | Minor tissue variations |
| 🔶 OPMD | Oral Potentially Malignant Disorder |
| 🔴 OC | Oral Cancer — immediate referral required |

---

## Key Results

| Model | Validation Accuracy |
|-------|-------------------|
| Image-Only (MobileNetV2) | 90.3% |
| Multimodal (Image + Metadata) | **93.1%** |

---

## Dataset

### Primary — SMART-OM
- **2,469 images**, 331 subjects, Tamil Nadu, India
- 4 classes: Normal, Variation, OPMD, Oral Cancer
- Metadata: areca nut, tobacco type, alcohol, age, sex
- DOI: [10.6084/m9.figshare.31341790](https://doi.org/10.6084/m9.figshare.31341790)

### Supplementary — CODE *(planned)*
- ~500 images, 110 subjects, Ragas Dental College, Chennai
- DOI: [10.6084/m9.figshare.30550889](https://doi.org/10.6084/m9.figshare.30550889)

### Validation — Peradeniya *(planned)*
- 3,000 images, 714 subjects, Sri Lanka
- Cross-population validation

---

## Project Structure

```
oral_cancer_ai/
├── app.py                        # Flask backend API
├── database.py                   # SQLite patient database
├── index.html                    # OralSense web app frontend
├── patient_history.html          # Patient history dashboard
├── data/
│   ├── augmented/                # Augmented training dataset
│   │   ├── Normal/               # 2,145 images
│   │   ├── OC/                   # 500 images (augmented from 20)
│   │   ├── OPMD/                 # 500 images (augmented from 125)
│   │   └── Variations/           # 500 images (augmented from 179)
│   ├── metadata_clean.csv        # Cleaned behavioral metadata
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── best_model.h5             # Image-only MobileNetV2 model
│   └── multimodal_model.h5       # Multimodal fusion model
├── notebooks/
│   ├── check_dataset.py
│   ├── explore_dataset.py
│   ├── prepare_dataset.py
│   ├── augment_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── gradcam.py
│   ├── explore_metadata.py
│   ├── prepare_multimodal.py
│   ├── train_multimodal.py
│   ├── evaluate_multimodal.py
│   ├── retrain_multimodal.py
│   └── generate_report.py
└── results/
    ├── sample_images.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── gradcam_results.png
    ├── multimodal_training.png
    ├── multimodal_confusion_matrix.png
    └── project_report.pdf
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Pretrained Model | MobileNetV2 (ImageNet) |
| Loss Function | Focal Loss (γ=2, α=0.25) |
| Uncertainty | Monte Carlo Dropout (20 runs) |
| Image Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainability | Grad-CAM |
| Web Backend | Flask + Flask-CORS |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |

---

## Model Architecture

### Image-Only Model

```
Input (224×224×3)
    → MobileNetV2 (pretrained ImageNet, last 30 layers unfrozen)
    → GlobalAveragePooling2D
    → Dense(256, ReLU) → Dropout(0.5)
    → Dense(128, ReLU) → Dropout(0.4)
    → Dense(4, Softmax)
```

### Multimodal Fusion Model

```
Image Input (224×224×3)         Metadata Input (6 features)
    → MobileNetV2                   → Dense(32, ReLU)
    → GlobalAveragePooling2D        → BatchNormalization
    → Dense(256, ReLU)              → Dense(16, ReLU)
    → Dropout(0.5)                          |
              └──────── Concatenate ────────┘
                            → Dense(128, ReLU)
                            → Dropout(0.4)
                            → Dense(4, Softmax)
```

### Metadata Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Continuous (0–1) | Patient age, normalized |
| Sex | Binary | 0 = Female, 1 = Male |
| Smoking | Binary | Smoking habit |
| Chewing | Binary | Chewing tobacco |
| Areca Nut | Binary | Areca nut usage |
| Alcohol | Binary | Alcohol consumption |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Focal Loss (γ=2.0, α=0.25) |
| Class Weights | Balanced — OC: 1.82×, Normal: 0.42× |
| Augmentation | Flip, rotation, brightness, zoom, shear |
| Early Stopping | Patience = 10 on `val_accuracy` |
| Uncertainty | Monte Carlo Dropout — 20 inference passes |
| Decision Rule | Higher risk result between both models |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/oralsense-ai.git
cd oralsense-ai
```

### 2. Create conda environment

```bash
conda create -n oralcancer python=3.10
conda activate oralcancer
```

### 3. Install dependencies

```bash
pip install tensorflow opencv-python pandas numpy matplotlib seaborn \
            scikit-learn flask flask-cors tqdm openpyxl
```

### 4. Download the dataset

Visit the [SMART-OM Figshare page](https://doi.org/10.6084/m9.figshare.31341790) and download the dataset. Extract it into the `data/` folder.

### 5. Run notebooks in order

```bash
python notebooks/check_dataset.py
python notebooks/augment_dataset.py
python notebooks/train_model.py
python notebooks/train_multimodal.py
```

### 6. Start the web app

```bash
python app.py
```

Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Web App Features

- 📷 Upload oral cavity smartphone photo
- 👤 Patient details — name, ID, age, sex, phone, address, doctor, department
- ⚠️ Habit history — smoking, chewing, areca nut, alcohol
- 🤖 Dual AI prediction — image model + multimodal model
- 📊 Monte Carlo Dropout — 20 forward passes for uncertainty estimation
- 🔥 Grad-CAM heatmap — visual explanation of model focus
- 📋 Auto-generated patient report with clinical recommendation
- 🖨️ Print / PDF export of patient report
- 📊 Patient history — track risk progression across visits
- 🔴 Risk alerts — automatic alert if risk has increased since last visit
- 💾 SQLite database — all scans saved automatically

---

## Clinical Risk Levels

| Risk | Class | Recommended Action |
|------|-------|--------------------|
| ✅ Low | Normal | Routine follow-up in 12 months |
| ⚠️ Low-Medium | Variations | Follow-up in 3–6 months |
| 🔶 Medium | OPMD | Refer to specialist within 2–4 weeks |
| 🔴 High | OC | Immediate oncologist referral |

---

## References

1. Devindi et al. — *Multimodal Deep CNN Pipeline for AI-Assisted Early Detection of Oral Cancer* — IEEE Access, 2024
2. Frontiers in Oral Health — *AI and the Diagnosis of Oral Cavity Cancer from Clinical Photographs* — 2025
3. Ou et al. — *Deep Learning Based Multimodal Fusion Model for Skin Lesion Diagnosis* — Frontiers in Surgery, 2022
4. Sharma et al. — *Exploring Data Modalities and Advances in AI for Oral Cancer Detection* — IET Image Processing, 2025
5. Pivarathne et al. — *A Comprehensive Dataset of Annotated Oral Cavity Images* — Oral Oncology, 2024

---

## Authors

**S. Sruti & M P Kavi Nisha**  
Oral Cancer Risk Stratification using Multimodal AI  
Tamil Nadu, India · May 2026
