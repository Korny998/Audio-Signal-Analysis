# Neural Networks for Audio Signal Analysis (Music Genre Classification)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A simple deep learning project demonstrating how neural networks can be
used for **audio signal analysis** and **music genre classification**.

The project processes audio files from the **GTZAN dataset**, extracts
spectral and temporal features, and trains a dense neural network to
classify tracks by genre.

The goal is to illustrate how deep learning models can learn from
engineered audio features such as:

- Spectral centroid
- Spectral rolloff
- Zero-crossing rate
- MFCC
- Chroma STFT

---

# Project Structure

```bash
├── constants.py        # Global configuration and hyperparameters
├── dataset.py          # Dataset download, feature extraction, and splits
├── model.py            # Neural network architecture
├── train.py            # Training and evaluation pipeline
├── visualization.py    # Confusion matrix and model evaluation
├── LICENSE             # Project license
└── README.md           # Project documentation
```

---

# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Visualization](#visualization)
- [License](#license)

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Korny998/Audio-Signal-Analysis.git
cd Audio-Signal-Analysis
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Usage

To train the model and evaluate it on the test set, run:

```bash
python train.py
```

The script will automatically:

1. Download the **GTZAN music genre dataset** if it is not already
   present locally
2. Extract audio features from each `.wav` file
3. Normalize the features using **StandardScaler**
4. Split the data into training, validation, and test sets
5. Train the neural network
6. Evaluate the model and display a confusion matrix

---

# Dataset

The project uses the **GTZAN dataset** for music genre classification.

Source:

```text
https://storage.yandexcloud.net/academy.ai/gtzan-dataset-music-genre-classification.zip
```

Each audio file belongs to one genre folder, and the model predicts the
genre label from extracted frame-level audio features.

Key dataset settings from `constants.py`:

- `DURATION_SEC = 30`
- `CLASS_FILES = 100`
- `FILE_INDEX_TRAIN_SPLIT = 90`
- `HOP_LENGTH = 512`
- `VALIDATION_SPLIT = 0.1`

---

# Feature Extraction

The preprocessing pipeline extracts several standard audio features using
`librosa`.

Features used:

- **Spectral Centroid** - indicates the center of mass of the spectrum
- **Spectral Rolloff** - estimates the frequency below which most energy
  is concentrated
- **Zero-Crossing Rate** - measures how frequently the signal changes sign
- **MFCC** - compact representation of timbral properties
- **Chroma STFT** - captures pitch class information

The extracted features are stacked into a feature matrix where each row
represents one time step.

---

# Model Architecture

The project uses a fully connected neural network for multiclass genre
classification.

Architecture:

```text
Input
Dense(256) -> ReLU
Dropout(0.3)
BatchNormalization
Dense(128) -> ReLU
Dropout(0.3)
BatchNormalization
Dense(64) -> ReLU
Dropout(0.3)
BatchNormalization
Dense(CLASS_COUNT) -> Softmax
```

Purpose:

- Learn patterns from extracted audio descriptors
- Classify each audio frame into one of the dataset genres
- Provide a simple baseline architecture for audio classification

---

# Training Pipeline

The training process follows these steps:

1. Load and preprocess the dataset
2. Extract frame-level audio features
3. Encode target labels as one-hot vectors
4. Normalize input features with `StandardScaler`
5. Split the training data into train and validation sets
6. Build the model
7. Compile it with:

```text
Adam optimizer
Categorical Crossentropy loss
Accuracy metric
```

8. Train the model for:

```text
EPOCHS = 100
BATCH_SIZE = 512
```

9. Evaluate the model on the test set

---

# Visualization

The `visualization.py` module generates an evaluation report using a
normalized confusion matrix.

It provides:

- Test loss and accuracy
- Predicted class labels
- A normalized confusion matrix for all genres

This helps identify:

- which genres are classified well
- which genres are often confused
- how balanced the classifier is across classes

---

# Example Workflow

```text
Download dataset -> Extract features -> Scale data ->
Train model -> Evaluate predictions -> Plot confusion matrix
```

---

# License

This project is released under the **MIT License**.

You are free to use, modify, and distribute the code for educational or
commercial purposes.
