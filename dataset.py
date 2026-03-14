import io
import os
import zipfile
from typing import TypeAlias

import librosa
import numpy as np
import requests
import tensorflow.keras as tf
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import (
    DURATION_SEC,
    CLASS_FILES,
    FILE_INDEX_TRAIN_SPLIT,
    HOP_LENGTH,
    PROJECT_DIR,
    VALIDATION_SPLIT
)


url = (
    'https://storage.yandexcloud.net/academy.ai/'
    'gtzan-dataset-music-genre-classification.zip'
)

data_dir = os.path.join(PROJECT_DIR, 'genres_original')

# Download and unpack the dataset only if it is not already available locally.
if not os.path.isdir(data_dir) or not os.listdir(data_dir):
    request = requests.get(url, timeout=60)
    request.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(request.content)) as z:
        z.extractall(PROJECT_DIR)

# Keep only genre folders, ignoring any unrelated files.
CLASS_LIST = [
    entry for entry in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, entry))
]
CLASS_LIST.sort()
CLASS_COUNT = len(CLASS_LIST)

FloatArray: TypeAlias = NDArray[np.float_]
Float32Array: TypeAlias = NDArray[np.float32]


def get_features(
    y: FloatArray,
    sr: int,
    hop_length: int = HOP_LENGTH,
) -> dict[str, FloatArray]:
    # Extract several standard audio descriptors from the waveform.
    spec_cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )
    rollof = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=y, hop_length=hop_length
    )
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, hop_length=hop_length
    )
    chroma_stft = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length
    )
    return {
        'spectral_centroid': spec_cent,
        'spectral_rolloff': rollof,
        'zero_crossing_rate': zcr,
        'mfcc': mfcc,
        'chroma_stft': chroma_stft,
    }


def stack_features(feat: dict[str, FloatArray]) -> FloatArray:
    # Stack feature matrices vertically.
    features: FloatArray | None = None
    for v in feat.values():
        if features is None:
            features = v
        else:
            features = np.vstack((features, v))
    if features is None:
        raise ValueError('No features were provided for stacking.')
    return features.T


def process_file(
    class_index: int,
    file_index: int,
    duration_sec: int,
) -> tuple[str, Float32Array, Float32Array]:
    x_list: list[FloatArray] = []
    y_list: list[FloatArray] = []

    class_name = CLASS_LIST[class_index]

    file_name = f'{class_name}.{str(file_index).zfill(5)}.wav'
    song_name = os.path.join(data_dir, class_name, file_name)

    # Load a fixed-duration mono signal from one audio file.
    y, sr = librosa.load(song_name, mono=True, duration=duration_sec)

    features = get_features(y, sr)
    feature_set = stack_features(features)

    # Convert the class id into a one-hot target vector.
    y_label = tf.utils.to_categorical(class_index, CLASS_COUNT)

    for j in range(feature_set.shape[0]):
        x_list.append(feature_set[j])
        y_list.append(y_label)

    return (
        song_name,
        np.array(x_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


x_train_data: Float32Array | None = None
y_train_data: Float32Array | None = None

x_test: Float32Array | None = None
y_test: Float32Array | None = None

for class_index in range(len(CLASS_LIST)):
    for file_index in range(0, FILE_INDEX_TRAIN_SPLIT):
        try:
            _, file_x_data, file_y_data = process_file(
                class_index,
                file_index,
                DURATION_SEC,
            )
            x_train_data = (
                file_x_data
                if x_train_data is None
                else np.vstack([x_train_data, file_x_data])
            )
            y_train_data = (
                file_y_data
                if y_train_data is None
                else np.vstack([y_train_data, file_y_data])
            )
        except Exception as exc:
            print(
                'Skipping train file '
                f'class_index={class_index}, file_index={file_index}: {exc}'
            )
            continue

    for file_index in range(FILE_INDEX_TRAIN_SPLIT, CLASS_FILES):
        try:
            _, file_x_data, file_y_data = process_file(
                class_index,
                file_index,
                DURATION_SEC,
            )
            x_test = (
                file_x_data
                if x_test is None
                else np.vstack([x_test, file_x_data])
            )
            y_test = (
                file_y_data
                if y_test is None
                else np.vstack([y_test, file_y_data])
            )
        except Exception as exc:
            print(
                'Skipping test file '
                f'class_index={class_index}, file_index={file_index}: {exc}'
            )
            continue

if (
    x_train_data is None
    or y_train_data is None
    or x_test is None
    or y_test is None
):
    raise RuntimeError(
        'Dataset processing produced no usable samples. '
        'Check the printed file errors above.'
    )

# Normalize features using statistics from the training split only.
x_scaler = StandardScaler()
x_train_data_scaled = x_scaler.fit_transform(x_train_data)
x_test = x_scaler.transform(x_test)

# Create a validation split while preserving the class balance.
x_train, x_val, y_train, y_val = train_test_split(
    x_train_data_scaled,
    y_train_data,
    stratify=np.argmax(y_train_data, axis=1),
    test_size=VALIDATION_SPLIT,
    random_state=42
)
