from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model


def build_model(input_shape: tuple[int, ...], class_count: int) -> Model:
    # A simple fully connected classifier for per-frame audio features.
    return Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(class_count, activation='softmax')
    ])
