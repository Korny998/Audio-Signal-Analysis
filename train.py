from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dataset import (
    x_train, x_val,
    y_train, y_val,
    x_test, y_test
)
from model import build_model
from visualization import eval_model


def train() -> tuple[Model, History]:
    # Build the network using the feature width and number of classes.
    model = build_model(
        input_shape=x_train.shape[1:],
        class_count=y_train.shape[1],
    )
    # Configure optimization, loss, and the metric shown during training.
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    # Train on the training split and monitor performance.
    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=512,
        validation_data=(x_val, y_val),
    )
    return model, history


if __name__ == '__main__':
    model, history = train()
    # Evaluate the trained model on the held-out test split.
    eval_model(model, x_test, y_test)
