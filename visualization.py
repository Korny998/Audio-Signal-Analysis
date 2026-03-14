import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dataset import CLASS_LIST


def eval_model(model, x, y_true):
    loss, acc = model.evaluate(x, y_true)
    print(f'Ошибка: {loss}, точность: {acc}')

    y_pred = model.predict(x)
    cm = confusion_matrix(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        normalize='true'
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Матрица ошибок нормализованная')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_LIST,
    )
    disp.plot(ax=ax)
    plt.show()
