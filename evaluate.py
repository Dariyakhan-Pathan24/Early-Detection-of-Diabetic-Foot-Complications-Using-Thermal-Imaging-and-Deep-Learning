# evaluate.py
# ------------------------------------------------------
# Evaluate the saved best model on the test set
# ------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf

from preprocessing import get_datasets

def main():
    # Config
    DATA_DIR = "dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    SEED = 123
    MODEL_PATH = "saved_models/best_model.h5"  # use the best model

    # Load datasets
    _, _, test_ds = get_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=0.15,
        test_split=0.15,
        seed=SEED,
    )

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Collect predictions
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy().astype(int))
        y_pred.extend((preds > 0.5).astype(int).flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"]
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()