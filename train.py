# train.py
# ------------------------------------------------------
# Training script
# ------------------------------------------------------

import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocessing import get_datasets
from cnn_model import build_custom_cnn

def main():
    # Config
    DATA_DIR = "dataset"       # path to dataset folder
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 25
    SEED = 123
    MODEL_DIR = "saved_models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load datasets
    train_ds, val_ds, test_ds = get_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=0.15,
        test_split=0.15,
        seed=SEED,
    )

    # Build model
    model = build_custom_cnn(input_shape=(*IMG_SIZE, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    model.summary()

    # Callbacks
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, "final_model.h5")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    # Optionally: you can use test_ds here or evaluate separately

if __name__ == "__main__":
    main()