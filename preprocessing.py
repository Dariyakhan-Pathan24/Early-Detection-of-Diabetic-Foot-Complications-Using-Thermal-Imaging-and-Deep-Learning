# preprocessing.py
# ------------------------------------------------------
# Data loading + augmentation + normalization
# ------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

def get_datasets(
    data_dir,
    img_size=(224, 224),
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    seed=123,
):

    # Load dataset
    full_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    class_names = full_ds.class_names
    print(f"Classes detected: {class_names}")

    # Calculate sizes
    total_batches = full_ds.cardinality().numpy()
    train_batches = int(total_batches * (1 - val_split - test_split))
    val_batches = int(total_batches * val_split)

    # Split dataset
    train_ds = full_ds.take(train_batches)
    val_ds = full_ds.skip(train_batches).take(val_batches)
    test_ds = full_ds.skip(train_batches + val_batches)

    # 🔥 Data Augmentation (Improved)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ])

    def augment(images, labels):
        images = data_augmentation(images)
        return images, labels

    # Normalize
    def normalize(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    # Apply transformations
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)

    val_ds = val_ds.map(normalize, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(normalize, num_parallel_calls=AUTOTUNE)

    # Optimize pipeline
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds