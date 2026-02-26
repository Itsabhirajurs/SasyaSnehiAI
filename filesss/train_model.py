"""Minimal MobileNetV2 transfer-learning training script for MVP."""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes):
    """Build transfer-learning model using MobileNetV2 base."""
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Sashyasnehi AI disease classifier")
    parser.add_argument("--train-dir", required=True, help="Path to training images directory")
    parser.add_argument("--val-dir", required=True, help="Path to validation images directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (5-10 suggested)")
    parser.add_argument(
        "--output",
        default="model/disease_model.keras",
        help="Output model file path",
    )
    args = parser.parse_args()

    train_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        args.train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )
    val_data = val_gen.flow_from_directory(
        args.val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    model = build_model(num_classes=train_data.num_classes)
    model.fit(train_data, validation_data=val_data, epochs=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
