import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


DEFAULT_DATASET = "vipoooool/new-plant-diseases-dataset"
DEFAULT_DATA_ROOT = "./data"
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_INITIAL_EPOCHS = 8
DEFAULT_FINETUNE_EPOCHS = 10
DEFAULT_SEED = 1337


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def ensure_dataset(dataset_slug: str, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)

    if any(data_root.iterdir()):
        print(f"Dataset root already has files: {data_root}")
        return

    print(f"Downloading dataset: {dataset_slug}")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(data_root), unzip=True, quiet=False)
    print("Dataset download complete")


def find_train_valid_dirs(data_root: Path) -> tuple[Path, Path]:
    direct_train = data_root / "New Plant Diseases Dataset (Augmented)" / "train"
    direct_valid = data_root / "New Plant Diseases Dataset (Augmented)" / "valid"
    if direct_train.exists() and direct_valid.exists():
        return direct_train, direct_valid

    for candidate in data_root.rglob("train"):
        parent = candidate.parent
        valid_candidate = parent / "valid"
        if valid_candidate.exists() and candidate.is_dir() and valid_candidate.is_dir():
            return candidate, valid_candidate

    raise FileNotFoundError(
        "Could not locate train/valid folders. Expected 'New Plant Diseases Dataset (Augmented)/train' and '/valid'."
    )


def build_datasets(train_dir: Path, valid_dir: Path, img_size: int, batch_size: int, seed: int):
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    valid_ds = keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    return train_ds, valid_ds, class_names


def build_model(num_classes: int, img_size: int):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.18),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)

    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_plant_disease")
    return model, base_model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )


def make_callbacks(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "efficientnet_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def unfreeze_top_layers(base_model: keras.Model, trainable_layers: int = 40) -> None:
    base_model.trainable = True

    if trainable_layers <= 0:
        return

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def main(args):
    set_seed(args.seed)
    configure_gpu()

    data_root = Path(args.data_root)
    ensure_dataset(args.dataset, data_root)
    train_dir, valid_dir = find_train_valid_dirs(data_root)

    print(f"Train directory: {train_dir}")
    print(f"Valid directory: {valid_dir}")

    train_ds, valid_ds, class_names = build_datasets(
        train_dir=train_dir,
        valid_dir=valid_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model, base_model = build_model(num_classes=len(class_names), img_size=args.img_size)

    callbacks = make_callbacks(Path(args.output_dir))

    print("\nPhase 1: Training classifier head")
    compile_model(model, learning_rate=3e-4)
    history_phase1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.initial_epochs,
        callbacks=callbacks,
    )

    print("\nPhase 2: Fine-tuning top backbone layers")
    unfreeze_top_layers(base_model, trainable_layers=args.unfreeze_layers)
    compile_model(model, learning_rate=1e-5)
    history_phase2 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.initial_epochs + args.finetune_epochs,
        initial_epoch=history_phase1.epoch[-1] + 1,
        callbacks=callbacks,
    )

    eval_metrics = model.evaluate(valid_ds, verbose=1)
    metric_names = model.metrics_names
    metrics_map = {name: float(val) for name, val in zip(metric_names, eval_metrics)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = output_dir / "efficientnet_final.keras"
    model.save(final_model_path)

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_map, f, indent=2)

    print("\nTraining complete")
    print(f"Best checkpoint: {output_dir / 'efficientnet_best.keras'}")
    print(f"Final model: {final_model_path}")
    print(f"Metrics: {metrics_map}")
    print(f"Total epochs run: {len(history_phase1.epoch) + len(history_phase2.epoch)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a better plant disease model using EfficientNetB0")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Kaggle dataset slug")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Dataset download/extract folder")
    parser.add_argument("--output-dir", type=str, default="./model_artifacts", help="Folder for checkpoints and outputs")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Image size")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--initial-epochs", type=int, default=DEFAULT_INITIAL_EPOCHS, help="Head training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS, help="Fine-tuning epochs")
    parser.add_argument("--unfreeze-layers", type=int, default=40, help="How many top backbone layers to unfreeze")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

    args = parser.parse_args()
    main(args)
