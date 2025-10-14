#!/usr/bin/env python3
"""Train the transformer classifier on a CSV dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import dump

from gentrification.data import DataConfig, prepare_datasets
from gentrification.model import ModelConfig, TransformerClassifier
from gentrification.training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to the training CSV file")
    parser.add_argument("--target", default="clust", help="Target column name")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        help="Columns to drop before training",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help="Explicit list of feature columns to keep",
    )
    parser.add_argument(
        "--feature-start",
        type=int,
        default=3,
        help="Index of the first feature column when using positional slicing",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--delta", type=float, default=0.0, help="Early stopping improvement margin")
    parser.add_argument("--device", default="auto", help="Training device identifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store checkpoints and scalers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        target_column=args.target,
        drop_columns=args.drop_columns,
        feature_columns=args.feature_columns,
        feature_slice=slice(args.feature_start, None) if args.feature_columns is None else None,
        batch_size=args.batch_size,
    )

    bundle = prepare_datasets(args.data, data_config)

    model_config = ModelConfig(
        input_size=len(bundle.feature_names),
        num_classes=len(bundle.class_names),
    )
    model = TransformerClassifier(model_config)

    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        delta=args.delta,
        device=args.device,
    )

    checkpoint_path = output_dir / "transformer.pt"
    history = train_model(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        config=training_config,
        save_path=checkpoint_path,
    )

    dump(bundle.scaler, output_dir / "scaler.joblib")
    dump(bundle.label_encoder, output_dir / "label_encoder.joblib")

    metadata = {
        "feature_names": bundle.feature_names,
        "class_names": bundle.class_names,
        "model_config": model_config.__dict__,
        "training_history": history,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Training complete. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
