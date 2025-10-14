#!/usr/bin/env python3
"""Generate predictions for a new dataset using the trained transformer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from gentrification.data import load_dataframe
from gentrification.model import ModelConfig
from gentrification.prediction import generate_predictions, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to the inference CSV file")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory that contains checkpoint and metadata from training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path for saving predictions",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Name of the output column added to the dataframe",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts

    metadata_path = artifacts / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing training metadata at {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    model_config = ModelConfig(**metadata["model_config"])

    scaler = load(artifacts / "scaler.joblib")
    label_encoder = load(artifacts / "label_encoder.joblib")

    df = load_dataframe(args.data)
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise KeyError(f"Inference data missing required features: {sorted(missing)}")

    features = df[feature_names].values
    scaled_features = scaler.transform(features)

    model = load_model(artifacts / "transformer.pt", model_config)
    predictions = generate_predictions(model, scaled_features)
    labels = label_encoder.inverse_transform(predictions)

    df_with_predictions = df.copy()
    df_with_predictions[args.prediction_column] = labels

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_with_predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(df_with_predictions.head())


if __name__ == "__main__":
    main()
