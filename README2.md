# Gentrification Prediction Capstone

Refactored toolkit for experimenting with gentrification prediction models. The project now exposes reusable Python modules plus simple CLI scripts for training a transformer classifier and running baseline classical ML demos.

## Repository Layout

- `src/gentrification/` – reusable code for data loading, model definition, training utilities, inference helpers, visualisation, and classic ML demos.
- `scripts/` – command line entry points (`train_transformer.py`, `predict_transformer.py`).
- `notebooks/` – original research notebooks preserved for reference.
- `artifacts/` – default output directory for checkpoints and preprocessing artefacts (created after training).

## Quick Start

1. Install dependencies (example using pip):
   ```bash
   pip install torch scikit-learn pandas numpy matplotlib seaborn joblib
   ```
2. Train the transformer model on your CSV dataset:
   ```bash
   python scripts/train_transformer.py path/to/result_after_cluster.csv --target clust --feature-start 3
   ```
   Adjust `--drop-columns` or `--feature-columns` when your schema differs.
3. Generate predictions for new data using the saved artefacts:
   ```bash
   python scripts/predict_transformer.py path/to/newdata2023.csv --artifacts artifacts --output predictions.csv
   ```

## Module Overview

- `gentrification.data` – `DataConfig` for declaring dataset schema and `prepare_datasets` for building PyTorch dataloaders together with scalers and label encoders.
- `gentrification.model` – `TransformerClassifier` exposes a batch-first multi-head attention model tailored for tabular data.
- `gentrification.training` – `TrainingConfig` and `train_model` implement early-stopped optimisation with tracked metrics.
- `gentrification.prediction` – utilities for loading checkpoints and producing batched predictions.
- `gentrification.visualization` – helpers to convert attention weights into dataframes and bar charts.
- `gentrification.demos` – reusable Iris/MNIST demo functions covering KNN, KMeans, and decision trees.

## Next Steps

- Extend `scripts/train_transformer.py` with custom logging or wandb integration when experimenting at scale.
- Wrap data preprocessing (categorical encoding, imputation) before calling `prepare_datasets` if your inputs require it.

