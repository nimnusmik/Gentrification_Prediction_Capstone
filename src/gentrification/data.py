"""Data loading and preprocessing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataConfig:
    """Configuration for building training and validation datasets."""

    target_column: str = "clust"
    drop_columns: Optional[Sequence[str]] = None
    feature_columns: Optional[Sequence[str]] = None
    feature_slice: Optional[slice] = slice(3, None)
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 128


@dataclass
class DatasetBundle:
    """Collection of artefacts produced while preparing the dataset."""

    train_loader: DataLoader
    val_loader: DataLoader
    feature_names: List[str]
    class_names: List[str]
    scaler: StandardScaler
    label_encoder: LabelEncoder


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Load a CSV file into a dataframe with basic validation."""

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty")
    return df


def _select_features(df: pd.DataFrame, config: DataConfig) -> Tuple[pd.DataFrame, pd.Series]:
    if config.target_column not in df.columns:
        raise KeyError(f"Target column '{config.target_column}' missing from dataset")

    working_df = df.copy()
    if config.drop_columns:
        missing = set(config.drop_columns) - set(working_df.columns)
        if missing:
            raise KeyError(f"Columns to drop not present: {sorted(missing)}")
        working_df = working_df.drop(columns=list(config.drop_columns))

    y = working_df.pop(config.target_column)

    if config.feature_columns is not None:
        missing = set(config.feature_columns) - set(working_df.columns)
        if missing:
            raise KeyError(f"Requested feature columns not present: {sorted(missing)}")
        X = working_df.loc[:, list(config.feature_columns)]
    elif config.feature_slice is not None:
        X = working_df.iloc[:, config.feature_slice]
    else:
        X = working_df

    if X.select_dtypes(include=["number"]).shape[1] != X.shape[1]:
        non_numeric = X.columns.difference(X.select_dtypes(include=["number"]).columns)
        raise TypeError(
            "All features must be numeric after preprocessing. Non-numeric columns: "
            f"{list(non_numeric)}"
        )

    return X, y


def prepare_datasets(path: Path | str, config: DataConfig) -> DatasetBundle:
    """Load a dataset, split it, and return PyTorch dataloaders with metadata."""

    df = load_dataframe(path)
    X_df, y_series = _select_features(df, config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series.values)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y_encoded,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None,
    )

    X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
    X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        feature_names=list(X_df.columns),
        class_names=list(label_encoder.classes_.astype(str)),
        scaler=scaler,
        label_encoder=label_encoder,
    )
