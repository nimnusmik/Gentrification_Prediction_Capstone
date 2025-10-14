"""Helpers for loading checkpoints and generating predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import ModelConfig, TransformerClassifier


def load_model(checkpoint_path: Path | str, config: ModelConfig, device: str = "cpu") -> TransformerClassifier:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(device)
    model = TransformerClassifier(config)
    state = torch.load(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(state)
    model = model.to(resolved_device)
    model.eval()
    return model


def _build_loader(features: np.ndarray, batch_size: int) -> DataLoader:
    tensor = torch.from_numpy(features.astype(np.float32))
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def generate_predictions(
    model: TransformerClassifier,
    features: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    loader = _build_loader(features, batch_size=batch_size)
    device = next(model.parameters()).device

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features)
            outputs.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs)
