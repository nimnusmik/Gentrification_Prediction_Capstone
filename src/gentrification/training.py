"""Training utilities for transformer classifier."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model import TransformerClassifier


@dataclass
class TrainingConfig:
    epochs: int = 200
    learning_rate: float = 1e-4
    patience: int = 30
    delta: float = 0.0
    device: str = "auto"


class EarlyStopping:
    def __init__(self, patience: int, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()


def _rmse(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1).float()
    return torch.sqrt(((predictions - targets.float()) ** 2).mean()).item()


def train_model(
    model: TransformerClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    save_path: Path | str,
) -> Dict[str, List[float]]:
    """Train the model and persist the best checkpoints."""

    device = _resolve_device(config.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    stopper = EarlyStopping(patience=config.patience, delta=config.delta)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_rmse": [],
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                logits = model(features)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                val_accuracy += _accuracy(logits, targets)
                val_rmse += _rmse(logits, targets)

        val_loss /= max(1, len(val_loader))
        val_accuracy /= max(1, len(val_loader))
        val_rmse /= max(1, len(val_loader))

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_rmse"].append(val_rmse)

        if stopper.step(val_loss):
            best_state = model.state_dict()
            torch.save(best_state, save_path)

        if config.patience > 0 and stopper.should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["config"] = [asdict(config)]
    history["checkpoint"] = [str(save_path)]
    return history
