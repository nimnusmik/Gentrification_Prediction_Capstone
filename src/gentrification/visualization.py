"""Visual helpers for inspecting model attention weights."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import TransformerClassifier


def attention_to_dataframe(
    model: TransformerClassifier,
    features: np.ndarray,
    feature_names: Iterable[str],
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Return a dataframe mapping feature names to average attention weights."""

    if features.ndim != 2:
        raise ValueError("Expected 2D feature array [n_samples, n_features]")

    feature_tensor = torch.from_numpy(features.astype(np.float32))
    if device:
        feature_tensor = feature_tensor.to(device)

    model_device = next(model.parameters()).device
    feature_tensor = feature_tensor.to(model_device)

    with torch.no_grad():
        weights = model.attention_weights(feature_tensor)

    weights_mean = weights.mean(dim=0).cpu().numpy().flatten()
    frame = pd.DataFrame({"feature": list(feature_names), "weight": weights_mean})
    frame = frame.sort_values("weight", ascending=False).reset_index(drop=True)
    return frame


def plot_weight_bars(
    weights_df: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    positive_color: str = "#1f77b4",
    negative_color: str = "#d62728",
    title: str = "Attention Weight Breakdown",
) -> plt.Figure:
    """Plot positive and negative weights side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    positive = weights_df[weights_df["weight"] >= 0]
    negative = weights_df[weights_df["weight"] < 0]

    axes[0].bar(positive["feature"], positive["weight"], color=positive_color)
    axes[0].set_title("Weight ≥ 0")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(negative["feature"], negative["weight"], color=negative_color)
    axes[1].set_title("Weight < 0")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
