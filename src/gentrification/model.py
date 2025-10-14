"""Model definitions for gentrification prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int = 128
    embedding_size: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    num_classes: int = 3


class TransformerClassifier(nn.Module):
    """A lightweight transformer-style classifier for tabular data."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.input_size, config.embedding_size)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size),
        )
        self.layer_norm1 = nn.LayerNorm(config.embedding_size)
        self.layer_norm2 = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        self.output_head = nn.Linear(config.embedding_size, config.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Expect features shaped [batch, num_features]. Project to embedding space.
        x = self.input_projection(features)
        x = x.unsqueeze(1)  # add sequence dimension
        attn_out, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))

        ff_out = self.feedforward(x)
        x = self.layer_norm2(x + self.dropout(ff_out))

        x = x.squeeze(1)
        logits = self.output_head(x)
        return logits

    def attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(features)
        x = x.unsqueeze(1)
        _, weights = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
        return weights.mean(dim=1)
