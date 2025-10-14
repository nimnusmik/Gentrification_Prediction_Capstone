"""Utilities for the gentrification prediction project."""

from .data import DataConfig, DatasetBundle, load_dataframe, prepare_datasets
from .demos import (
    DemoResult,
    iris_decision_tree_demo,
    iris_kmeans_demo,
    iris_knn_demo,
    mnist_kmeans_demo,
    mnist_knn_demo,
)
from .model import ModelConfig, TransformerClassifier
from .prediction import generate_predictions, load_model
from .training import TrainingConfig, train_model
from .visualization import attention_to_dataframe, plot_weight_bars

__all__ = [
    "DataConfig",
    "DatasetBundle",
    "TrainingConfig",
    "ModelConfig",
    "TransformerClassifier",
    "DemoResult",
    "iris_knn_demo",
    "iris_kmeans_demo",
    "iris_decision_tree_demo",
    "mnist_knn_demo",
    "mnist_kmeans_demo",
    "load_dataframe",
    "prepare_datasets",
    "train_model",
    "load_model",
    "generate_predictions",
    "attention_to_dataframe",
    "plot_weight_bars",
]
