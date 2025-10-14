"""Reusable demos for classic machine-learning baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


@dataclass
class DemoResult:
    figure: plt.Figure
    metrics: Dict[str, float]


def _plot_confusion_matrix(matrix: np.ndarray, labels: Tuple[str, ...]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def iris_knn_demo(n_neighbors: int = 3, test_size: float = 0.3, random_state: int = 42) -> DemoResult:
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    confusion = metrics.confusion_matrix(y_test, predictions)
    fig = _plot_confusion_matrix(confusion, iris.target_names)

    report = metrics.classification_report(y_test, predictions, output_dict=True)
    accuracy = metrics.accuracy_score(y_test, predictions)

    return DemoResult(figure=fig, metrics={"accuracy": accuracy, "f1_macro": report["macro avg"]["f1-score"]})


def iris_kmeans_demo(n_clusters: int = 3, random_state: int = 42) -> DemoResult:
    iris = datasets.load_iris()
    reducer = PCA(n_components=2, random_state=random_state)
    reduced = reducer.fit_transform(iris.data)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(reduced)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=50)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", s=150, marker="x")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("KMeans on Iris (PCA reduced)")

    inertia = model.inertia_
    return DemoResult(figure=fig, metrics={"inertia": inertia})


def iris_decision_tree_demo(random_state: int = 42) -> DemoResult:
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=random_state
    )

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(
        model,
        filled=True,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        ax=ax,
        rounded=True,
    )
    ax.set_title("Decision Tree on Iris")

    return DemoResult(figure=fig, metrics={"accuracy": accuracy})


def mnist_knn_demo(n_neighbors: int = 3, test_size: float = 0.5, random_state: int = 42) -> DemoResult:
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=test_size, random_state=random_state
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    confusion = metrics.confusion_matrix(y_test, predictions)
    fig = _plot_confusion_matrix(confusion, tuple(map(str, digits.target_names)))

    accuracy = metrics.accuracy_score(y_test, predictions)
    return DemoResult(figure=fig, metrics={"accuracy": accuracy})


def mnist_kmeans_demo(n_clusters: int = 10, random_state: int = 42) -> DemoResult:
    digits = datasets.load_digits()
    data = digits.data / 16.0

    reducer = PCA(n_components=2, random_state=random_state)
    reduced = reducer.fit_transform(data)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(reduced)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=20)
    ax.set_title("KMeans on Digits (PCA reduced)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")

    inertia = model.inertia_
    return DemoResult(figure=fig, metrics={"inertia": inertia})
