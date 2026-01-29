from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
from typing import Callable

class DistanceMetric:
    type Fn = Callable[[np.ndarray, np.ndarray], np.ndarray]

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(a - b))

    @staticmethod
    def minkowski(a: np.ndarray, b: np.ndarray, p: int = 3) -> np.ndarray:
        return np.sum(np.abs(a - b) ** p) ** (1 / p)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.mean(a != b)  # pyright: ignore[reportUnknownVariableType]


class UniformWeightedKNN:
    def __init__(
        self,
        k: int,
        distance_fn: DistanceMetric.Fn = DistanceMetric.euclidean,
    ):
        self.k: int = k
        self.distance_fn: DistanceMetric.Fn = distance_fn
        self._train_data: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._train_data = (X_train, y_train)

    def predict(self, X_test: np.ndarray) -> list[int]:
        if self._train_data is None:
            raise RuntimeError("please fit first.")
        X_train, y_train = self._train_data  # pyright: ignore[reportConstantRedefinition]
        predictions: list[int] = []
        for x in X_test:
            k_nearest = [
                label
                for _, label in heapq.nsmallest(
                    self.k,
                    (
                        (self.distance_fn(x, x_train), y)
                        for x_train, y in zip(X_train, y_train)
                    ),
                    key=lambda t: t[0],
                )
            ]
            predictions.append(
                max(set(k_nearest), key=k_nearest.count)
            )  # max-voting and in case of draw choosing first
        return predictions


def load_cifar_batch(file_path: Path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X = batch[b"data"]  # shape (10000, 3072)
    y = np.array(batch[b"labels"])
    return X, y


def load_cifar10(root: Path):
    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        X, y = load_cifar_batch(root / f"data_batch_{i}")
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)

    X_test, y_test = load_cifar_batch(root / "test_batch")

    return X_train, y_train, X_test, y_test


def main(basedir: Path) -> None:
    X_train, y_train, X_test, y_test = load_cifar10(
        basedir.joinpath("dataset", "cifar-10-batches-py")
    )

    # Normalize to [0,1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # ---------- SUBSAMPLING (reduce training time) ----------
    train_size = 5000
    test_size = 2000

    train_idx = random.sample(range(len(X_train)), train_size)
    test_idx = random.sample(range(len(X_test)), test_size)

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]

    Ks = [1, 3, 5, 7, 9]
    distance_functions = {
        "Euclidean": DistanceMetric.euclidean,
        "Manhattan": DistanceMetric.manhattan,
        "Minkowski": DistanceMetric.minkowski,
        "Cosine": DistanceMetric.cosine,
        "Hamming": DistanceMetric.hamming,
    }

    results = {}
    for name, dist_fn in distance_functions.items():
        results[name] = []
        print(f"\nDistance Metric: {name}")
        for k in Ks:
            knn = UniformWeightedKNN(k=k, distance_fn=dist_fn)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name].append(acc)
            print(f"K={k}, Accuracy={acc:.4f}")

    # ---------- Best Model ----------
    best_acc = 0
    best_k = None
    best_dist = None

    for dist, accs in results.items():
        for k, acc in zip(Ks, accs):
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_dist = dist

    print("\nBest Model")
    print("----------")
    print(f"K = {best_k}")
    print(f"Distance = {best_dist}")
    print(f"Accuracy = {best_acc:.4f}")

    best_knn = UniformWeightedKNN(
        k=best_k,
        distance_fn=distance_functions[best_dist],
    )
    best_knn.fit(X_train, y_train)
    best_preds = best_knn.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_preds))

    print(
        f"Precision (macro): {precision_score(y_test, best_preds, average='macro'):.4f}"
    )
    print(f"Recall (macro): {recall_score(y_test, best_preds, average='macro'):.4f}")

    # ---------- Plot ----------
    plt.figure(figsize=(10, 6))
    for dist, accs in results.items():
        plt.plot(Ks, accs, marker="o", label=dist)

    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10: K vs Accuracy for different distance metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(basedir.joinpath("task2.png"))


if __name__ == "__main__":
    main(basedir=Path(os.path.dirname(__file__)))
