from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from pathlib import Path
import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

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
        return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.mean(a != b)  # type: ignore


class UniformWeightedKNN:
    def __init__(
        self, k: int, distance_fn: DistanceMetric.Fn = DistanceMetric.euclidean
    ):
        self.k = k
        self.distance_fn = distance_fn

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions: list[int] = []

        for x in X:
            distances = [self.distance_fn(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Majority Vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)

def load_dataset(
    basedir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(basedir.joinpath("dataset", "data.csv"))  # pyright: ignore[reportUnknownMemberType]
    df = df.drop(columns="Unnamed: 32")
    X = df.drop(columns=["id", "diagnosis"]).values
    Y = df["diagnosis"].map({"B": 0, "M": 1}).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=42, train_size=0.80, shuffle=True
    )  # type: ignore
    
    
    # Normalize Features (Min-Max Scaling)
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    range_val = max_val - min_val

    # Avoid division by zero for constant columns
    range_val[range_val == 0] = 1
    X_train_normalized = (X_train - min_val) / range_val
    X_test_normalized = (X_test - min_val) / range_val
    
    
    return X_train_normalized, X_test_normalized, y_train, y_test  # type: ignore

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall, [[tn, fp], [fn, tp]]


def plot_decision_boundary(basedir: Path, distance_functions, X_train, y_train, best_config, feature_names):
    print("\nGenerating Decision Boundary Plot (using first 2 features)...")

    feature_idx_1, feature_idx_2 = 0, 1
    X_vis = X_train[:, [feature_idx_1, feature_idx_2]]

    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 0.1, X_vis[:, 0].max() + 0.1
    y_min, y_max = X_vis[:, 1].min() - 0.1, X_vis[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Re-train KNN on just these 2 features for visualization
    knn_vis = UniformWeightedKNN(k=best_config['k'], distance_fn=distance_functions[best_config['metric']])
    knn_vis.fit(X_vis, y_train)

    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn_vis.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm, s=20)
    plt.xlabel(feature_names[feature_idx_1])
    plt.ylabel(feature_names[feature_idx_2])
    plt.title(f"Decision Boundary (K={best_config['k']}, {best_config['metric']})")
    plt.savefig(basedir.joinpath("task1_extra.png"))


def main(basedir: Path) -> None:
    X_train, X_test, y_train, y_test = load_dataset(basedir)
    

    k_values = [3, 4, 9, 20, 47]

    distance_functions: dict[str, DistanceMetric.Fn] = {
        "Euclidean": DistanceMetric.euclidean,
        "Manhattan": DistanceMetric.manhattan,
        "Minkowski": DistanceMetric.minkowski,
        "Cosine": DistanceMetric.cosine,
        "Hamming": DistanceMetric.hamming,
    }
    
    
    
    print("\n---- KNN Experiment Results ----\n")
    
    accuracy_results: dict[str, list[float]] = {name: [] for name, _ in distance_functions.items()}
    best_accuracy = -1
    best_info = {}
    best_model = None


    for k in k_values:
        for metric, dist_fn in distance_functions.items():
            knn = UniformWeightedKNN(k, dist_fn)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            
            acc, prec, rec, cm = calculate_metrics(y_test, y_pred)
            accuracy_results[metric].append(acc)

            print(f"K={k:2d}, Distance={metric:10s}, Accuracy={acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = knn
                best_info = {
                    'config': {'k': k, 'metric': metric},
                    'metrics': {'acc': acc, 'prec': prec, 'rec': rec, 'cm': cm}
                }
                
    # -------------------------------
    # Best Model Results
    # -------------------------------
    config = best_info['config']
    metrics = best_info['metrics']

    print("\n===================================")
    print("Best Model Selected")
    print(f"K = {config['k']}")
    print(f"Distance Metric = {config['metric']}")
    print(f"Test Accuracy = {metrics['acc']:.4f}")
    print("===================================\n")

    print(f"Precision: {metrics['prec']:.4f}")
    print(f"Recall:    {metrics['rec']:.4f}")
    
    print(f"Confusion Matrix:\n{np.array(metrics['cm'])}")
    
    
    plt.figure(figsize=(10, 6))
    for metric, accs in accuracy_results.items():
        plt.plot(k_values, accs, marker='o', label=metric)

    plt.title("K vs Accuracy for Different Distance Metrics")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(basedir.joinpath("task1.png"))
    
    
    # Inference for Euclidean / Manhattan / Minkowski
    # These metrics are sensitive to feature scaling
    # Since no normalization was applied, distances get dominated by large-scale features
    # Larger K increases bias → underfitting
    
    # Inference for Cosine
    # Cosine distance performs best because:
    # It measures orientation, not magnitude
    # Feature scaling issues are minimized
    # Medium K (9) balances noise and generalization
    # Very large K (47) causes underfitting
    
    # Inference for Hamming
    # Lowest performance across all K
    # Dataset is continuous → binarization loses information
    # Accuracy decreases with increasing K
    
    plot_decision_boundary(basedir, distance_functions, X_train, y_train, best_info['config'], {1: 'M', 0: 'B'})



if __name__ == "__main__":
    main(basedir=Path(os.path.dirname(__file__)))
