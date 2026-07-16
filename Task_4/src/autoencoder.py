from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {
    "path", "fn", "spec", "pos", "mID", "time", "rID", "sID", "label",
    "mID_count", "source_file_count",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column for column in df.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


class AutoencoderDetector:
    """Dense autoencoder with a reconstruction-error anomaly score."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 8, 32),
        threshold_quantile: float = 0.95,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.threshold_quantile = threshold_quantile
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            tol=1e-5,
            random_state=random_state,
        )
        self.threshold_: float | None = None
        self.train_errors_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame) -> "AutoencoderDetector":
        scaled = self.scaler.fit_transform(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(scaled, scaled)
        reconstructed = self.model.predict(scaled)
        self.train_errors_ = np.mean(np.square(scaled - reconstructed), axis=1)
        self.threshold_ = float(
            np.quantile(self.train_errors_, self.threshold_quantile, method="higher")
        )
        return self

    def anomaly_score(self, x: pd.DataFrame) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Das Modell wurde noch nicht trainiert.")
        scaled = self.scaler.transform(x)
        reconstructed = self.model.predict(scaled)
        return np.mean(np.square(scaled - reconstructed), axis=1)

    def predict_label(self, x: pd.DataFrame) -> np.ndarray:
        # Projektkonvention: 0 = beschaedigt/anomal, 1 = gesund.
        return np.where(self.anomaly_score(x) > self.threshold_, 0, 1)

    def save(self, output_path: Path) -> None:
        joblib.dump(self, output_path)


def calculate_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    anomaly_true = np.asarray(true_labels) == 0
    anomaly_pred = np.asarray(predicted_labels) == 0
    tp = int(np.sum(anomaly_true & anomaly_pred))
    fn = int(np.sum(anomaly_true & ~anomaly_pred))
    fp = int(np.sum(~anomaly_true & anomaly_pred))
    tn = int(np.sum(~anomaly_true & ~anomaly_pred))

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    anomaly_targets = anomaly_true.astype(int)
    auc = roc_auc_score(anomaly_targets, scores) if len(np.unique(anomaly_targets)) == 2 else np.nan
    return {
        "n_samples": len(true_labels),
        "n_anomaly": int(np.sum(anomaly_true)),
        "n_normal": int(np.sum(~anomaly_true)),
        "true_positive_anomaly": tp,
        "false_negative_anomaly": fn,
        "false_positive_anomaly": fp,
        "true_negative_anomaly": tn,
        "sensitivity_anomaly": sensitivity,
        "specificity_normal": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision_anomaly": precision,
        "f1_anomaly": f1,
        "roc_auc_anomaly": auc,
    }


def plot_confusion_matrix(metrics: dict, output_path: Path, title: str) -> None:
    matrix = np.array(
        [
            [metrics["true_positive_anomaly"], metrics["false_negative_anomaly"]],
            [metrics["false_positive_anomaly"], metrics["true_negative_anomaly"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Anomal", "Gesund"])
    ax.set_yticks([0, 1], ["Anomal", "Gesund"])
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsaechlich")
    ax.set_title(title)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_error_distribution(
    train_errors: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.histogram_bin_edges(np.concatenate([train_errors, test_scores]), bins=28)
    ax.hist(train_errors, bins=bins, alpha=0.45, label="Training gesund", color="#4c78a8")
    ax.hist(test_scores[test_labels == 1], bins=bins, alpha=0.55, label="Test gesund", color="#59a14f")
    ax.hist(test_scores[test_labels == 0], bins=bins, alpha=0.55, label="Test Z05", color="#e15759")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Schwelle {threshold:.3f}")
    ax.set_xlabel("Mittlerer quadratischer Rekonstruktionsfehler")
    ax.set_ylabel("Anzahl")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_roc(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str) -> None:
    anomaly_true = (labels == 0).astype(int)
    fpr, tpr, _ = roc_curve(anomaly_true, scores)
    auc = roc_auc_score(anomaly_true, scores)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"Autoencoder, AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Zufall")
    ax.set_xlabel("False-Positive-Rate")
    ax.set_ylabel("True-Positive-Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_loss_curve(detector: AutoencoderDetector, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(detector.model.loss_curve_) + 1), detector.model.loss_curve_)
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Rekonstruktions-Loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
