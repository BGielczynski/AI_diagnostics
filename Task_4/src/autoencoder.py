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
        hidden_layer_sizes: tuple[int, ...] = (20, 12, 20),
        threshold_quantile: float = 0.98,
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
        # Fachübliche Konvention: 1 = beschädigt/anomal, 0 = gesund.
        return np.where(self.anomaly_score(x) > self.threshold_, 1, 0)

    def save(self, output_path: Path) -> None:
        joblib.dump(self, output_path)


def calculate_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    # Fachübliche Konvention: 1 = anomal/positiv, 0 = gesund/negativ.
    anomaly_true = true_labels == 1
    anomaly_pred = predicted_labels == 1
    tn = int(np.sum(~anomaly_true & ~anomaly_pred))
    fp = int(np.sum(~anomaly_true & anomaly_pred))
    fn = int(np.sum(anomaly_true & ~anomaly_pred))
    tp = int(np.sum(anomaly_true & anomaly_pred))

    sensitivity_anomaly = tp / (tp + fn) if tp + fn else 0.0
    specificity_healthy = tn / (tn + fp) if tn + fp else 0.0
    precision_anomaly = tp / (tp + fp) if tp + fp else 0.0
    f1_anomaly = (
        2 * precision_anomaly * sensitivity_anomaly
        / (precision_anomaly + sensitivity_anomaly)
        if precision_anomaly + sensitivity_anomaly
        else 0.0
    )
    precision_healthy = tn / (tn + fn) if tn + fn else 0.0
    recall_healthy = specificity_healthy
    f1_healthy = (
        2 * precision_healthy * recall_healthy
        / (precision_healthy + recall_healthy)
        if precision_healthy + recall_healthy
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(true_labels, scores) if len(np.unique(true_labels)) == 2 else np.nan
    return {
        "n_samples": len(true_labels),
        "n_anomaly": int(np.sum(anomaly_true)),
        "n_healthy": int(np.sum(~anomaly_true)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sensitivity_anomaly_1": sensitivity_anomaly,
        "specificity_healthy_0": specificity_healthy,
        "accuracy": accuracy,
        "balanced_accuracy": (sensitivity_anomaly + specificity_healthy) / 2,
        "recall_anomaly_1": sensitivity_anomaly,
        "precision_anomaly_1": precision_anomaly,
        "f1_anomaly_1": f1_anomaly,
        "recall_healthy_0": recall_healthy,
        "precision_healthy_0": precision_healthy,
        "f1_healthy_0": f1_healthy,
        "roc_auc_anomaly_1": auc,
    }


def plot_confusion_matrix(metrics: dict, output_path: Path, title: str) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    cell_labels = np.array(
        [
            ["TN", "FP"],
            ["FN", "TP"],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Gesund\n(Label 0)", "Anomal\n(Label 1)"])
    ax.set_yticks([0, 1], ["Gesund\n(Label 0)", "Anomal\n(Label 1)"])
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title(f"{title}\nNegativ: gesund (0) | Positiv: anomal (1)")
    color_threshold = float(matrix.max()) / 2 if matrix.size else 0.0
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                f"{value}\n({cell_labels[row, column]})",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="semibold",
                color="white" if value > color_threshold else "black",
            )
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
    ax.hist(
        test_scores[test_labels == 0],
        bins=bins,
        alpha=0.55,
        label="Test gesund (Label 0)",
        color="#59a14f",
    )
    ax.hist(
        test_scores[test_labels == 1],
        bins=bins,
        alpha=0.55,
        label="Test anomal (Label 1)",
        color="#e15759",
    )
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
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"Autoencoder, AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Zufall")
    ax.set_xlabel("False Positive Rate (Anomalie als Zielklasse)")
    ax.set_ylabel("Sensitivität / Recall der Anomalieklasse")
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
