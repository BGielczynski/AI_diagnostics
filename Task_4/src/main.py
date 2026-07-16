import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoencoder import (
    AutoencoderDetector,
    calculate_metrics,
    get_feature_columns,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_loss_curve,
    plot_roc,
)
from dataframe_manager import DataFrameManager
from extraction import extract_features
from split import VALIDATION_SPLITS, select_train_test


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "Task_2" / "data" / "sig"
RESULTS_DIR = ROOT_DIR / "results"
FEATURE_FILE = RESULTS_DIR / "features_per_measurement.csv"
CHANNELS = ("Ch1", "Ch2")
PREDICTION_COLUMNS = [
    "path", "fn", "spec", "pos", "mID", "time", "rID", "sID", "label"
]


def load_or_extract_features(force_extract: bool = False) -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)
    if FEATURE_FILE.exists() and not force_extract:
        print(f"Lade vorhandene Einzelmessungs-Features: {FEATURE_FILE}")
        return pd.read_csv(FEATURE_FILE, dtype={"mID": str, "rID": str})

    manager = DataFrameManager(str(DATA_DIR))
    manager.load_signals()
    raw_df = manager.get_dataframe()
    if raw_df.empty:
        raise FileNotFoundError(f"Keine WAV-Dateien unter {DATA_DIR}")
    print(f"Extrahiere Features fuer {len(raw_df)} einzelne Messzeilen ...")
    feature_df = extract_features(raw_df)
    feature_df.to_csv(FEATURE_FILE, index=False)
    return feature_df


def run_model(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    split,
    channel: str,
) -> tuple[dict, pd.DataFrame]:
    train_df, test_df = select_train_test(feature_df, split, channel)
    model_dir = RESULTS_DIR / f"{split.name}_{channel}"
    model_dir.mkdir(exist_ok=True)

    x_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    detector = AutoencoderDetector().fit(x_train)
    scores = detector.anomaly_score(x_test)
    predicted = detector.predict_label(x_test)
    labels = test_df["label"].astype(int).to_numpy()
    metrics = calculate_metrics(labels, predicted, scores)
    metrics.update(
        {
            "fold": split.name,
            "channel": channel,
            "train_specs": "+".join(split.train_specs),
            "test_specs": "+".join(split.test_specs),
            "n_train": len(train_df),
            "threshold": detector.threshold_,
            "epochs": detector.model.n_iter_,
        }
    )

    predictions = test_df[PREDICTION_COLUMNS].copy()
    predictions.insert(0, "fold", split.name)
    predictions["predicted_label"] = predicted
    predictions["anomaly_score"] = scores
    predictions["threshold"] = detector.threshold_
    predictions["normalized_anomaly_score"] = scores / detector.threshold_
    predictions.to_csv(model_dir / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)
    detector.save(model_dir / "autoencoder.joblib")

    plot_confusion_matrix(metrics, model_dir / "confusion_matrix.png", f"{split.name} – {channel}")
    plot_error_distribution(
        detector.train_errors_, labels, scores, detector.threshold_,
        model_dir / "reconstruction_errors.png", f"Rekonstruktionsfehler: {split.name} – {channel}",
    )
    plot_roc(labels, scores, model_dir / "roc_curve.png", f"ROC: {split.name} – {channel}")
    plot_loss_curve(detector, model_dir / "training_loss.png", f"Training: {split.name} – {channel}")
    return metrics, predictions


def aggregate_predictions(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    grouped = predictions.groupby(group_columns, sort=True) if group_columns else [((), predictions)]
    for key, group in grouped:
        metrics = calculate_metrics(
            group["label"].to_numpy(),
            group["predicted_label"].to_numpy(),
            group["normalized_anomaly_score"].to_numpy(),
        )
        if group_columns:
            key_values = key if isinstance(key, tuple) else (key,)
            metrics.update(dict(zip(group_columns, key_values)))
        rows.append(metrics)
    return pd.DataFrame(rows)


def plot_metric_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    columns = ["sensitivity_anomaly", "specificity_normal", "balanced_accuracy", "f1_anomaly"]
    labels = metrics_df["fold"] + " " + metrics_df["channel"]
    x = np.arange(len(metrics_df))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, column in enumerate(columns):
        ax.bar(x + (index - 1.5) * width, metrics_df[column], width, label=column)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Separate Ergebnisse der acht Autoencoder")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summarize_scores_by_spec(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["fold", "sID", "spec", "label"], as_index=False)
        .agg(
            n_samples=("label", "size"),
            predicted_anomaly_rate=("predicted_label", lambda values: float(np.mean(values == 0))),
            mean_normalized_score=("normalized_anomaly_score", "mean"),
            median_normalized_score=("normalized_anomaly_score", "median"),
            max_normalized_score=("normalized_anomaly_score", "max"),
        )
    )


def main(force_extract: bool = False) -> None:
    feature_df = load_or_extract_features(force_extract)
    feature_columns = get_feature_columns(feature_df)
    print(f"Samples: {len(feature_df)}, Features: {len(feature_columns)}")
    print(feature_df.groupby(["spec", "sID"]).size().to_string())

    all_metrics = []
    all_predictions = []
    for split in VALIDATION_SPLITS:
        for channel in CHANNELS:
            print(f"\nTrainiere {split.name}/{channel}: train={split.train_specs}, test={split.test_specs}")
            metrics, predictions = run_model(feature_df, feature_columns, split, channel)
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            print(
                f"Sens={metrics['sensitivity_anomaly']:.3f}, "
                f"Spec={metrics['specificity_normal']:.3f}, "
                f"BAR={metrics['balanced_accuracy']:.3f}, F1={metrics['f1_anomaly']:.3f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(RESULTS_DIR / "all_model_metrics.csv", index=False)
    predictions_df.to_csv(RESULTS_DIR / "all_test_predictions.csv", index=False)
    summarize_scores_by_spec(predictions_df).to_csv(
        RESULTS_DIR / "score_summary_by_spec.csv", index=False
    )

    by_fold = aggregate_predictions(predictions_df, ["fold"])
    by_channel = aggregate_predictions(predictions_df, ["sID"])
    overall = aggregate_predictions(predictions_df, [])
    by_fold.to_csv(RESULTS_DIR / "aggregated_by_fold_metrics.csv", index=False)
    by_channel.to_csv(RESULTS_DIR / "aggregated_by_channel_metrics.csv", index=False)
    overall.to_csv(RESULTS_DIR / "overall_metrics.csv", index=False)
    metrics_df.select_dtypes(include="number").mean().to_frame().T.to_csv(
        RESULTS_DIR / "macro_average_model_metrics.csv", index=False
    )

    plot_metric_summary(metrics_df, RESULTS_DIR / "model_metric_summary.png")
    overall_metrics = overall.iloc[0].to_dict()
    plot_confusion_matrix(
        overall_metrics, RESULTS_DIR / "overall_confusion_matrix.png",
        "Aggregierte Verwechslungsmatrix (alle Folds und Kanaele)",
    )
    plot_roc(
        predictions_df["label"].to_numpy(), predictions_df["normalized_anomaly_score"].to_numpy(),
        RESULTS_DIR / "overall_roc_curve.png", "Aggregierte ROC-Kurve",
    )
    print("\nAggregiertes Ergebnis:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4: Ein-Klassen-Klassifikation")
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Audiofeatures auch bei vorhandenem CSV neu extrahieren",
    )
    args = parser.parse_args()
    main(force_extract=args.force_extract)
