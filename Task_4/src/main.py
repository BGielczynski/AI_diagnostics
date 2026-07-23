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
from dataframe_manager import DataFrameManager, LABEL_BY_SPEC
from extraction import extract_features
from split import VALIDATION_SPLITS, select_train_test


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "Task_2" / "data" / "sig"
RESULTS_ROOT = ROOT_DIR / "results"
SHARED_RESULTS_DIR = RESULTS_ROOT / "shared"
PER_MEASUREMENT_DIR = RESULTS_ROOT / "per_measurement"
MID_AVERAGED_DIR = RESULTS_ROOT / "mid_averaged"
COMPARISON_DIR = RESULTS_ROOT / "comparison"
FEATURE_FILE = SHARED_RESULTS_DIR / "features_per_measurement.csv"
LEGACY_FEATURE_FILE = RESULTS_ROOT / "features_per_measurement.csv"
CHANNELS = ("Ch1", "Ch2")
PIPELINE_DISPLAY_NAMES = {
    "per_measurement": "Einzelmessungen",
    "mid_averaged": "mID-gemittelte Merkmale",
}
PREDICTION_COLUMNS = [
    "path", "fn", "spec", "pos", "mID", "time", "rID", "sID", "label"
]
AVERAGE_GROUP_COLUMNS = ["spec", "pos", "rID", "sID", "label"]


def load_or_extract_features(force_extract: bool = False) -> pd.DataFrame:
    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_FILE.exists() and not force_extract:
        print(f"Lade gemeinsamen Einzelmessungs-Featurecache: {FEATURE_FILE}")
        feature_df = pd.read_csv(FEATURE_FILE, dtype={"mID": str, "rID": str})
        expected_labels = feature_df["spec"].map(LABEL_BY_SPEC).astype(int)
        if not feature_df["label"].astype(int).equals(expected_labels):
            feature_df["label"] = expected_labels
            feature_df.to_csv(FEATURE_FILE, index=False)
        return feature_df

    if LEGACY_FEATURE_FILE.exists() and not force_extract:
        print(f"Migriere vorhandenen Featurecache nach: {FEATURE_FILE}")
        feature_df = pd.read_csv(LEGACY_FEATURE_FILE, dtype={"mID": str, "rID": str})
        feature_df["label"] = feature_df["spec"].map(LABEL_BY_SPEC).astype(int)
        feature_df.to_csv(FEATURE_FILE, index=False)
        return feature_df

    manager = DataFrameManager(str(DATA_DIR))
    manager.load_signals()
    raw_df = manager.get_dataframe()
    if raw_df.empty:
        raise FileNotFoundError(f"Keine WAV-Dateien unter {DATA_DIR}")
    print(f"Extrahiere Features fuer {len(raw_df)} einzelne Messzeilen ...")
    feature_df = extract_features(raw_df)
    feature_df.to_csv(FEATURE_FILE, index=False)
    return feature_df


def average_features_over_mid(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Average already extracted features over mID, as in Task 3."""
    feature_columns = get_feature_columns(feature_df)
    averaged = (
        feature_df.groupby(AVERAGE_GROUP_COLUMNS, as_index=False)
        .agg(
            {
                **{column: "mean" for column in feature_columns},
                "mID": "nunique",
                "fn": "count",
            }
        )
        .rename(columns={"mID": "mID_count", "fn": "source_file_count"})
    )
    averaged["path"] = "features_averaged_over_mid"
    averaged["fn"] = (
        averaged["spec"] + "_" + averaged["pos"] + "_avgMID_"
        + averaged["rID"].astype(str) + "_" + averaged["sID"]
    )
    averaged["mID"] = "mean"
    averaged["time"] = "mean"
    return averaged


def run_model(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    split,
    channel: str,
    results_dir: Path,
    pipeline_name: str,
) -> tuple[dict, pd.DataFrame]:
    train_df, test_df = select_train_test(feature_df, split, channel)
    model_dir = results_dir / f"{split.name}_{channel}"
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    detector = AutoencoderDetector().fit(x_train)
    scores = detector.anomaly_score(x_test)
    predicted = detector.predict_label(x_test)
    labels = test_df["label"].astype(int).to_numpy()
    metrics = calculate_metrics(labels, predicted, scores)
    metrics.update(
        {
            "pipeline": pipeline_name,
            "fold": split.name,
            "channel": channel,
            "train_specs": "+".join(split.train_specs),
            "test_specs": "+".join(split.test_specs),
            "n_train": len(train_df),
            "threshold": detector.threshold_,
            "epochs": detector.model.n_iter_,
        }
    )

    available_metadata = [column for column in PREDICTION_COLUMNS if column in test_df]
    predictions = test_df[available_metadata].copy()
    predictions.insert(0, "pipeline", pipeline_name)
    predictions.insert(1, "fold", split.name)
    if "mID_count" in test_df:
        predictions["mID_count"] = test_df["mID_count"].to_numpy()
        predictions["source_file_count"] = test_df["source_file_count"].to_numpy()
    predictions["predicted_label"] = predicted
    predictions["anomaly_score"] = scores
    predictions["threshold"] = detector.threshold_
    predictions["normalized_anomaly_score"] = scores / detector.threshold_
    predictions.to_csv(model_dir / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)
    detector.save(model_dir / "autoencoder.joblib")

    pipeline_display = PIPELINE_DISPLAY_NAMES.get(pipeline_name, pipeline_name)
    fold_display = split.name.replace("_", " ").title()
    title_suffix = f"{pipeline_display} – {fold_display}, {channel}"
    plot_confusion_matrix(metrics, model_dir / "confusion_matrix.png", title_suffix)
    plot_error_distribution(
        detector.train_errors_, labels, scores, detector.threshold_,
        model_dir / "reconstruction_errors.png", f"Rekonstruktionsfehler: {title_suffix}",
    )
    plot_roc(labels, scores, model_dir / "roc_curve.png", f"ROC: {title_suffix}")
    plot_loss_curve(detector, model_dir / "training_loss.png", f"Training: {title_suffix}")
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


def plot_metric_summary(metrics_df: pd.DataFrame, output_path: Path, pipeline_name: str) -> None:
    columns = [
        "sensitivity_healthy_1",
        "specificity_anomaly_0",
        "balanced_accuracy",
        "f1_healthy_1",
    ]
    display_labels = {
        "sensitivity_healthy_1": "TPR / Sensitivität",
        "specificity_anomaly_0": "TNR / Spezifität",
        "balanced_accuracy": "Balanced Accuracy",
        "f1_healthy_1": "F1-Score gesund (Label 1)",
    }
    labels = metrics_df["fold"].str.replace("_", " ").str.title() + " – " + metrics_df["channel"]
    x = np.arange(len(metrics_df))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, column in enumerate(columns):
        ax.bar(
            x + (index - 1.5) * width,
            metrics_df[column],
            width,
            label=display_labels[column],
        )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    pipeline_display = PIPELINE_DISPLAY_NAMES.get(pipeline_name, pipeline_name)
    ax.set_title(f"Modellmetriken – {pipeline_display}")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summarize_scores_by_spec(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["pipeline", "fold", "sID", "spec", "label"], as_index=False)
        .agg(
            n_samples=("label", "size"),
            predicted_anomaly_rate=("predicted_label", lambda values: float(np.mean(values == 0))),
            mean_normalized_score=("normalized_anomaly_score", "mean"),
            median_normalized_score=("normalized_anomaly_score", "median"),
            max_normalized_score=("normalized_anomaly_score", "max"),
        )
    )


def run_pipeline(
    feature_df: pd.DataFrame, results_dir: Path, pipeline_name: str
) -> dict[str, pd.DataFrame]:
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = get_feature_columns(feature_df)
    if len(feature_columns) != 62:
        raise ValueError(f"Erwartet werden 62 Modellfeatures, erhalten: {len(feature_columns)}")

    sample_counts = feature_df.groupby(["spec", "sID"], as_index=False).size()
    sample_counts.to_csv(results_dir / "sample_counts.csv", index=False)
    print(f"\nPipeline {pipeline_name}: Samples={len(feature_df)}, Features={len(feature_columns)}")
    print(sample_counts.to_string(index=False))

    all_metrics = []
    all_predictions = []
    for split in VALIDATION_SPLITS:
        for channel in CHANNELS:
            print(f"Trainiere {pipeline_name}/{split.name}/{channel}")
            metrics, predictions = run_model(
                feature_df, feature_columns, split, channel, results_dir, pipeline_name
            )
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            print(
                f"  TPR/Sensitivität={metrics['sensitivity_healthy_1']:.3f}, "
                f"TNR/Spezifität={metrics['specificity_anomaly_0']:.3f}, "
                f"BA={metrics['balanced_accuracy']:.3f}, "
                f"F1(gesund)={metrics['f1_healthy_1']:.3f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(results_dir / "all_model_metrics.csv", index=False)
    predictions_df.to_csv(results_dir / "all_test_predictions.csv", index=False)
    summarize_scores_by_spec(predictions_df).to_csv(
        results_dir / "score_summary_by_spec.csv", index=False
    )

    by_fold = aggregate_predictions(predictions_df, ["fold"])
    by_channel = aggregate_predictions(predictions_df, ["sID"])
    overall = aggregate_predictions(predictions_df, [])
    by_fold.insert(0, "pipeline", pipeline_name)
    by_channel.insert(0, "pipeline", pipeline_name)
    overall.insert(0, "pipeline", pipeline_name)
    by_fold.to_csv(results_dir / "aggregated_by_fold_metrics.csv", index=False)
    by_channel.to_csv(results_dir / "aggregated_by_channel_metrics.csv", index=False)
    overall.to_csv(results_dir / "overall_metrics.csv", index=False)

    metric_columns = [
        "sensitivity_healthy_1", "specificity_anomaly_0", "accuracy",
        "balanced_accuracy", "recall_healthy_1", "precision_healthy_1",
        "f1_healthy_1", "recall_anomaly_0", "precision_anomaly_0",
        "f1_anomaly_0", "roc_auc_anomaly_0",
    ]
    macro = metrics_df[metric_columns].mean().to_frame().T
    macro.insert(0, "pipeline", pipeline_name)
    macro.to_csv(results_dir / "macro_average_model_metrics.csv", index=False)

    plot_metric_summary(metrics_df, results_dir / "model_metric_summary.png", pipeline_name)
    plot_confusion_matrix(
        overall.iloc[0].to_dict(), results_dir / "overall_confusion_matrix.png",
        f"Aggregierte Verwechslungsmatrix – {PIPELINE_DISPLAY_NAMES.get(pipeline_name, pipeline_name)}",
    )
    plot_roc(
        predictions_df["label"].to_numpy(),
        predictions_df["normalized_anomaly_score"].to_numpy(),
        results_dir / "overall_roc_curve.png",
        f"Aggregierte ROC – {PIPELINE_DISPLAY_NAMES.get(pipeline_name, pipeline_name)}",
    )
    print(f"Aggregiertes Ergebnis {pipeline_name}:")
    print(overall.to_string(index=False))
    return {
        "model_metrics": metrics_df,
        "predictions": predictions_df,
        "by_fold": by_fold,
        "by_channel": by_channel,
        "overall": overall,
        "macro": macro,
    }


def create_pipeline_comparison(results_by_pipeline: dict[str, dict[str, pd.DataFrame]]) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    fold_frames = []
    for pipeline_name, results in results_by_pipeline.items():
        pooled = results["overall"].iloc[0].to_dict()
        pooled["aggregation"] = "pooled"
        summary_rows.append(pooled)
        macro = results["macro"].iloc[0].to_dict()
        macro["aggregation"] = "macro_8_models"
        summary_rows.append(macro)
        fold_frames.append(results["by_fold"])

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(COMPARISON_DIR / "pipeline_summary.csv", index=False)
    by_fold = pd.concat(fold_frames, ignore_index=True)
    by_fold.to_csv(COMPARISON_DIR / "metrics_by_fold.csv", index=False)

    fold_names = [split.name for split in VALIDATION_SPLITS]
    x = np.arange(len(fold_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, (pipeline_name, results) in enumerate(results_by_pipeline.items()):
        values = (
            results["by_fold"].set_index("fold")
            .loc[fold_names, "balanced_accuracy"].to_numpy()
        )
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=PIPELINE_DISPLAY_NAMES.get(pipeline_name, pipeline_name),
        )
    fold_labels = [name.replace("_", " ").title() for name in fold_names]
    ax.set_xticks(x, fold_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Pipelinevergleich je Fold (Kanäle aggregiert)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "balanced_accuracy_by_fold.png", dpi=170)
    plt.close(fig)


def main(force_extract: bool = False, pipeline: str = "both") -> None:
    per_measurement_df = load_or_extract_features(force_extract)
    results_by_pipeline = {}

    if pipeline in {"both", "per_measurement"}:
        results_by_pipeline["per_measurement"] = run_pipeline(
            per_measurement_df, PER_MEASUREMENT_DIR, "per_measurement"
        )

    if pipeline in {"both", "mid_averaged"}:
        mid_averaged_df = average_features_over_mid(per_measurement_df)
        MID_AVERAGED_DIR.mkdir(parents=True, exist_ok=True)
        mid_averaged_df.to_csv(MID_AVERAGED_DIR / "features_mid_averaged.csv", index=False)
        results_by_pipeline["mid_averaged"] = run_pipeline(
            mid_averaged_df, MID_AVERAGED_DIR, "mid_averaged"
        )

    if pipeline == "both":
        create_pipeline_comparison(results_by_pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4: Ein-Klassen-Klassifikation")
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Audiofeatures auch bei vorhandenem CSV neu extrahieren",
    )
    parser.add_argument(
        "--pipeline", choices=("both", "per_measurement", "mid_averaged"),
        default="both", help="Auszufuehrende Ergebnispipeline (Standard: beide)",
    )
    args = parser.parse_args()
    main(force_extract=args.force_extract, pipeline=args.pipeline)
