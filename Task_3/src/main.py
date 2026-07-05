from pathlib import Path

import pandas as pd

from dataframe_manager import DataFrameManager
from extraction import extract_features
from model import (
    create_decision_boundary_gif,
    create_learning_curve,
    evaluate_routed_channel_models,
    evaluate_final_model,
    get_model_feature_columns,
    train_final_model,
    tune_hyperparameters,
)
from split import add_position_split, summarize_split


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "Task_2" / "data" / "sig"
RESULTS_DIR = ROOT_DIR / "results"
AVERAGE_GROUP_COLUMNS = ["spec", "pos", "rID", "sID", "label"]
RAW_METADATA_COLUMNS = [
    "path",
    "fn",
    "spec",
    "pos",
    "mID",
    "time",
    "rID",
    "sID",
    "label",
    "split",
]


def extract_features_splitwise(raw_split_df):
    feature_frames = []
    for split_name in ["train", "dev", "test"]:
        split_raw_df = raw_split_df[raw_split_df["split"] == split_name].copy()
        split_feature_df = extract_features(split_raw_df)
        split_feature_df["split"] = split_name
        feature_frames.append(split_feature_df)

    return pd.concat(feature_frames, ignore_index=True)


def average_features_over_mid(feature_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = AVERAGE_GROUP_COLUMNS + ["split"]
    feature_columns = get_model_feature_columns(feature_df)
    aggregated = (
        feature_df.groupby(group_columns, as_index=False)
        .agg(
            {
                **{column: "mean" for column in feature_columns},
                "mID": "nunique",
                "fn": "count",
            }
        )
        .rename(columns={"mID": "mID_count", "fn": "source_file_count"})
    )
    aggregated["path"] = "features_averaged_over_mid"
    aggregated["fn"] = (
        aggregated["spec"]
        + "_"
        + aggregated["pos"]
        + "_avgFeatures_"
        + aggregated["rID"]
        + "_"
        + aggregated["sID"]
    )
    aggregated["mID"] = "mean"
    aggregated["time"] = "mean"
    return aggregated


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    manager = DataFrameManager(data_dir=str(DATA_DIR))
    manager.load_signals()
    raw_df = manager.get_dataframe()
    print(f"Rohsignale: {len(raw_df)}")

    raw_split_df = add_position_split(raw_df)
    raw_split_df[RAW_METADATA_COLUMNS].to_csv(RESULTS_DIR / "raw_signal_split.csv", index=False)
    print("Rohsignal-Split-Uebersicht:")
    print(summarize_split(raw_split_df).to_string(index=False))

    per_file_feature_df = extract_features_splitwise(raw_split_df)
    feature_df = average_features_over_mid(per_file_feature_df)
    feature_df.to_csv(RESULTS_DIR / "features_split.csv", index=False)
    print(f"Feature-Zeilen vor mID-Mittelung: {len(per_file_feature_df)}")
    print(f"Feature-Zeilen nach Feature-Mittelung: {len(feature_df)}")

    feature_columns = get_model_feature_columns(feature_df)
    print(f"Modell-Features: {len(feature_columns)}")

    models_by_channel = {}
    for channel in sorted(feature_df["sID"].unique()):
        channel_dir = RESULTS_DIR / f"model_{channel}"
        channel_dir.mkdir(exist_ok=True)
        channel_df = feature_df[feature_df["sID"] == channel].copy()

        print(f"\nChannel {channel}: {len(channel_df)} Feature-Zeilen")
        print(summarize_split(channel_df).to_string(index=False))

        best_params, tuning_results = tune_hyperparameters(
            channel_df,
            feature_columns,
            channel_dir,
        )
        print(f"Beste Hyperparameter {channel}:")
        print(best_params)
        print(f"Beste Entwicklungsmetriken {channel}:")
        print(tuning_results.head(1).to_string(index=False))

        final_model = train_final_model(channel_df, feature_columns, best_params)
        models_by_channel[channel] = final_model
        channel_metrics = evaluate_final_model(
            final_model,
            channel_df,
            feature_columns,
            channel_dir,
        )
        print(f"Finale Testmetriken {channel}:")
        for metric_name, metric_value in channel_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        learning_curve = create_learning_curve(
            channel_df,
            feature_columns,
            best_params,
            channel_dir / "learning_curve.csv",
            channel_dir / "learning_curve.png",
        )
        print(f"Learning Curve {channel}:")
        print(learning_curve.to_string(index=False))

        create_decision_boundary_gif(
            channel_df,
            feature_columns,
            channel_dir / "training_decision_boundary.gif",
            title=f"Channel {channel}",
        )

    routed_metrics = evaluate_routed_channel_models(
        models_by_channel,
        feature_df,
        feature_columns,
        RESULTS_DIR,
    )
    print("\nFinale Testmetriken mit Channel-Routing:")
    for metric_name, metric_value in routed_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    main()
