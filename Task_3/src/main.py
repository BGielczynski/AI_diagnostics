from pathlib import Path

from aggregation import add_channel_one_hot, aggregate_over_mid
from dataframe_manager import DataFrameManager
from extraction import extract_features
from model import (
    evaluate_final_model,
    get_model_feature_columns,
    train_final_model,
    tune_hyperparameters,
)
from split import add_split, summarize_split


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "Task_2" / "data" / "sig"
RESULTS_DIR = ROOT_DIR / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    manager = DataFrameManager(data_dir=str(DATA_DIR))
    manager.load_signals()
    raw_df = manager.get_dataframe()
    print(f"Rohsignale: {len(raw_df)}")

    feature_df = extract_features(raw_df)
    feature_df.to_csv(RESULTS_DIR / "features_per_file.csv", index=False)
    print(f"Feature-Zeilen vor Mittelung: {len(feature_df)}")

    aggregated_df = aggregate_over_mid(feature_df)
    aggregated_df = add_channel_one_hot(aggregated_df)
    aggregated_df.to_csv(RESULTS_DIR / "features_aggregated_mid.csv", index=False)
    print(f"Feature-Zeilen nach mID-Mittelung: {len(aggregated_df)}")

    split_df = add_split(aggregated_df)
    split_df.to_csv(RESULTS_DIR / "features_split.csv", index=False)
    print("Split-Uebersicht:")
    print(summarize_split(split_df).to_string(index=False))

    feature_columns = get_model_feature_columns(split_df)
    print(f"Modell-Features: {len(feature_columns)}")

    best_params, tuning_results = tune_hyperparameters(
        split_df,
        feature_columns,
        RESULTS_DIR,
    )
    print("Beste Hyperparameter:")
    print(best_params)
    print("Beste Entwicklungsmetriken:")
    print(tuning_results.head(1).to_string(index=False))

    final_model = train_final_model(split_df, feature_columns, best_params)
    test_metrics = evaluate_final_model(final_model, split_df, feature_columns, RESULTS_DIR)
    print("Finale Testmetriken:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    main()
