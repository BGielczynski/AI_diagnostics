from itertools import product
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "label"
SPLIT_COLUMN = "split"

NON_MODEL_COLUMNS = {
    "spec",
    "pos",
    "rID",
    "sID",
    "label",
    "split",
    "split_group",
    "mID_count",
    "source_file_count",
}


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in NON_MODEL_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def make_xy(df: pd.DataFrame, feature_columns: list[str]):
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[TARGET_COLUMN].astype(int)
    return x, y


def get_split_xy(df: pd.DataFrame, feature_columns: list[str], split_name: str):
    split_df = df[df[SPLIT_COLUMN] == split_name].copy()
    if split_df.empty:
        raise ValueError(f"Split ist leer: {split_name}")
    return make_xy(split_df, feature_columns)


def build_mlp_pipeline(
    hidden_layer_sizes=(32,),
    learning_rate_init=0.001,
    batch_size=16,
    alpha=0.001,
    random_state=42,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    batch_size=batch_size,
                    alpha=alpha,
                    activation="relu",
                    solver="adam",
                    max_iter=800,
                    random_state=random_state,
                    tol=1e-4,
                    n_iter_no_change=30,
                ),
            ),
        ]
    )


def _probability_for_class(model: Pipeline, x: pd.DataFrame, class_label: int) -> np.ndarray:
    classes = list(model.classes_)
    if class_label not in classes:
        raise ValueError(f"Klasse {class_label} ist im Modell nicht vorhanden: {classes}")
    class_index = classes.index(class_label)
    return model.predict_proba(x)[:, class_index]


def evaluate_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(x)
    score_damaged = _probability_for_class(model, x, class_label=0)
    y_damaged = (y == 0).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_damaged_0": precision_score(y, y_pred, pos_label=0, zero_division=0),
        "recall_damaged_0": recall_score(y, y_pred, pos_label=0, zero_division=0),
        "f1_damaged_0": f1_score(y, y_pred, pos_label=0, zero_division=0),
        "precision_good_1": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall_good_1": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "f1_good_1": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "roc_auc_damaged_0": roc_auc_score(y_damaged, score_damaged),
    }
    return metrics


def tune_hyperparameters(
    df: pd.DataFrame,
    feature_columns: list[str],
    results_dir: Path,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    x_train, y_train = get_split_xy(df, feature_columns, "train")
    x_dev, y_dev = get_split_xy(df, feature_columns, "dev")

    hidden_layer_options = [(16,), (32,), (64,), (32, 16)]
    learning_rate_options = [0.0001, 0.001, 0.01]
    batch_size_options = [8, 16, 32]
    alpha_options = [0.0001, 0.001, 0.01]

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        for hidden_layers, learning_rate, batch_size, alpha in product(
            hidden_layer_options,
            learning_rate_options,
            batch_size_options,
            alpha_options,
        ):
            model = build_mlp_pipeline(
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=learning_rate,
                batch_size=batch_size,
                alpha=alpha,
                random_state=random_state,
            )
            model.fit(x_train, y_train)

            metrics = evaluate_model(model, x_dev, y_dev)
            mlp = model.named_steps["mlp"]
            rows.append(
                {
                    "hidden_layer_sizes": str(hidden_layers),
                    "learning_rate_init": learning_rate,
                    "batch_size": batch_size,
                    "alpha": alpha,
                    "n_iter": mlp.n_iter_,
                    "final_loss": mlp.loss_,
                    **{f"dev_{key}": value for key, value in metrics.items()},
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["dev_roc_auc_damaged_0", "dev_balanced_accuracy", "dev_f1_damaged_0"],
        ascending=False,
    )
    results.to_csv(results_dir / "hyperparameter_results.csv", index=False)
    results.head(5).to_csv(results_dir / "top_models.csv", index=False)

    best = results.iloc[0].to_dict()
    best_params = {
        "hidden_layer_sizes": _parse_hidden_layers(best["hidden_layer_sizes"]),
        "learning_rate_init": float(best["learning_rate_init"]),
        "batch_size": int(best["batch_size"]),
        "alpha": float(best["alpha"]),
    }
    return best_params, results


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    cleaned = value.strip().strip("()")
    if not cleaned:
        return tuple()
    return tuple(int(part.strip()) for part in cleaned.split(",") if part.strip())


def train_final_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    best_params: dict,
    random_state: int = 42,
) -> Pipeline:
    train_dev_df = df[df[SPLIT_COLUMN].isin(["train", "dev"])].copy()
    x_train_dev, y_train_dev = make_xy(train_dev_df, feature_columns)

    model = build_mlp_pipeline(**best_params, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_train_dev, y_train_dev)

    return model


def evaluate_final_model(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    results_dir: Path,
) -> dict:
    x_test, y_test = get_split_xy(df, feature_columns, "test")
    metrics = evaluate_model(model, x_test, y_test)

    pd.DataFrame([metrics]).to_csv(results_dir / "final_test_metrics.csv", index=False)
    plot_loss_curve(model, results_dir / "training_loss_curve.png")
    plot_confusion_matrix(model, x_test, y_test, results_dir / "confusion_matrix_test.png")
    plot_roc_curve(model, x_test, y_test, results_dir / "roc_curve_test.png")

    return metrics


def plot_loss_curve(model: Pipeline, output_path: Path) -> None:
    loss_curve = model.named_steps["mlp"].loss_curve_

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, marker="o", markersize=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Trainingsverlust")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    labels = [0, 1]
    matrix = confusion_matrix(y, model.predict(x), labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix Testsatz")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsaechlich")
    tick_labels = ["beschaedigt (0)", "gut (1)"]
    plt.xticks(np.arange(len(labels)), tick_labels, rotation=20, ha="right")
    plt.yticks(np.arange(len(labels)), tick_labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_curve(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    y_damaged = (y == 0).astype(int)
    score_damaged = _probability_for_class(model, x, class_label=0)
    fpr, tpr, _ = roc_curve(y_damaged, score_damaged)
    auc_value = roc_auc_score(y_damaged, score_damaged)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"MLP, AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Zufall")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Kurve fuer Klasse 0: beschaedigt")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
