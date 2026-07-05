from itertools import product
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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
    "path",
    "fn",
    "sig",
    "fs",
    "spec",
    "pos",
    "mID",
    "time",
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
    return calculate_metrics_from_predictions(y, pd.Series(y_pred, index=y.index), score_damaged)


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


def evaluate_routed_channel_models(
    models_by_channel: dict[str, Pipeline],
    df: pd.DataFrame,
    feature_columns: list[str],
    results_dir: Path,
) -> dict:
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if test_df.empty:
        raise ValueError("Testsatz ist leer.")

    x_test, y_test = make_xy(test_df, feature_columns)
    y_pred = pd.Series(index=test_df.index, dtype=int)
    score_damaged = pd.Series(index=test_df.index, dtype=float)

    for channel, channel_df in test_df.groupby("sID"):
        if channel not in models_by_channel:
            raise ValueError(f"Kein Modell fuer Channel {channel} vorhanden.")

        channel_x = x_test.loc[channel_df.index]
        channel_model = models_by_channel[channel]
        y_pred.loc[channel_df.index] = channel_model.predict(channel_x).astype(int)
        score_damaged.loc[channel_df.index] = _probability_for_class(
            channel_model,
            channel_x,
            class_label=0,
        )

    metrics = calculate_metrics_from_predictions(y_test, y_pred.astype(int), score_damaged)
    pd.DataFrame([metrics]).to_csv(results_dir / "final_test_metrics.csv", index=False)

    predictions = test_df[
        ["path", "fn", "spec", "pos", "mID", "time", "rID", "sID", "label"]
    ].copy()
    predictions["predicted_label"] = y_pred.astype(int).to_numpy()
    predictions["score_damaged_0"] = score_damaged.to_numpy()
    predictions.to_csv(results_dir / "routed_channel_test_predictions.csv", index=False)

    plot_confusion_matrix_from_predictions(
        y_test,
        y_pred.astype(int),
        results_dir / "confusion_matrix_test.png",
    )
    plot_roc_curve_from_scores(
        y_test,
        score_damaged,
        results_dir / "roc_curve_test.png",
    )

    return metrics


def create_learning_curve(
    df: pd.DataFrame,
    feature_columns: list[str],
    best_params: dict,
    output_csv: Path,
    output_png: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    dev_df = df[df[SPLIT_COLUMN] == "dev"].copy()
    if train_df.empty or dev_df.empty:
        raise ValueError("Training- und Entwicklungssatz werden fuer die Learning Curve benoetigt.")

    x_dev, y_dev = make_xy(dev_df, feature_columns)
    train_sizes = np.linspace(0.2, 1.0, 5)
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        for train_fraction in train_sizes:
            train_subset = _stratified_train_subset(
                train_df,
                train_fraction,
                random_state=random_state,
            )
            x_train, y_train = make_xy(train_subset, feature_columns)

            model = build_mlp_pipeline(**best_params, random_state=random_state)
            model.fit(x_train, y_train)

            train_pred = pd.Series(model.predict(x_train), index=y_train.index)
            dev_pred = pd.Series(model.predict(x_dev), index=y_dev.index)

            rows.append(
                {
                    "train_fraction": train_fraction,
                    "train_samples": len(train_subset),
                    "train_accuracy": accuracy_score(y_train, train_pred),
                    "dev_accuracy": accuracy_score(y_dev, dev_pred),
                    "train_balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
                    "dev_balanced_accuracy": balanced_accuracy_score(y_dev, dev_pred),
                }
            )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_csv, index=False)
    plot_learning_curve(curve_df, output_png)
    return curve_df


def create_decision_boundary_gif(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    title: str,
    random_state: int = 42,
) -> None:
    train_df = df[df[SPLIT_COLUMN].isin(["train", "dev"])].copy()
    x_train, y_train = make_xy(train_df, feature_columns)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=2, random_state=random_state)
    x_2d = pca.fit_transform(x_scaled)

    x_min, x_max = x_2d[:, 0].min() - 1.0, x_2d[:, 0].max() + 1.0
    y_min, y_max = x_2d[:, 1].min() - 1.0, x_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 140),
        np.linspace(y_min, y_max, 140),
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        learning_rate="constant",
        eta0=0.01,
        random_state=random_state,
    )
    classes = np.array([0, 1])
    total_iterations = 80
    snapshots = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for iteration in range(1, total_iterations + 1):
            if iteration == 1:
                model.partial_fit(x_2d, y_train, classes=classes)
            else:
                model.partial_fit(x_2d, y_train)

            if iteration == 1 or iteration % 4 == 0 or iteration == total_iterations:
                score = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                snapshots.append((iteration, score, model.score(x_2d, y_train)))

    y_values = y_train.to_numpy()
    damaged = y_values == 0
    good = y_values == 1
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame_index):
        iteration, z, train_acc = snapshots[frame_index]
        ax.clear()
        max_abs = max(1.0, float(np.max(np.abs(z))))
        ax.contourf(xx, yy, z, levels=np.linspace(-max_abs, max_abs, 11), cmap="RdYlBu_r", alpha=0.24)
        ax.contour(xx, yy, z, levels=[0.0], colors="black", linewidths=2.0)
        ax.scatter(x_2d[good, 0], x_2d[good, 1], c="#1f77b4", s=38, label="gut (1)")
        ax.scatter(x_2d[damaged, 0], x_2d[damaged, 1], c="#d62728", s=38, label="beschaedigt (0)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"{title}: lineare Entscheidungsgrenze\nIteration {iteration}/{total_iterations}, Accuracy {train_acc:.3f}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)

    animation = FuncAnimation(fig, update, frames=len(snapshots), interval=140, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=7))
    plt.close(fig)


def _stratified_train_subset(
    train_df: pd.DataFrame,
    train_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if train_fraction >= 1.0:
        return train_df.copy()

    parts = []
    for _, label_df in train_df.groupby(TARGET_COLUMN):
        sample_count = max(1, int(round(len(label_df) * train_fraction)))
        parts.append(
            label_df.sample(
                n=sample_count,
                random_state=random_state,
                replace=False,
            )
        )

    return pd.concat(parts).sample(frac=1.0, random_state=random_state)


def calculate_metrics_from_predictions(
    y: pd.Series,
    y_pred: pd.Series,
    score_damaged: pd.Series,
) -> dict:
    y_damaged = (y == 0).astype(int)
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])
    tp_damaged = matrix[0, 0]
    fn_damaged = matrix[0, 1]
    fp_damaged = matrix[1, 0]
    tn_damaged = matrix[1, 1]

    specificity_denominator = tn_damaged + fp_damaged
    npv_denominator = tn_damaged + fn_damaged

    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_damaged_0": precision_score(y, y_pred, pos_label=0, zero_division=0),
        "recall_damaged_0": recall_score(y, y_pred, pos_label=0, zero_division=0),
        "specificity_damaged_0": (
            tn_damaged / specificity_denominator if specificity_denominator else 0.0
        ),
        "npv_damaged_0": tn_damaged / npv_denominator if npv_denominator else 0.0,
        "f1_damaged_0": f1_score(y, y_pred, pos_label=0, zero_division=0),
        "precision_good_1": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall_good_1": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "f1_good_1": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y, y_pred),
        "roc_auc_damaged_0": roc_auc_score(y_damaged, score_damaged),
    }


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


def plot_learning_curve(curve_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve_df["train_samples"],
        curve_df["train_accuracy"],
        marker="o",
        linewidth=2.0,
        label="Train Accuracy",
    )
    plt.plot(
        curve_df["train_samples"],
        curve_df["dev_accuracy"],
        marker="o",
        linewidth=2.0,
        label="Dev Accuracy",
    )
    plt.plot(
        curve_df["train_samples"],
        curve_df["train_balanced_accuracy"],
        marker="s",
        linestyle="--",
        linewidth=1.7,
        label="Train Balanced Accuracy",
    )
    plt.plot(
        curve_df["train_samples"],
        curve_df["dev_balanced_accuracy"],
        marker="s",
        linestyle="--",
        linewidth=1.7,
        label="Dev Balanced Accuracy",
    )
    plt.xlabel("Trainingsbeispiele")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.title("Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
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


def plot_confusion_matrix_from_predictions(
    y: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> None:
    labels = [0, 1]
    matrix = confusion_matrix(y, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix Testsatz mit Channel-Routing")
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


def plot_roc_curve_from_scores(
    y: pd.Series,
    score_damaged: pd.Series,
    output_path: Path,
) -> None:
    y_damaged = (y == 0).astype(int)
    fpr, tpr, _ = roc_curve(y_damaged, score_damaged)
    auc_value = roc_auc_score(y_damaged, score_damaged)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Channel-Routing MLP, AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Zufall")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Kurve fuer Klasse 0: beschaedigt")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
