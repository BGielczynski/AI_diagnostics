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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from model import get_model_feature_columns, make_xy, train_final_model


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
SPLIT_CSV = RESULTS_DIR / "features_split.csv"
TOP_MODELS_CSV = RESULTS_DIR / "top_models.csv"
OUTPUT_GIF = RESULTS_DIR / "training_iteration_demo.gif"
OUTPUT_BOUNDARY_GIF = RESULTS_DIR / "training_decision_boundary.gif"
OUTPUT_LOSS_CSV = RESULTS_DIR / "training_loss_values.csv"
OUTPUT_ACCURACY_PNG = RESULTS_DIR / "accuracy_curve_train_dev.png"
OUTPUT_ACCURACY_CSV = RESULTS_DIR / "accuracy_curve_train_dev.csv"


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    cleaned = value.strip().strip("()")
    if not cleaned:
        return tuple()
    return tuple(int(part.strip()) for part in cleaned.split(",") if part.strip())


def load_best_params() -> dict:
    top_models = pd.read_csv(TOP_MODELS_CSV)
    best = top_models.iloc[0]
    return {
        "hidden_layer_sizes": parse_hidden_layers(str(best["hidden_layer_sizes"])),
        "learning_rate_init": float(best["learning_rate_init"]),
        "batch_size": int(best["batch_size"]),
        "alpha": float(best["alpha"]),
    }


def train_best_model():
    split_df = pd.read_csv(SPLIT_CSV)
    feature_columns = get_model_feature_columns(split_df)
    best_params = load_best_params()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = train_final_model(split_df, feature_columns, best_params)

    return model, best_params


def save_loss_values(loss_curve: list[float]) -> None:
    pd.DataFrame(
        {
            "iteration": np.arange(1, len(loss_curve) + 1),
            "loss": loss_curve,
        }
    ).to_csv(OUTPUT_LOSS_CSV, index=False)


def create_training_gif(loss_curve: list[float], best_params: dict) -> None:
    loss = np.asarray(loss_curve, dtype=float)
    iterations = np.arange(1, len(loss) + 1)

    frame_count = min(80, len(loss))
    frame_indices = np.unique(
        np.linspace(1, len(loss), frame_count, dtype=int)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("MLP Training pro Iteration", fontsize=14, y=0.98)
    line, = ax.plot([], [], color="#1f77b4", linewidth=2.5)
    point, = ax.plot([], [], marker="o", color="#d62728", markersize=6)
    info = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_xlim(1, len(loss))
    ax.set_ylim(max(0.0, loss.min() * 0.9), loss.max() * 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    params_text = (
        f"hidden={best_params['hidden_layer_sizes']}, "
        f"lr={best_params['learning_rate_init']}, "
        f"batch={best_params['batch_size']}, "
        f"alpha={best_params['alpha']}"
    )
    ax.text(0.5, 1.01, params_text, transform=ax.transAxes, ha="center", fontsize=9)

    def update(frame_index):
        current_iteration = frame_indices[frame_index]
        x = iterations[:current_iteration]
        y = loss[:current_iteration]

        line.set_data(x, y)
        point.set_data([x[-1]], [y[-1]])
        info.set_text(
            "1. Vorhersage berechnen\n"
            "2. Fehler/Loss bestimmen\n"
            "3. Gewichte per Backpropagation anpassen\n\n"
            f"Iteration: {current_iteration}/{len(loss)}\n"
            f"Aktueller Loss: {y[-1]:.5f}"
        )
        return line, point, info

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=120,
        blit=True,
    )
    animation.save(OUTPUT_GIF, writer=PillowWriter(fps=8))
    plt.close(fig)


def create_decision_boundary_gif(best_params: dict, total_iterations: int) -> None:
    split_df = pd.read_csv(SPLIT_CSV)
    feature_columns = get_model_feature_columns(split_df)
    train_df = split_df[split_df["split"] == "train"].copy()
    x_train, y_train = make_xy(train_df, feature_columns)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    pca = PCA(n_components=2)
    x_train_2d = pca.fit_transform(x_train_scaled)

    x_min, x_max = x_train_2d[:, 0].min() - 1.0, x_train_2d[:, 0].max() + 1.0
    y_min, y_max = x_train_2d[:, 1].min() - 1.0, x_train_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 160),
        np.linspace(y_min, y_max, 160),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled_full = pca.inverse_transform(grid_2d)

    model = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        alpha=best_params["alpha"],
        activation="relu",
        solver="adam",
        max_iter=1,
        random_state=42,
        warm_start=True,
    )

    frame_step = 3
    snapshots = []
    classes = np.array([0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for iteration in range(1, total_iterations + 1):
            if iteration == 1:
                model.partial_fit(x_train_scaled, y_train, classes=classes)
            else:
                model.partial_fit(x_train_scaled, y_train)

            should_save_frame = iteration == 1 or iteration % frame_step == 0 or iteration == total_iterations
            if should_save_frame:
                proba_damaged = model.predict_proba(grid_scaled_full)[:, list(model.classes_).index(0)]
                z = proba_damaged.reshape(xx.shape)
                train_acc = model.score(x_train_scaled, y_train)
                snapshots.append((iteration, z, train_acc))

    fig, ax = plt.subplots(figsize=(8, 6))

    y_array = y_train.to_numpy()
    damaged = y_array == 0
    good = y_array == 1

    def update(frame_index):
        iteration, z, train_acc = snapshots[frame_index]
        ax.clear()

        ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.22)
        ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=2.0)
        ax.scatter(
            x_train_2d[good, 0],
            x_train_2d[good, 1],
            c="#1f77b4",
            edgecolors="white",
            linewidths=0.8,
            label="gut (1)",
            s=42,
        )
        ax.scatter(
            x_train_2d[damaged, 0],
            x_train_2d[damaged, 1],
            c="#d62728",
            edgecolors="white",
            linewidths=0.8,
            label="beschaedigt (0)",
            s=42,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(
            "64D-MLP Entscheidungsgrenze in 2D-PCA-Projektion\n"
            f"Iteration {iteration}/{total_iterations}, Train Accuracy {train_acc:.3f}"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=140,
        blit=False,
    )
    animation.save(OUTPUT_BOUNDARY_GIF, writer=PillowWriter(fps=7))
    plt.close(fig)


def create_accuracy_curve(best_params: dict, total_iterations: int) -> None:
    split_df = pd.read_csv(SPLIT_CSV)
    feature_columns = get_model_feature_columns(split_df)

    train_df = split_df[split_df["split"] == "train"].copy()
    dev_df = split_df[split_df["split"] == "dev"].copy()
    x_train, y_train = make_xy(train_df, feature_columns)
    x_dev, y_dev = make_xy(dev_df, feature_columns)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)

    model = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        alpha=best_params["alpha"],
        activation="relu",
        solver="adam",
        max_iter=1,
        random_state=42,
        warm_start=True,
    )

    rows = []
    classes = np.array([0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for iteration in range(1, total_iterations + 1):
            if iteration == 1:
                model.partial_fit(x_train_scaled, y_train, classes=classes)
            else:
                model.partial_fit(x_train_scaled, y_train)

            rows.append(
                {
                    "iteration": iteration,
                    "train_accuracy": model.score(x_train_scaled, y_train),
                    "dev_accuracy": model.score(x_dev_scaled, y_dev),
                }
            )

    accuracy_df = pd.DataFrame(rows)
    accuracy_df.to_csv(OUTPUT_ACCURACY_CSV, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        accuracy_df["iteration"],
        accuracy_df["train_accuracy"],
        label="Training Accuracy",
        linewidth=2.2,
    )
    plt.plot(
        accuracy_df["iteration"],
        accuracy_df["dev_accuracy"],
        label="Dev Accuracy",
        linewidth=2.2,
    )
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Training- und Entwicklungs-Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_ACCURACY_PNG, dpi=160)
    plt.close()


def main() -> None:
    model, best_params = train_best_model()
    loss_curve = model.named_steps["mlp"].loss_curve_
    save_loss_values(loss_curve)
    create_training_gif(loss_curve, best_params)
    create_decision_boundary_gif(best_params, total_iterations=len(loss_curve))
    create_accuracy_curve(best_params, total_iterations=len(loss_curve))
    print(f"GIF gespeichert: {OUTPUT_GIF}")
    print(f"Entscheidungsgrenzen-GIF gespeichert: {OUTPUT_BOUNDARY_GIF}")
    print(f"Loss-Werte gespeichert: {OUTPUT_LOSS_CSV}")
    print(f"Accuracy-Plot gespeichert: {OUTPUT_ACCURACY_PNG}")
    print(f"Accuracy-Werte gespeichert: {OUTPUT_ACCURACY_CSV}")


if __name__ == "__main__":
    main()
