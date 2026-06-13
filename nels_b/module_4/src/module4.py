"""Module 4 — Bike rental demand prediction."""

import os

# Must be set BEFORE tensorflow is imported — BLAS/oneDNN read these at import time.
_PHYSICAL_CORES = "8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _PHYSICAL_CORES)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_URL = "https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv"
REF_DATE = pd.Timestamp("2000-01-01")
RANDOM_STATE = 42


PHYSICAL_CORES = 8  # i9-9980HK: 8 physical, 16 logical. Hyperthreading hurts FMA-heavy ops.


def configure_cpu_parallelism(n_cores: int = PHYSICAL_CORES) -> None:
    """Pin TF + BLAS to physical cores so threads don't oversubscribe via hyperthreading."""
    tf.config.threading.set_intra_op_parallelism_threads(n_cores)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n_cores))


def load_and_engineer_features(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    df["hour_sin"] = np.sin(df["hr"] * (2.0 * np.pi / 24))
    df["hour_cos"] = np.cos(df["hr"] * (2.0 * np.pi / 24))

    df["season_sin"] = np.sin(df["season"] * (2.0 * np.pi / 4))
    df["season_cos"] = np.cos(df["season"] * (2.0 * np.pi / 4))

    df["dteday"] = pd.to_datetime(df["dteday"])
    df["day"] = df["dteday"].dt.day
    df["month"] = df["dteday"].dt.month
    df["year"] = df["dteday"].dt.year

    # Day-of-week (cyclic) and an explicit working-day flag: weekday AND not a holiday.
    df["dow"] = df["dteday"].dt.dayofweek
    df["dow_sin"] = np.sin(df["dow"] * (2.0 * np.pi / 7))
    df["dow_cos"] = np.cos(df["dow"] * (2.0 * np.pi / 7))
    df["workingday"] = ((df["dow"] < 5) & (df["holiday"] == 0)).astype(int)

    # Hour×working-day interactions: lets the model separate the bimodal commute peaks
    # on working days from the single midday hump on weekends/holidays.
    df["hour_sin_work"] = df["hour_sin"] * df["workingday"]
    df["hour_cos_work"] = df["hour_cos"] * df["workingday"]

    df["total_hours"] = (df["dteday"] - REF_DATE).dt.total_seconds() / 3600 + df["hr"]
    df["time_sin"] = np.sin(df["total_hours"] * (2.0 * np.pi / 24))
    df["time_cos"] = np.cos(df["total_hours"] * (2.0 * np.pi / 24))

    p = 2
    df["weather_intensity"] = (
        (df["temp_c"] ** p + df["hum"] ** p + df["windspeed"] ** p + df["weathersit"] ** p) / 3
    ) ** (1 / p)

    # No scaling here — features are min-max scaled once, downstream in prepare_data.
    return df


def plot_user_histograms(df: pd.DataFrame) -> None:
    daily = df.groupby("dteday")[["casual", "registered"]].sum().reset_index()
    daily["total"] = daily["casual"] + daily["registered"]

    specs = [
        ("total", "Total Users Per Day", "black"),
        ("casual", "Daily Casual Users", "skyblue"),
        ("registered", "Daily Registered Users", "lightcoral"),
    ]
    for col, title, color in specs:
        plt.figure(figsize=(10, 6))
        plt.hist(daily[col], bins=50, edgecolor="black", color=color)
        plt.title(f"Histogram of {title}")
        plt.xlabel(f"Number of {title}")
        plt.ylabel("Number of Days")
        plt.grid(axis="y", alpha=0.75)
        plt.show()


def build_model(
    n_features: int,
    widths: tuple[int, ...] = (512, 256, 128, 64),
    dropout: float = 0.2,
    learning_rate: float = 0.01,
) -> tf.keras.Sequential:
    """Swish MLP with batch norm, Huber loss for outlier robustness."""
    layers: list = [tf.keras.Input(shape=(n_features,))]
    for i, w in enumerate(widths):
        layers.append(tf.keras.layers.Dense(w, activation="swish"))
        if i < len(widths) - 1:
            layers.append(tf.keras.layers.BatchNormalization())
            if i < 2:
                layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(2, activation="linear"))

    model = tf.keras.Sequential(layers)
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae", "mse"],
    )
    return model


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    """Parallel input pipeline: prefetched, optionally shuffled, multi-threaded mapping."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plot_history(history: tf.keras.callbacks.History, title: str) -> None:
    hist = pd.DataFrame(history.history).reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(hist["index"], hist["loss"], label="Train loss")
    plt.plot(hist["index"], hist["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true: pd.DataFrame, y_pred: np.ndarray) -> None:
    pred = pd.DataFrame(y_pred, columns=["pred_casual", "pred_registered"])
    pred["actual_casual"] = y_true["casual"].reset_index(drop=True)
    pred["actual_registered"] = y_true["registered"].reset_index(drop=True)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, actual_col, pred_col, label in zip(
        axes,
        ["actual_casual", "actual_registered"],
        ["pred_casual", "pred_registered"],
        ["Casual Rentals", "Registered Rentals"],
    ):
        mx = max(pred[actual_col].max(), pred[pred_col].max()) * 1.05 or 1.0
        ax.scatter(pred[actual_col], pred[pred_col], alpha=0.3, s=10)
        ax.plot([0, mx], [0, mx], "r--")
        ax.set_xlabel(f"Actual {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(label)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)

    plt.tight_layout()
    plt.show()


def ensemble_predict(results: list[dict], weighted: bool = True) -> np.ndarray:
    """Combine models by averaging their test predictions.

    weighted=True weights each model by 1/RMSE (better models count more);
    weighted=False is a plain mean. Returns rounded, non-negative rentals.
    """
    preds = np.stack([r["y_pred"] for r in results])
    if weighted:
        w = np.array([1.0 / r["rmse"] for r in results])
        w /= w.sum()
        combined = np.tensordot(w, preds, axes=([0], [0]))
    else:
        combined = preds.mean(axis=0)
    return np.clip(np.round(combined, 1), 0, None)


def plot_sweep(results: list[dict], y_test: pd.DataFrame) -> None:
    """Compare swept models: RMSE/R² bars, val-loss curves, and best-model fit."""
    names = [r["name"] for r in results]

    fig, (ax_rmse, ax_loss) = plt.subplots(1, 2, figsize=(15, 5))

    x = np.arange(len(names))
    ax_rmse.bar(x, [r["rmse"] for r in results], color="steelblue")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(names, rotation=30, ha="right")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE by model (lower is better)")
    for xi, r in zip(x, results):
        ax_rmse.text(xi, r["rmse"], f"R²={r['r2']:.3f}", ha="center", va="bottom", fontsize=8)

    for r in results:
        ax_loss.plot(r["history"]["val_loss"], label=r["name"])
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val Huber loss")
    ax_loss.set_title("Validation loss")
    ax_loss.legend()

    plt.tight_layout()
    plt.show()

    best = results[0]
    print(f"\nBest model: {best['name']} (RMSE={best['rmse']:.3f}, R²={best['r2']:.4f})")
    plot_actual_vs_predicted(y_test, best["y_pred"])


def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray) -> None:
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nRMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


def invert_targets(y_scaler: MinMaxScaler, pred_scaled: np.ndarray) -> np.ndarray:
    """Map scaled-log model outputs back to non-negative rounded counts.

    Targets are trained as MinMax(log1p(counts)); this reverses both steps.
    """
    return np.clip(np.round(np.expm1(y_scaler.inverse_transform(pred_scaled)), 1), 0, None)


def prepare_data():
    data = load_and_engineer_features(DATA_URL)
    drop_cols = ["dteday", "hr", "dow", "casual", "registered"]
    y = data[["casual", "registered"]]
    x = data.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_STATE
    )

    x_scaler = MinMaxScaler().fit(X_train)
    X_train_arr = x_scaler.transform(X_train)
    X_test_arr = x_scaler.transform(X_test)

    # Train on log1p(counts): compresses the right-skewed target and tames the
    # heteroscedastic error that grows with demand. y_test stays raw for evaluation.
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(np.log1p(y_train))

    return X_train_arr, X_test_arr, y_test, y_scaler, y_train_scaled


class TqdmEpochBar(tf.keras.callbacks.Callback):
    """One tqdm progress bar per model, advancing one step per epoch."""

    def __init__(self, total_epochs: int, position: int, desc: str, early_stop, reduce_lr):
        super().__init__()
        self.total_epochs = total_epochs
        self.position = position
        self.desc = desc
        self.early_stop = early_stop
        self.reduce_lr = reduce_lr
        self.bar = None

    def on_train_begin(self, logs=None):
        from tqdm.auto import tqdm

        self.bar = tqdm(
            total=self.total_epochs,
            position=self.position,
            desc=self.desc,
            leave=True,
            dynamic_ncols=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # EarlyStopping/ReduceLROnPlateau run before this callback, so their
        # `wait` counters already reflect the epoch just finished.
        stop_in = max(self.early_stop.patience - self.early_stop.wait, 0)
        lr_in = max(self.reduce_lr.patience - self.reduce_lr.wait, 0)
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.bar.set_postfix(
            val_loss=f"{logs.get('val_loss', 0):.4f}",
            lr=f"{lr:.2e}",
            stop_in=stop_in,
            lr_in=lr_in,
        )
        self.bar.update(1)

    def on_train_end(self, logs=None):
        if self.bar is not None:
            # Pin the bar in place and recolor it green instead of letting close()
            # reclaim the line (which shuffles the other workers' bars).
            self.bar.colour = "green"
            self.bar.total = self.bar.n  # show 100% at whatever epoch it stopped
            self.bar.set_description_str(f"\033[92m{self.desc}\033[0m")
            self.bar.refresh()
            self.bar.leave = True
            self.bar.close()


def train_one(
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test: np.ndarray,
    y_test: pd.DataFrame,
    y_scaler: MinMaxScaler,
    config: dict,
    threads_per_worker: int = 8,
    verbose: int = 0,
    position: int = 0,
    desc: str = "model",
    save_dir: str | None = None,
    name: str = "model",
) -> dict:
    """Train a single model with given hyperparameters. Top-level for joblib pickling."""
    tf.config.threading.set_intra_op_parallelism_threads(threads_per_worker)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    batch_size = config["batch_size"]
    val_size = int(len(X_train) * 0.2)
    train_ds = make_dataset(
        X_train[val_size:], y_train_scaled[val_size:], batch_size, shuffle=True
    )
    val_ds = make_dataset(X_train[:val_size], y_train_scaled[:val_size], batch_size, shuffle=False)

    model = build_model(
        X_train.shape[1],
        widths=config["widths"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
    )
    epochs = config.get("epochs", 1000)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            early_stop,
            reduce_lr,
            TqdmEpochBar(epochs, position, desc, early_stop, reduce_lr),
        ],
        verbose=verbose,  # type: ignore[arg-type]
    )

    pred = invert_targets(y_scaler, model.predict(X_test, verbose=0))
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    model_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{name}.keras")
        model.save(model_path)

    return {
        "config": config,
        "name": name,
        "rmse": float(rmse),
        "r2": float(r2),
        "epochs_trained": len(history.history["loss"]),
        "history": history.history,
        "y_pred": pred,
        "model_path": model_path,
    }


DEFAULT_SWEEP = {
    "n_jobs": 4,
    "effort": 5,
    "epochs": 1000,
    "base_lr": 0.005,  # lr at ref_batch
    "ref_batch": 1024,
    "save_dir": None,  # set to a folder to write each trained model as <name>.keras
    "visualize": True,  # plot metrics, val-loss curves, and best-model fit after the sweep
    "models": [
        {"batch_size": 256, "dropout": 0.2},
        {"batch_size": 512, "dropout": 0.25},
        {"batch_size": 1024, "dropout": 0.2},
        {"batch_size": 2048, "dropout": 0.3},
    ],
}


def load_sweep_config(path: str | None) -> dict:
    """Merge a TOML config file over the built-in defaults. Missing keys fall back."""
    cfg = dict(DEFAULT_SWEEP)
    if path:
        import tomllib

        with open(path, "rb") as f:
            cfg.update(tomllib.load(f))
    return cfg


def make_configs(
    effort: int,
    models: list[dict],
    epochs: int = 1000,
    base_lr: float = 0.005,
    ref_batch: int = 1024,
) -> list[dict]:
    """Different builds, equal per-epoch compute.

    Per-epoch cost ≈ (params per step) × (steps per epoch) ∝ step_compute / batch_size,
    and step_compute ∝ u² for the (16u, 8u, 4u, u) shape. Fixing the ratio gives
    u ∝ √batch_size: a model with a smaller batch gets proportionally fewer parameters,
    so every config — though a distinct architecture — takes the same wall-clock per epoch.

    `effort` (1–10) sets that shared compute level: 1 is tiny and very fast, 10 is large
    and ~half the speed of the old model 4. Each entry in `models` is a build; `widths`
    and `learning_rate` are derived from `batch_size` unless given explicitly.
    """
    effort = max(1, min(10, effort))
    # Width unit at the reference batch. u0=64 reproduces old model 4's compute;
    # effort 10 → u0=128 ⇒ ~2× its epoch time (i.e. half the speed).
    u0 = 8 * 16 ** ((effort - 1) / 9)

    configs = []
    for m in models:
        batch = m["batch_size"]
        if "widths" in m:
            widths = tuple(m["widths"])
        else:
            u = max(8, int(round(u0 * (batch / ref_batch) ** 0.5)))
            widths = (16 * u, 8 * u, 4 * u, u)
        # lr scales WITH batch (linear scaling rule): a bigger batch does fewer, lower-noise
        # updates per epoch, so it needs a proportionally larger step to learn at the same rate.
        lr = m.get("learning_rate", base_lr * (batch / ref_batch))
        configs.append(
            {
                "widths": widths,
                "batch_size": batch,
                "learning_rate": lr,
                "dropout": m.get("dropout", 0.2),
                "epochs": epochs,
            }
        )
    return configs


def run_sweep(config: dict | None = None) -> None:
    """Parallel hyperparameter sweep. Each worker gets PHYSICAL_CORES // n_jobs threads."""
    from joblib import Parallel, delayed

    cfg = config or DEFAULT_SWEEP
    n_jobs = cfg["n_jobs"]
    effort = cfg["effort"]
    save_dir = cfg.get("save_dir")

    X_train, X_test, y_test, y_scaler, y_train_scaled = prepare_data()
    threads_per_worker = max(1, PHYSICAL_CORES // n_jobs)

    configs = make_configs(
        effort,
        cfg["models"],
        epochs=cfg.get("epochs", 1000),
        base_lr=cfg.get("base_lr", 0.005),
        ref_batch=cfg.get("ref_batch", 1024),
    )

    print(f"Running {len(configs)} configs (effort={effort}) "
          f"with {n_jobs} workers × {threads_per_worker} threads each")

    raw = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(train_one)(
            X_train,
            y_train_scaled,
            X_test,
            y_test,
            y_scaler,
            model_cfg,
            threads_per_worker,
            position=i,
            desc=f"model {i + 1}/{len(configs)}",
            save_dir=save_dir,
            name=f"model{i + 1}_b{model_cfg['batch_size']}",
        )
        for i, model_cfg in enumerate(configs)
    )
    results = sorted([r for r in (raw or []) if r is not None], key=lambda r: r["rmse"])

    print("\n=== Sweep Results (sorted by RMSE) ===")
    for r in results:
        loc = f"  -> {r['model_path']}" if r.get("model_path") else ""
        print(
            f"{r['name']:18s} RMSE={r['rmse']:8.3f}  R²={r['r2']:.4f}  "
            f"epochs={r['epochs_trained']:3d}{loc}"
        )

    if len(results) > 1:
        for label, weighted in (("ensemble (mean)", False), ("ensemble (1/RMSE)", True)):
            pred = ensemble_predict(results, weighted=weighted)
            rmse = root_mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            print(f"{label:18s} RMSE={rmse:8.3f}  R²={r2:.4f}")

    if cfg.get("visualize", True) and results:
        plot_sweep(results, y_test)


def main() -> None:
    configure_cpu_parallelism()
    X_train, X_test, y_test, y_scaler, y_train_scaled = prepare_data()
    plot_user_histograms(load_and_engineer_features(DATA_URL))

    config = {
        "widths": (512, 256, 128, 64),
        "dropout": 0.2,
        "learning_rate": 0.01,
        "batch_size": 1024,  # Bigger batches → better CPU saturation
    }

    val_size = int(len(X_train) * 0.2)
    train_ds = make_dataset(
        X_train[val_size:], y_train_scaled[val_size:], config["batch_size"], shuffle=True
    )
    val_ds = make_dataset(
        X_train[:val_size], y_train_scaled[:val_size], config["batch_size"], shuffle=False
    )

    model = build_model(
        X_train.shape[1], config["widths"], config["dropout"], config["learning_rate"]
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1000,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=1,
    )

    plot_history(history, "Training History")
    pred = invert_targets(y_scaler, model.predict(X_test))
    evaluate(y_test, pred)
    plot_actual_vs_predicted(y_test, pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run parallel hyperparameter sweep")
    parser.add_argument("--config", help="TOML config file for --sweep")
    parser.add_argument("--jobs", type=int, help="Parallel workers (overrides config)")
    parser.add_argument(
        "--effort", type=int, help="Model size/speed 1=tiny/fast .. 10=large (overrides config)"
    )
    parser.add_argument("--save-dir", help="Folder to save trained models (overrides config)")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip the post-sweep visualizations"
    )
    args = parser.parse_args()

    if args.sweep:
        cfg = load_sweep_config(args.config)
        if args.jobs is not None:
            cfg["n_jobs"] = args.jobs
        if args.effort is not None:
            cfg["effort"] = args.effort
        if args.save_dir is not None:
            cfg["save_dir"] = args.save_dir
        if args.no_plots:
            cfg["visualize"] = False
        run_sweep(cfg)
    else:
        main()
