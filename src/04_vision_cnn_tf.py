import os, sys
sys.path.append('/home/runner/work/aptispace-datascience-projet/aptispace-datascience-projet')

from pathlib import Path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("report/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(42)

print("TensorFlow version :", tf.__version__)

train_signals = np.load(PROCESSED_DIR / "har_signals_train.npz", allow_pickle=True)
test_signals = np.load(PROCESSED_DIR / "har_signals_test.npz", allow_pickle=True)

X_train_full = train_signals["X"].astype("float32")
y_train_full = train_signals["y"].astype("int64") - 1

X_test_full = test_signals["X"].astype("float32")
y_test_full = test_signals["y"].astype("int64") - 1

train_df = pd.read_csv(PROCESSED_DIR / "har_train.csv")
activity_labels = pd.read_csv(PROCESSED_DIR / "activity_labels.csv")

print("X_train_full :", X_train_full.shape)
print("X_test_full  :", X_test_full.shape)

display(activity_labels)

def stratified_sample_indices(y, max_total, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_total // len(classes))
    indices = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        n = min(per_class, len(cls_indices))
        indices.extend(rng.choice(cls_indices, size=n, replace=False))

    return np.array(indices)

groups = train_df["subject_id"].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(splitter.split(X_train_full, y_train_full, groups=groups))

train_sample_idx = train_idx[stratified_sample_indices(y_train_full[train_idx], 1200)]
val_sample_idx = val_idx[stratified_sample_indices(y_train_full[val_idx], 300)]
test_sample_idx = stratified_sample_indices(y_test_full, 600)

X_train = X_train_full[train_sample_idx]
y_train = y_train_full[train_sample_idx]

X_val = X_train_full[val_sample_idx]
y_val = y_train_full[val_sample_idx]

X_test = X_test_full[test_sample_idx]
y_test = y_test_full[test_sample_idx]

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

n_timesteps = X_train.shape[1]
n_channels = X_train.shape[2]
n_classes = len(activity_labels)

model = models.Sequential([
    layers.Input(shape=(n_timesteps, n_channels)),

    layers.Conv1D(filters=16, kernel_size=5, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),

    layers.Dense(32, activation="relu"),
    layers.Dense(n_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=128,
    callbacks=[early_stop],
    verbose=0
)

history_df = pd.DataFrame(history.history)
display(history_df)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_df["loss"], label="Train loss")
ax.plot(history_df["val_loss"], label="Validation loss")
ax.set_title("Évolution de la loss")
ax.set_xlabel("Époque")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_df["accuracy"], label="Train accuracy")
ax.plot(history_df["val_accuracy"], label="Validation accuracy")
ax.set_title("Évolution de l'accuracy")
ax.set_xlabel("Époque")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

y_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

dl_results = {
    "model": "CNN 1D",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "test_loss": test_loss
}

dl_results_df = pd.DataFrame([dl_results])
display(dl_results_df)

target_names = activity_labels["activity"].tolist()

print(classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)
disp.plot(ax=ax, xticks_rotation=45, values_format="d")
ax.set_title("Matrice de confusion - CNN 1D")
fig.tight_layout()

fig.savefig(FIGURES_DIR / "confusion_matrix_cnn_1d.png", dpi=150)
plt.show()

ml_results_path = PROCESSED_DIR / "ml_test_results.csv"

if ml_results_path.exists():
    ml_results_df = pd.read_csv(ml_results_path)

    comparison_df = pd.concat([
        ml_results_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]],
        dl_results_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    ], ignore_index=True)

    comparison_df = comparison_df.sort_values("f1_macro", ascending=False)
    display(comparison_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    comparison_df.sort_values("f1_macro").plot(
        x="model",
        y="f1_macro",
        kind="barh",
        ax=ax,
        legend=False
    )
    ax.set_title("Comparaison Machine Learning vs CNN 1D")
    ax.set_xlabel("F1-score macro")
    ax.set_ylabel("Modèle")
    fig.tight_layout()
    plt.show()
else:
    print("Résultats ML non trouvés.")

dl_results_df.to_csv(PROCESSED_DIR / "dl_test_results.csv", index=False)

print("Résultats Deep Learning sauvegardés.")
