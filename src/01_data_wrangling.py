import os, sys
sys.path.append('/home/runner/work/aptispace-datascience-projet/aptispace-datascience-projet')

from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/UCI HAR Dataset")
PROCESSED_DIR = Path("data/processed")

print("Dossier brut existe :", RAW_DIR.exists())
print("Dossier processed existe :", PROCESSED_DIR.exists())

required_files = [
    RAW_DIR / "activity_labels.txt",
    RAW_DIR / "features.txt",
    RAW_DIR / "train" / "X_train.txt",
    RAW_DIR / "train" / "y_train.txt",
    RAW_DIR / "train" / "subject_train.txt",
    RAW_DIR / "test" / "X_test.txt",
    RAW_DIR / "test" / "y_test.txt",
    RAW_DIR / "test" / "subject_test.txt",
]

for file in required_files:
    print(file, "OK" if file.exists() else "MANQUANT")

activity_labels = pd.read_csv(
    RAW_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["activity_id", "activity"]
)

activity_labels

features = pd.read_csv(
    RAW_DIR / "features.txt",
    sep=r"\s+",
    header=None,
    names=["feature_id", "feature_name"]
)

print("Nombre de variables :", len(features))
features.head()

def make_unique_columns(columns):
    counts = {}
    unique = []

    for col in columns:
        if col in counts:
            counts[col] += 1
            unique.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            unique.append(col)

    return unique

features["feature_name"] = make_unique_columns(features["feature_name"].tolist())

print("Colonnes uniques :", features["feature_name"].is_unique)

def load_split(split_name):
    split_dir = RAW_DIR / split_name

    X = pd.read_csv(
        split_dir / f"X_{split_name}.txt",
        sep=r"\s+",
        header=None
    )
    X.columns = features["feature_name"].tolist()

    y = pd.read_csv(
        split_dir / f"y_{split_name}.txt",
        sep=r"\s+",
        header=None,
        names=["activity_id"]
    )

    subjects = pd.read_csv(
        split_dir / f"subject_{split_name}.txt",
        sep=r"\s+",
        header=None,
        names=["subject_id"]
    )

    meta = pd.concat([subjects, y], axis=1)
    meta = meta.merge(activity_labels, on="activity_id", how="left")
    meta.insert(0, "split", split_name)

    df = pd.concat([meta, X], axis=1)
    return df

train_df = load_split("train")
test_df = load_split("test")
full_df = pd.concat([train_df, test_df], ignore_index=True)

print("Train shape :", train_df.shape)
print("Test shape  :", test_df.shape)
print("Full shape  :", full_df.shape)

full_df.head()

print("Valeurs manquantes train :", train_df.isna().sum().sum())
print("Valeurs manquantes test  :", test_df.isna().sum().sum())
print("Valeurs manquantes full  :", full_df.isna().sum().sum())

print("\nRépartition des activités :")
full_df["activity"].value_counts()

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

activity_labels.to_csv(PROCESSED_DIR / "activity_labels.csv", index=False)
features.to_csv(PROCESSED_DIR / "features.csv", index=False)
train_df.to_csv(PROCESSED_DIR / "har_train.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "har_test.csv", index=False)
full_df.to_csv(PROCESSED_DIR / "har_full.csv", index=False)

print("Fichiers tabulaires sauvegardés.")

def load_inertial_signals(split_name):
    signal_dir = RAW_DIR / split_name / "Inertial Signals"

    signal_files = [
        f"body_acc_x_{split_name}.txt",
        f"body_acc_y_{split_name}.txt",
        f"body_acc_z_{split_name}.txt",
        f"body_gyro_x_{split_name}.txt",
        f"body_gyro_y_{split_name}.txt",
        f"body_gyro_z_{split_name}.txt",
        f"total_acc_x_{split_name}.txt",
        f"total_acc_y_{split_name}.txt",
        f"total_acc_z_{split_name}.txt",
    ]

    arrays = []

    for file_name in signal_files:
        arr = pd.read_csv(
            signal_dir / file_name,
            sep=r"\s+",
            header=None
        ).values
        arrays.append(arr)

    signals = np.stack(arrays, axis=-1)
    return signals, signal_files

X_train_signals, signal_files = load_inertial_signals("train")
X_test_signals, _ = load_inertial_signals("test")

y_train = train_df["activity_id"].values
y_test = test_df["activity_id"].values

print("Signaux train :", X_train_signals.shape)
print("Signaux test  :", X_test_signals.shape)
print("Nombre de signaux :", len(signal_files))

np.savez_compressed(
    PROCESSED_DIR / "har_signals_train.npz",
    X=X_train_signals,
    y=y_train,
    signal_files=signal_files
)

np.savez_compressed(
    PROCESSED_DIR / "har_signals_test.npz",
    X=X_test_signals,
    y=y_test,
    signal_files=signal_files
)

print("Fichiers signaux sauvegardés.")
