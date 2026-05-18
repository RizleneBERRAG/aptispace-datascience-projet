from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/UCI HAR Dataset")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

def load_activity_labels():
    labels = pd.read_csv(
        RAW_DIR / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["activity_id", "activity"]
    )
    return labels

def load_features():
    features = pd.read_csv(
        RAW_DIR / "features.txt",
        sep=r"\s+",
        header=None,
        names=["feature_id", "feature_name"]
    )
    features["feature_name"] = make_unique_columns(features["feature_name"].tolist())
    return features

def load_split(split_name, features, activity_labels):
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

    df = pd.concat([subjects, y, X], axis=1)
    df = df.merge(activity_labels, on="activity_id", how="left")
    df["split"] = split_name

    cols_first = ["split", "subject_id", "activity_id", "activity"]
    other_cols = [col for col in df.columns if col not in cols_first]
    df = df[cols_first + other_cols]

    return df

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

def main():
    print("Chargement des labels d'activité...")
    activity_labels = load_activity_labels()
    print(activity_labels)

    print("\nChargement des features...")
    features = load_features()
    print(f"Nombre de variables : {len(features)}")

    print("\nChargement train/test tabulaire...")
    train_df = load_split("train", features, activity_labels)
    test_df = load_split("test", features, activity_labels)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Train shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")
    print(f"Full shape  : {full_df.shape}")

    print("\nSauvegarde des CSV propres...")
    activity_labels.to_csv(PROCESSED_DIR / "activity_labels.csv", index=False)
    features.to_csv(PROCESSED_DIR / "features.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "har_train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "har_test.csv", index=False)
    full_df.to_csv(PROCESSED_DIR / "har_full.csv", index=False)

    print("\nChargement des signaux inertiels...")
    X_train_signals, signal_files = load_inertial_signals("train")
    X_test_signals, _ = load_inertial_signals("test")

    y_train = train_df["activity_id"].values
    y_test = test_df["activity_id"].values

    print(f"Signaux train : {X_train_signals.shape}")
    print(f"Signaux test  : {X_test_signals.shape}")

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

    print("\nPréparation terminée.")
    print("Fichiers créés dans data/processed :")
    for file in sorted(PROCESSED_DIR.glob("har_*")):
        print(f"- {file}")

if __name__ == "__main__":
    main()
