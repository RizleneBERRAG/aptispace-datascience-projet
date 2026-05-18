import os, sys
sys.path.append('/workspaces/aptispace-datascience-projet')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROCESSED_DIR = Path("data/processed")

df = pd.read_csv(PROCESSED_DIR / "har_full.csv")
train_df = pd.read_csv(PROCESSED_DIR / "har_train.csv")
test_df = pd.read_csv(PROCESSED_DIR / "har_test.csv")
activity_labels = pd.read_csv(PROCESSED_DIR / "activity_labels.csv")

print("Données complètes :", df.shape)
print("Train :", train_df.shape)
print("Test :", test_df.shape)
df.head()

print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", df.shape[1])
print("Nombre d'activités :", df["activity"].nunique())
print("Nombre de sujets :", df["subject_id"].nunique())

print("\nActivités :")
display(activity_labels)

missing_total = df.isna().sum().sum()
print("Nombre total de valeurs manquantes :", missing_total)

missing_by_col = df.isna().sum()
missing_by_col[missing_by_col > 0].head()

activity_counts = df["activity"].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
activity_counts.plot(kind="bar", ax=ax)
ax.set_title("Répartition globale des activités")
ax.set_xlabel("Activité")
ax.set_ylabel("Nombre d'observations")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

activity_counts

split_counts = df["split"].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
split_counts.plot(kind="bar", ax=ax)
ax.set_title("Répartition train/test")
ax.set_xlabel("Split")
ax.set_ylabel("Nombre d'observations")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()

split_counts

activity_by_split = pd.crosstab(df["activity"], df["split"])

fig, ax = plt.subplots(figsize=(10, 5))
activity_by_split.plot(kind="bar", ax=ax)
ax.set_title("Répartition des activités selon le split")
ax.set_xlabel("Activité")
ax.set_ylabel("Nombre d'observations")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

activity_by_split

dynamic_activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]

df["activity_type"] = np.where(
    df["activity"].isin(dynamic_activities),
    "Dynamique",
    "Statique"
)

type_counts = df["activity_type"].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
type_counts.plot(kind="bar", ax=ax)
ax.set_title("Activités dynamiques vs statiques")
ax.set_xlabel("Type d'activité")
ax.set_ylabel("Nombre d'observations")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()

type_counts

subject_counts = df["subject_id"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
subject_counts.plot(kind="bar", ax=ax)
ax.set_title("Nombre d'observations par sujet")
ax.set_xlabel("Sujet")
ax.set_ylabel("Nombre d'observations")
fig.tight_layout()
plt.show()

subject_counts.describe()

meta_cols = ["split", "subject_id", "activity_id", "activity", "activity_type"]
numeric_cols = [col for col in df.columns if col not in meta_cols]

print("Nombre de variables numériques :", len(numeric_cols))

df[numeric_cols[:10]].describe().T

X = df[numeric_cols].values
y = df["activity"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "activity": y
})

print("Variance expliquée PC1 + PC2 :", pca.explained_variance_ratio_.sum())
pca_df.head()

fig, ax = plt.subplots(figsize=(9, 6))

for activity, group in pca_df.groupby("activity"):
    ax.scatter(group["PC1"], group["PC2"], label=activity, alpha=0.35, s=12)

ax.set_title("Projection PCA des activités")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
fig.tight_layout()
plt.show()

signals_train = np.load(PROCESSED_DIR / "har_signals_train.npz", allow_pickle=True)

X_signals = signals_train["X"]
y_signals = signals_train["y"]
signal_files = signals_train["signal_files"]

print("Shape des signaux train :", X_signals.shape)
print("Shape des labels train :", y_signals.shape)
print("Signaux disponibles :")
for signal in signal_files:
    print("-", signal)

activity_map = dict(zip(activity_labels["activity_id"], activity_labels["activity"]))

# Index du signal total_acc_x dans la liste des 9 signaux
signal_index = list(signal_files).index("total_acc_x_train.txt")

fig, ax = plt.subplots(figsize=(12, 6))

for activity_id, activity_name in activity_map.items():
    idx = np.where(y_signals == activity_id)[0][0]
    ax.plot(X_signals[idx, :, signal_index], label=activity_name, alpha=0.8)

ax.set_title("Exemple du signal total_acc_x par activité")
ax.set_xlabel("Pas de temps")
ax.set_ylabel("Amplitude du signal")
ax.legend()
fig.tight_layout()
plt.show()
