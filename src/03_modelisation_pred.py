import os, sys
sys.path.append('/home/runner/work/aptispace-datascience-projet/aptispace-datascience-projet')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("report/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(PROCESSED_DIR / "har_train.csv")
test_df = pd.read_csv(PROCESSED_DIR / "har_test.csv")
activity_labels = pd.read_csv(PROCESSED_DIR / "activity_labels.csv")

print("Train complet :", train_df.shape)
print("Test complet :", test_df.shape)

display(activity_labels)

def stratified_sample(df, target_col, n_per_class, seed=42):
    samples = []

    for value in sorted(df[target_col].unique()):
        group = df[df[target_col] == value]
        n = min(len(group), n_per_class)
        samples.append(group.sample(n=n, random_state=seed))

    return pd.concat(samples, ignore_index=True)

train_sample = stratified_sample(train_df, "activity_id", n_per_class=250)
test_sample = stratified_sample(test_df, "activity_id", n_per_class=120)

print("Train échantillonné :", train_sample.shape)
print("Test échantillonné :", test_sample.shape)

print("\nColonnes principales disponibles :")
print(train_sample[["split", "subject_id", "activity_id", "activity"]].head())

print("\nRépartition train :")
display(train_sample["activity"].value_counts())

print("\nRépartition test :")
display(test_sample["activity"].value_counts())

meta_cols = ["split", "subject_id", "activity_id", "activity"]

all_feature_cols = [col for col in train_sample.columns if col not in meta_cols]

variances = train_sample[all_feature_cols].var().sort_values(ascending=False)
selected_features = variances.head(50).index.tolist()

X_train = train_sample[selected_features]
y_train = train_sample["activity_id"]

X_test = test_sample[selected_features]
y_test = test_sample["activity_id"]

print("Nombre de variables initiales :", len(all_feature_cols))
print("Nombre de variables sélectionnées :", len(selected_features))
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=300))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),
    "Gaussian Naive Bayes": GaussianNB()
}

list(models.keys())

results = []
trained_models = {}

for model_name, model in models.items():
    print("Entraînement :", model_name)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    trained_models[model_name] = model
    
    results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0)
    })

results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
display(results_df)

fig, ax = plt.subplots(figsize=(8, 5))

results_df.sort_values("f1_macro").plot(
    x="model",
    y="f1_macro",
    kind="barh",
    ax=ax,
    legend=False
)

ax.set_title("Comparaison des modèles Machine Learning")
ax.set_xlabel("F1-score macro")
ax.set_ylabel("Modèle")
fig.tight_layout()
plt.show()

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

y_pred_best = best_model.predict(X_test)

print("Meilleur modèle :", best_model_name)

print(classification_report(
    y_test,
    y_pred_best,
    labels=activity_labels["activity_id"].tolist(),
    target_names=activity_labels["activity"].tolist(),
    zero_division=0
))

labels = activity_labels["activity_id"].tolist()
display_labels = activity_labels["activity"].tolist()

cm = confusion_matrix(y_test, y_pred_best, labels=labels)

fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)
disp.plot(ax=ax, xticks_rotation=45, values_format="d")
ax.set_title(f"Matrice de confusion - {best_model_name}")
fig.tight_layout()

fig.savefig(FIGURES_DIR / "confusion_matrix_best_ml_model.png", dpi=150)
plt.show()

errors_df = test_sample[["subject_id", "activity_id", "activity"]].copy()
errors_df["predicted_activity_id"] = y_pred_best

id_to_activity = dict(zip(activity_labels["activity_id"], activity_labels["activity"]))
errors_df["predicted_activity"] = errors_df["predicted_activity_id"].map(id_to_activity)
errors_df["is_error"] = errors_df["activity_id"] != errors_df["predicted_activity_id"]

errors_by_activity = (
    errors_df
    .groupby("activity")["is_error"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

display(errors_by_activity)

results_df.to_csv(PROCESSED_DIR / "ml_test_results.csv", index=False)

print("Résultats Machine Learning sauvegardés.")
