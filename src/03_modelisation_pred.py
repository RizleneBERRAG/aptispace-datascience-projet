import os, sys
sys.path.append('/workspaces/aptispace-datascience-projet')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
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

print("Train :", train_df.shape)
print("Test :", test_df.shape)

display(activity_labels)

meta_cols = ["split", "subject_id", "activity_id", "activity"]

feature_cols = [col for col in train_df.columns if col not in meta_cols]

X_train = train_df[feature_cols]
y_train = train_df["activity_id"]

X_test = test_df[feature_cols]
y_test = test_df["activity_id"]

groups_train = train_df["subject_id"]

print("Nombre de variables :", X_train.shape[1])
print("Nombre de classes :", y_train.nunique())
print("Taille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, n_jobs=-1))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    ),
    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(max_iter=5000, random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}

list(models.keys())

cv = GroupKFold(n_splits=3)

scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro"
}

cv_results = []

for model_name, model in models.items():
    print(f"Validation croisée : {model_name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        "model": model_name,
        "mean_accuracy": scores["test_accuracy"].mean(),
        "std_accuracy": scores["test_accuracy"].std(),
        "mean_f1_macro": scores["test_f1_macro"].mean(),
        "std_f1_macro": scores["test_f1_macro"].std()
    })

cv_results_df = pd.DataFrame(cv_results).sort_values(
    by="mean_f1_macro",
    ascending=False
)

display(cv_results_df)

fig, ax = plt.subplots(figsize=(9, 5))

cv_results_df.sort_values("mean_f1_macro").plot(
    x="model",
    y="mean_f1_macro",
    kind="barh",
    ax=ax,
    legend=False
)

ax.set_title("Comparaison des modèles - F1-score macro")
ax.set_xlabel("F1-score macro moyen")
ax.set_ylabel("Modèle")
fig.tight_layout()
plt.show()

test_results = []
trained_models = {}

for model_name, model in models.items():
    print(f"Entraînement final : {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    trained_models[model_name] = model

    test_results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0)
    })

test_results_df = pd.DataFrame(test_results).sort_values(
    by="f1_macro",
    ascending=False
)

display(test_results_df)

fig, ax = plt.subplots(figsize=(9, 5))

test_results_df.sort_values("f1_macro").plot(
    x="model",
    y="f1_macro",
    kind="barh",
    ax=ax,
    legend=False
)

ax.set_title("Comparaison des modèles sur le test - F1-score macro")
ax.set_xlabel("F1-score macro")
ax.set_ylabel("Modèle")
fig.tight_layout()
plt.show()

best_model_name = test_results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("Meilleur modèle :", best_model_name)

y_pred_best = best_model.predict(X_test)

print("\nRapport de classification :")
print(
    classification_report(
        y_test,
        y_pred_best,
        labels=activity_labels["activity_id"].tolist(),
        target_names=activity_labels["activity"].tolist(),
        zero_division=0
    )
)

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

errors_df = test_df[["subject_id", "activity_id", "activity"]].copy()
errors_df["predicted_activity_id"] = y_pred_best

id_to_activity = dict(zip(activity_labels["activity_id"], activity_labels["activity"]))
errors_df["predicted_activity"] = errors_df["predicted_activity_id"].map(id_to_activity)
errors_df["is_error"] = errors_df["activity_id"] != errors_df["predicted_activity_id"]

error_rate = errors_df["is_error"].mean()

print("Taux d'erreur :", round(error_rate, 4))

errors_by_activity = (
    errors_df
    .groupby("activity")["is_error"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

display(errors_by_activity)

fig, ax = plt.subplots(figsize=(9, 5))

errors_by_activity.sort_values("is_error").plot(
    x="activity",
    y="is_error",
    kind="barh",
    ax=ax,
    legend=False
)

ax.set_title("Taux d'erreur par activité")
ax.set_xlabel("Taux d'erreur")
ax.set_ylabel("Activité")
fig.tight_layout()
plt.show()

cv_results_df.to_csv(PROCESSED_DIR / "ml_cv_results.csv", index=False)
test_results_df.to_csv(PROCESSED_DIR / "ml_test_results.csv", index=False)

print("Résultats sauvegardés dans data/processed.")
