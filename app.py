from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="📱",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "data" / "dashboard"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_DIR = DASHBOARD_DIR if DASHBOARD_DIR.exists() else PROCESSED_DIR
FIGURES_DIR = BASE_DIR / "report" / "figures"

REPORT_URL = "https://rizleneberrag.github.io/aptispace-datascience-projet/"


# =========================
# STYLE PREMIUM MINIMAL
# =========================

st.markdown(
    """
<style>
    .stApp {
        background: #f7f4ef;
        color: #111827;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: rgba(247,244,239,0.85);
        backdrop-filter: blur(14px);
    }

    h1, h2, h3 {
        color: #111827;
        letter-spacing: -0.045em;
    }

    p, span, label, div {
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .hero {
        padding: 54px 46px;
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(255,255,255,.95), rgba(244,239,229,.95));
        border: 1px solid rgba(17,24,39,.08);
        box-shadow: 0 24px 80px rgba(17,24,39,.08);
        margin-bottom: 26px;
    }

    .eyebrow {
        display: inline-block;
        font-size: .78rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #6b7280;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .title {
        font-size: clamp(2.8rem, 6vw, 6.4rem);
        line-height: .88;
        font-weight: 950;
        max-width: 950px;
        margin-bottom: 22px;
    }

    .subtitle {
        font-size: 1.12rem;
        line-height: 1.75;
        color: #4b5563;
        max-width: 780px;
    }

    .metric-card {
        background: rgba(255,255,255,.86);
        border: 1px solid rgba(17,24,39,.08);
        border-radius: 26px;
        padding: 24px;
        box-shadow: 0 18px 50px rgba(17,24,39,.055);
        min-height: 124px;
    }

    .metric-label {
        color: #6b7280;
        font-size: .84rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #111827;
        font-size: 2.2rem;
        font-weight: 950;
        letter-spacing: -.04em;
    }

    .metric-note {
        color: #6b7280;
        font-size: .9rem;
        margin-top: 8px;
    }

    .panel {
        background: rgba(255,255,255,.86);
        border: 1px solid rgba(17,24,39,.08);
        border-radius: 30px;
        padding: 32px;
        box-shadow: 0 18px 55px rgba(17,24,39,.055);
        margin-top: 22px;
    }

    .panel-title {
        font-size: 2rem;
        font-weight: 950;
        letter-spacing: -.04em;
        margin-bottom: 8px;
        color: #111827;
    }

    .panel-text {
        color: #4b5563;
        font-size: 1rem;
        line-height: 1.72;
        max-width: 820px;
    }

    .pill {
        display: inline-flex;
        padding: 9px 14px;
        border-radius: 999px;
        background: #111827;
        color: #fff;
        font-size: .86rem;
        font-weight: 750;
        margin: 6px 6px 0 0;
    }

    .soft-pill {
        display: inline-flex;
        padding: 9px 14px;
        border-radius: 999px;
        background: #f3efe7;
        border: 1px solid rgba(17,24,39,.08);
        color: #374151;
        font-size: .86rem;
        font-weight: 700;
        margin: 6px 6px 0 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,.55);
        padding: 8px;
        border-radius: 999px;
        border: 1px solid rgba(17,24,39,.08);
        width: fit-content;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 10px 18px;
        color: #374151;
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        background: #111827;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(17,24,39,.08);
    }

    .stButton > button,
    .stLinkButton > a {
        border-radius: 999px !important;
        background: #111827 !important;
        color: white !important;
        border: none !important;
        padding: .75rem 1.25rem !important;
        font-weight: 800 !important;
    }

    hr {
        margin: 2.2rem 0;
        border-color: rgba(17,24,39,.08);
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# DATA
# =========================

@st.cache_data
def load_csv(name):
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


full = load_csv("har_full.csv")
train = load_csv("har_train.csv")
test = load_csv("har_test.csv")
ml_results = load_csv("ml_test_results.csv")
dl_results = load_csv("dl_test_results.csv")


def detect_label(df):
    # Détection robuste de la colonne cible / activité
    expected_activities = {
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    }

    priority_cols = [
        "activity",
        "Activity",
        "activity_name",
        "activity_label",
        "ActivityName",
        "label",
        "Label",
        "target",
        "y",
    ]

    for col in priority_cols:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            values = set(df[col].dropna().astype(str).unique())
            if values & expected_activities or 2 <= len(values) <= 10:
                return col

    # Cherche une colonne texte qui contient les vraies activités HAR
    for col in df.select_dtypes(include="object").columns:
        values = set(df[col].dropna().astype(str).unique())
        if values & expected_activities and df[col].nunique(dropna=True) > 1:
            return col

    # Dernier recours : colonne texte avec plusieurs classes, mais pas train/test
    forbidden_values = {"train", "test"}
    for col in df.select_dtypes(include="object").columns:
        values = set(df[col].dropna().astype(str).str.lower().unique())
        if 2 <= len(values) <= 10 and not values.issubset(forbidden_values):
            return col

    return None


def detect_subject(df):
    for col in ["subject", "Subject", "subject_id"]:
        if col in df.columns:
            return col
    return None


if full.empty:
    st.error("Les données du dashboard sont introuvables. Vérifie le dossier data/dashboard.")
    st.stop()

label_col = detect_label(full)
subject_col = detect_subject(full)

excluded = {label_col, subject_col, "activity_id", "label_id"}
feature_cols = [
    c for c in full.select_dtypes(include=np.number).columns
    if c not in excluded
]

activities = (
    sorted(full[label_col].dropna().astype(str).unique())
    if label_col else []
)


def find_model_col(df):
    for col in df.columns:
        if "model" in col.lower() or "modèle" in col.lower():
            return col
    return df.columns[0] if len(df.columns) else None


def find_metric_col(df, word):
    for col in df.columns:
        if word.lower() in col.lower():
            return col
    return None


def best_model():
    if ml_results.empty:
        return "Logistic Regression", "≈ 89%"

    m_col = find_model_col(ml_results)
    f1_col = find_metric_col(ml_results, "f1")
    acc_col = find_metric_col(ml_results, "accuracy")
    score_col = f1_col or acc_col

    if not m_col or not score_col:
        return "Logistic Regression", "meilleur score"

    tmp = ml_results.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp = tmp.dropna(subset=[score_col])

    if tmp.empty:
        return "Logistic Regression", "meilleur score"

    row = tmp.sort_values(score_col, ascending=False).iloc[0]
    score = row[score_col]
    score_txt = f"{score:.2%}" if score <= 1 else f"{score:.2f}"
    return str(row[m_col]), score_txt


best_name, best_score = best_model()


def metric(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title, text):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# HERO
# =========================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Projet Data Science · Smartphone Sensors</div>
        <div class="title">Human Activity Recognition</div>
        <div class="subtitle">
            Reconnaître automatiquement une activité humaine à partir des capteurs d’un smartphone.
            Ce dashboard présente les données, l’analyse, les modèles et les résultats du projet.
        </div>
        <br>
        <span class="soft-pill">Machine Learning</span>
        <span class="soft-pill">CNN 1D</span>
        <span class="soft-pill">UCI HAR Dataset</span>
        <span class="soft-pill">Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric("Observations", f"{len(full):,}", "données chargées")
with c2:
    metric("Activités", len(activities), "classes à prédire")
with c3:
    metric("Features", len(feature_cols), "variables numériques")
with c4:
    metric("Meilleur modèle", best_name, best_score)


# =========================
# NAVIGATION SIMPLE
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Vue d’ensemble", "Données", "Analyse", "Résultats", "Démo"]
)


# =========================
# TAB 1
# =========================

with tab1:
    panel(
        "Objectif du projet",
        "Le projet consiste à prédire l’activité réalisée par une personne à partir des signaux "
        "enregistrés par un smartphone. Les capteurs utilisés sont principalement l’accéléromètre "
        "et le gyroscope."
    )

    st.markdown("### Activités reconnues")

    pills = "".join([f'<span class="pill">{a}</span>' for a in activities])
    st.markdown(pills, unsafe_allow_html=True)

    st.markdown("### Pipeline du projet")

    pipeline = pd.DataFrame(
        {
            "Étape": [
                "1. Préparation",
                "2. Analyse exploratoire",
                "3. Modélisation ML",
                "4. Deep Learning",
                "5. Évaluation",
                "6. Dashboard",
            ],
            "Description": [
                "Reconstruction des fichiers et nettoyage des données",
                "Étude de la répartition des activités",
                "Comparaison de plusieurs modèles classiques",
                "Test d’un CNN 1D sur signaux temporels",
                "Accuracy, F1-score et matrices de confusion",
                "Visualisation interactive avec Streamlit",
            ],
        }
    )

    st.dataframe(pipeline, use_container_width=True, hide_index=True)


# =========================
# TAB 2
# =========================

with tab2:
    panel(
        "Dataset",
        "Les données utilisées proviennent du dataset UCI HAR. Chaque observation correspond "
        "à des mesures de mouvement associées à une activité réelle."
    )

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### Aperçu des données")
        st.dataframe(full.head(50), use_container_width=True)

    with right:
        st.markdown("### Structure")

        structure = pd.DataFrame(
            {
                "Élément": [
                    "Nombre de lignes",
                    "Activités",
                    "Variables numériques",
                    "Colonne cible",
                    "Dossier utilisé",
                ],
                "Valeur": [
                    f"{len(full):,}",
                    len(activities),
                    len(feature_cols),
                    label_col or "Non détectée",
                    str(DATA_DIR.relative_to(BASE_DIR)),
                ],
            }
        )

        st.dataframe(structure, use_container_width=True, hide_index=True)


# =========================
# TAB 3
# =========================

with tab3:
    panel(
        "Analyse exploratoire",
        "L’analyse exploratoire permet de vérifier la répartition des activités et de comprendre "
        "pourquoi certaines classes peuvent être plus difficiles à distinguer."
    )

    if label_col:
        counts = full[label_col].astype(str).value_counts().reset_index()
        counts.columns = ["Activité", "Nombre"]

        chart = (
            alt.Chart(counts)
            .mark_bar(cornerRadius=8, color="#111827")
            .encode(
                x=alt.X("Nombre:Q", title="Nombre d'observations"),
                y=alt.Y("Activité:N", sort="-x", title=None),
                tooltip=["Activité", "Nombre"],
            )
            .properties(height=360)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            Les activités dynamiques comme marcher, monter ou descendre les escaliers génèrent
            des signaux plus marqués. Les activités statiques comme assis, debout et allongé
            sont plus proches, ce qui peut expliquer certaines confusions.
            """
        )


# =========================
# TAB 4
# =========================

with tab4:
    panel(
        "Résultats des modèles",
        "Les modèles sont évalués avec l’accuracy, le F1-score macro et les matrices de confusion. "
        "La régression logistique obtient les meilleurs résultats dans cette version."
    )

    if not ml_results.empty:
        st.markdown("### Comparaison Machine Learning")
        st.dataframe(ml_results, use_container_width=True, hide_index=True)

        m_col = find_model_col(ml_results)
        f1_col = find_metric_col(ml_results, "f1")
        acc_col = find_metric_col(ml_results, "accuracy")
        score_col = f1_col or acc_col

        if m_col and score_col:
            plot_df = ml_results.copy()
            plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[score_col])

            chart = (
                alt.Chart(plot_df)
                .mark_bar(cornerRadius=8, color="#111827")
                .encode(
                    x=alt.X(f"{score_col}:Q", title=score_col),
                    y=alt.Y(f"{m_col}:N", sort="-x", title=None),
                    tooltip=[m_col, score_col],
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Aucun fichier de résultats ML trouvé.")

    st.markdown("### Matrices de confusion")

    img1 = FIGURES_DIR / "confusion_matrix_best_ml_model.png"
    img2 = FIGURES_DIR / "confusion_matrix_cnn_1d.png"

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Meilleur modèle ML**")
        if img1.exists():
            st.image(str(img1), use_container_width=True)
        else:
            st.info("Matrice ML non trouvée.")

    with col_b:
        st.markdown("**CNN 1D**")
        if img2.exists():
            st.image(str(img2), use_container_width=True)
        else:
            st.info("Matrice CNN non trouvée.")

    if not dl_results.empty:
        st.markdown("### Résultat CNN 1D")
        st.dataframe(dl_results, use_container_width=True, hide_index=True)


# =========================
# TAB 5
# =========================

with tab5:
    panel(
        "Démonstration",
        "Cette partie permet de tester une prédiction sur un exemple du dataset. "
        "Le modèle utilisé ici est une régression logistique entraînée rapidement pour la démonstration."
    )

    if not label_col or train.empty:
        st.info("Données insuffisantes pour lancer la démo.")
    else:
        demo_features = feature_cols[:80] if len(feature_cols) > 80 else feature_cols

        X = train[demo_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = train[label_col].astype(str)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )

        model.fit(X, y)

        demo_df = test if not test.empty else full
        selected_activity = st.selectbox(
            "Choisir une activité réelle",
            sorted(demo_df[label_col].astype(str).unique()),
        )

        subset = demo_df[demo_df[label_col].astype(str) == selected_activity]

        index = st.slider(
            "Choisir un exemple",
            0,
            max(len(subset) - 1, 0),
            0,
        )

        row = subset.iloc[[index]]
        X_row = row[demo_features].replace([np.inf, -np.inf], np.nan).fillna(0)

        prediction = model.predict(X_row)[0]

        a, b, c = st.columns(3)

        with a:
            metric("Activité réelle", selected_activity, "label du dataset")
        with b:
            metric("Prédiction", prediction, "sortie du modèle")
        with c:
            result = "Correct" if str(prediction) == str(selected_activity) else "Erreur"
            metric("Résultat", result, "comparaison")

        if hasattr(model.named_steps["clf"], "predict_proba"):
            proba = model.predict_proba(X_row)[0]
            classes = model.named_steps["clf"].classes_

            proba_df = pd.DataFrame(
                {
                    "Activité": classes,
                    "Probabilité": proba,
                }
            ).sort_values("Probabilité", ascending=False)

            chart = (
                alt.Chart(proba_df)
                .mark_bar(cornerRadius=8, color="#111827")
                .encode(
                    x=alt.X("Probabilité:Q", title="Probabilité"),
                    y=alt.Y("Activité:N", sort="-x", title=None),
                    tooltip=["Activité", alt.Tooltip("Probabilité:Q", format=".2%")],
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.markdown("### Rapport final")
    st.write("Le rapport complet du projet est disponible en ligne.")
    st.link_button("Ouvrir le rapport", REPORT_URL)


st.markdown("---")
st.caption("Projet Data Science — Human Activity Recognition · Berrag Rizlene")
