from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="📱",
    layout="wide",
)


# =========================
# Style
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(99, 102, 241, 0.16), transparent 35%),
            #080b13;
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 2.7rem;
        border-radius: 34px;
        background: linear-gradient(135deg, #172554 0%, #1d4ed8 52%, #4f46e5 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 28px 70px rgba(37, 99, 235, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .hero h1 {
        font-size: 2.8rem;
        line-height: 1.12;
        margin-bottom: 1rem;
        letter-spacing: -0.04em;
    }

    .hero p {
        font-size: 1.08rem;
        line-height: 1.7;
        color: rgba(255,255,255,.86);
        max-width: 980px;
    }

    .chip {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,.88);
        color: #1d4ed8;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-top: 0.65rem;
        font-size: 0.92rem;
    }

    .card {
        padding: 1.35rem 1.45rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 18px 45px rgba(0,0,0,.24);
        color: #e5e7eb;
        margin-bottom: 1rem;
    }

    .card strong {
        color: #ffffff;
    }

    .metric-card {
        padding: 1.25rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(30, 41, 59, .92), rgba(15, 23, 42, .92));
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 18px 45px rgba(0,0,0,.24);
        min-height: 128px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: .92rem;
        margin-bottom: .4rem;
    }

    .metric-value {
        color: white;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .metric-note {
        color: #cbd5e1;
        font-size: .88rem;
        margin-top: .3rem;
    }

    .section-title {
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        font-size: 1.65rem;
        font-weight: 850;
        color: #f8fafc;
        letter-spacing: -0.03em;
    }

    .step {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        text-align: center;
        color: #e2e8f0;
        font-weight: 700;
    }

    .ok-box {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: rgba(22, 163, 74, 0.16);
        border: 1px solid rgba(74, 222, 128, 0.28);
        color: #bbf7d0;
        font-weight: 700;
    }

    .warn-box {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: rgba(234, 179, 8, 0.14);
        border: 1px solid rgba(250, 204, 21, 0.28);
        color: #fef3c7;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    p, li {
        color: #dbeafe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Chemins
# =========================

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "report" / "figures"
REPORT_DIR = BASE_DIR / "report"


# =========================
# Fonctions
# =========================

@st.cache_data
def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def number(value):
    return f"{int(value):,}".replace(",", " ")


def card(text):
    st.markdown(f'<div class="card">{text}</div>', unsafe_allow_html=True)


def metric_card(label, value, note=""):
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


def status_box(label, ok=True):
    css = "ok-box" if ok else "warn-box"
    st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)


def get_feature_columns(df):
    excluded = {"subject_id", "activity_id", "activity", "split"}
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def activity_bar_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("activity:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("count:Q", title="Nombre d'observations"),
            tooltip=["activity", "count"],
            color=alt.value("#60a5fa"),
        )
        .properties(height=330)
    )
    st.altair_chart(chart, use_container_width=True)


def split_bar_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("split:N", title=None),
            y=alt.Y("nombre_observations:Q", title="Nombre d'observations"),
            tooltip=["split", "nombre_observations"],
            color=alt.value("#818cf8"),
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def model_bar_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("f1_macro:Q", title="F1-score macro", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color("approche:N", title="Approche"),
            tooltip=["model", "approche", "accuracy", "f1_macro"],
        )
        .properties(height=310)
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_resource
def train_demo_model(train_df):
    feature_cols = get_feature_columns(train_df)

    variances = train_df[feature_cols].var().sort_values(ascending=False)
    selected_features = variances.head(60).index.tolist()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=250,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train_df[selected_features], train_df["activity"])

    return model, selected_features


# =========================
# Chargement
# =========================

har_full = load_csv(PROCESSED_DIR / "har_full.csv")
har_train = load_csv(PROCESSED_DIR / "har_train.csv")
har_test = load_csv(PROCESSED_DIR / "har_test.csv")
activity_labels = load_csv(PROCESSED_DIR / "activity_labels.csv")
ml_results = load_csv(PROCESSED_DIR / "ml_test_results.csv")
dl_results = load_csv(PROCESSED_DIR / "dl_test_results.csv")


# =========================
# Navigation
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Aller vers",
    [
        "Vue d'ensemble",
        "Dataset",
        "Analyse exploratoire",
        "Modèles",
        "Démo prédiction",
        "Matrices de confusion",
        "Conclusion",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Projet Data Science — Human Activity Recognition")


# =========================
# Hero
# =========================

st.markdown(
    """
    <div class="hero">
        <h1>Reconnaissance d'activité humaine avec un smartphone</h1>
        <p>
            Interface de présentation du projet Data Science : préparation des données,
            analyse exploratoire, Machine Learning classique et Deep Learning sur signaux temporels avec CNN 1D.
        </p>
        <span class="chip">Classification supervisée</span>
        <span class="chip">Accéléromètre</span>
        <span class="chip">Gyroscope</span>
        <span class="chip">Machine Learning</span>
        <span class="chip">CNN 1D</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Pages
# =========================

if page == "Vue d'ensemble":
    st.markdown('<div class="section-title">Objectif du projet</div>', unsafe_allow_html=True)

    card(
        """
        Le projet consiste à reconnaître automatiquement l’activité réalisée par une personne
        à partir des données enregistrées par un smartphone. Les capteurs utilisés sont principalement
        l’accéléromètre et le gyroscope. Le problème est une classification supervisée en six activités.
        """
    )

    if har_full is not None:
        feature_count = len(get_feature_columns(har_full))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Observations", number(len(har_full)), "Fenêtres de mouvement")
        with col2:
            metric_card("Variables", number(feature_count), "Features numériques")
        with col3:
            metric_card("Activités", har_full["activity"].nunique(), "Classes à prédire")
        with col4:
            metric_card("Sujets", har_full["subject_id"].nunique(), "Participants")
    else:
        status_box("Données manquantes : lance d'abord task compile.", ok=False)

    st.markdown('<div class="section-title">Pipeline du projet</div>', unsafe_allow_html=True)

    steps = st.columns(6)
    labels = ["Données", "Nettoyage", "EDA", "ML", "CNN 1D", "Rapport"]
    for col, label in zip(steps, labels):
        with col:
            st.markdown(f'<div class="step">{label}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Fichiers de rendu</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        status_box("rapport.html généré" if (REPORT_DIR / "rapport.html").exists() else "rapport.html manquant", (REPORT_DIR / "rapport.html").exists())
    with col2:
        status_box("rapport.pdf généré" if (REPORT_DIR / "rapport.pdf").exists() else "rapport.pdf manquant", (REPORT_DIR / "rapport.pdf").exists())


elif page == "Dataset":
    st.markdown('<div class="section-title">Dataset utilisé</div>', unsafe_allow_html=True)

    card(
        """
        Le dataset utilisé est <strong>Human Activity Recognition Using Smartphones</strong>.
        Il contient des mesures de capteurs de smartphone associées à six activités humaines :
        marcher, monter les escaliers, descendre les escaliers, être assis, debout ou allongé.
        """
    )

    st.markdown('<div class="section-title">Activités</div>', unsafe_allow_html=True)

    if activity_labels is not None:
        st.dataframe(activity_labels, use_container_width=True, hide_index=True)
    else:
        status_box("activity_labels.csv manquant.", ok=False)

    if har_train is not None and har_test is not None:
        col1, col2 = st.columns(2)
        with col1:
            metric_card("Train", number(len(har_train)), "Données d'entraînement")
        with col2:
            metric_card("Test", number(len(har_test)), "Données d'évaluation")

        st.markdown('<div class="section-title">Taille des jeux train/test</div>', unsafe_allow_html=True)

        split_df = pd.DataFrame(
            {
                "split": ["train", "test"],
                "nombre_observations": [len(har_train), len(har_test)],
            }
        )
        split_bar_chart(split_df)
        st.dataframe(split_df, use_container_width=True, hide_index=True)


elif page == "Analyse exploratoire":
    st.markdown('<div class="section-title">Répartition des activités</div>', unsafe_allow_html=True)

    if har_full is not None:
        activity_counts = (
            har_full["activity"]
            .value_counts()
            .rename_axis("activity")
            .reset_index(name="count")
        )

        activity_bar_chart(activity_counts)

        st.markdown('<div class="section-title">Aperçu des données</div>', unsafe_allow_html=True)

        st.dataframe(
            har_full[["subject_id", "activity_id", "activity"]].head(12),
            use_container_width=True,
            hide_index=True,
        )

        card(
            """
            Les activités dynamiques comme la marche et les escaliers ont des signaux plus variables.
            Les activités statiques comme assis, debout et allongé sont plus stables, mais peuvent être
            plus difficiles à distinguer entre elles.
            """
        )
    else:
        status_box("har_full.csv manquant.", ok=False)


elif page == "Modèles":
    st.markdown('<div class="section-title">Résultats des modèles</div>', unsafe_allow_html=True)

    results_parts = []

    if ml_results is not None:
        tmp = ml_results.copy()
        tmp["approche"] = "Machine Learning"
        results_parts.append(tmp)

    if dl_results is not None:
        tmp = dl_results.copy()
        tmp["approche"] = "Deep Learning"
        results_parts.append(tmp)

    if results_parts:
        results = pd.concat(results_parts, ignore_index=True)

        cols = ["model", "approche", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        cols = [col for col in cols if col in results.columns]

        results = results[cols].sort_values("f1_macro", ascending=False)
        display_results = results.copy()

        for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            if col in display_results.columns:
                display_results[col] = display_results[col].round(3)

        st.dataframe(display_results, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Comparaison F1-score macro</div>', unsafe_allow_html=True)
        model_bar_chart(results)

        best_model = results.iloc[0]

        card(
            f"""
            <strong>Meilleur modèle observé :</strong> {best_model["model"]}<br>
            Le F1-score macro est utilisé parce qu’il donne le même poids aux six activités.
            Ici, la régression logistique ressort comme le modèle le plus stable sur les données tabulaires.
            """
        )
    else:
        status_box("Résultats de modèles manquants. Lance task compile.", ok=False)


elif page == "Démo prédiction":
    st.markdown('<div class="section-title">Démo de prédiction</div>', unsafe_allow_html=True)

    card(
        """
        Cette page entraîne rapidement un modèle de régression logistique sur les données tabulaires,
        puis teste une prédiction sur une observation du jeu de test. Cela permet de montrer concrètement
        comment le projet passe des capteurs à une activité prédite.
        """
    )

    if har_train is not None and har_test is not None:
        with st.spinner("Entraînement du modèle de démonstration..."):
            model, selected_features = train_demo_model(har_train)

        max_index = min(len(har_test) - 1, 500)
        sample_index = st.slider("Choisis une observation du jeu de test", 0, max_index, 0)

        sample = har_test.iloc[[sample_index]]
        true_activity = sample["activity"].iloc[0]
        predicted_activity = model.predict(sample[selected_features])[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Observation", sample_index, "Index dans le test")
        with col2:
            metric_card("Activité réelle", true_activity, "Label attendu")
        with col3:
            metric_card("Prédiction", predicted_activity, "Sortie du modèle")

        if predicted_activity == true_activity:
            status_box("Bonne prédiction : le modèle a reconnu la bonne activité.", ok=True)
        else:
            status_box("Mauvaise prédiction : le modèle a confondu cette activité avec une autre.", ok=False)

        if hasattr(model.named_steps["classifier"], "predict_proba"):
            proba = model.predict_proba(sample[selected_features])[0]
            classes = model.named_steps["classifier"].classes_

            proba_df = pd.DataFrame(
                {
                    "activity": classes,
                    "probability": proba,
                }
            ).sort_values("probability", ascending=False)

            st.markdown('<div class="section-title">Probabilités par activité</div>', unsafe_allow_html=True)

            chart = (
                alt.Chart(proba_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("activity:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("probability:Q", title="Probabilité", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["activity", "probability"],
                    color=alt.value("#38bdf8"),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown('<div class="section-title">Variables utilisées</div>', unsafe_allow_html=True)
        st.caption("Le modèle utilise les 60 variables les plus variables du dataset pour garder une démo rapide.")
        st.dataframe(pd.DataFrame({"features": selected_features}), use_container_width=True, hide_index=True)
    else:
        status_box("Données train/test manquantes.", ok=False)


elif page == "Matrices de confusion":
    st.markdown('<div class="section-title">Matrices de confusion</div>', unsafe_allow_html=True)

    ml_matrix = FIGURES_DIR / "confusion_matrix_best_ml_model.png"
    cnn_matrix = FIGURES_DIR / "confusion_matrix_cnn_1d.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Learning")
        if ml_matrix.exists():
            st.image(str(ml_matrix), use_container_width=True)
        else:
            status_box("Matrice ML non trouvée.", ok=False)

    with col2:
        st.subheader("CNN 1D")
        if cnn_matrix.exists():
            st.image(str(cnn_matrix), use_container_width=True)
        else:
            status_box("Matrice CNN non trouvée.", ok=False)


elif page == "Conclusion":
    st.markdown('<div class="section-title">Conclusion</div>', unsafe_allow_html=True)

    card(
        """
        Ce projet montre qu’il est possible de reconnaître automatiquement une activité humaine à partir
        des capteurs d’un smartphone. Les modèles classiques donnent une première base solide, tandis que
        le CNN 1D permet de travailler directement sur les signaux temporels.
        """
    )

    st.markdown('<div class="section-title">Limites</div>', unsafe_allow_html=True)

    st.markdown(
        """
        - Le dataset est déjà propre, ce qui n’est pas toujours le cas dans un vrai projet.
        - Les modèles ont été allégés pour accélérer la compilation.
        - Il faudrait tester le modèle sur plus de personnes et sur différents téléphones.
        - Le CNN 1D pourrait être entraîné plus longtemps pour améliorer ses résultats.
        """
    )

    st.markdown('<div class="section-title">Améliorations possibles</div>', unsafe_allow_html=True)

    st.markdown(
        """
        - Ajouter un dashboard encore plus interactif.
        - Tester plus de modèles.
        - Faire une recherche d’hyperparamètres.
        - Sauvegarder le meilleur modèle pour l’utiliser sans le réentraîner.
        - Tester avec de vraies données collectées depuis un smartphone.
        """
    )
