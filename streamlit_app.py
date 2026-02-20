import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yieldlens | Smart Agricultural Insights",
    page_icon="🌾",
    layout="wide"
)

# ── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main-banner {
            background: linear-gradient(90deg, #2E7D32, #66BB6A);
            padding: 25px;
            border-radius: 15px;
            color: white;
        }
        .main-banner h1 {
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            margin-top: 0px;
        }
        .tagline {
            font-style: italic;
            font-size: 14px;
            opacity: 0.9;
        }
    </style>
""", unsafe_allow_html=True)

# ── Banner Layout ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 5])

with col1:
    st.image(
        "https://www.clipartmax.com/png/small/98-988750_corn-icon-maize-icon.png",
        width=90
    )

with col2:
    st.markdown("""
        <div class="main-banner">
            <h1>🌾 Yieldlens</h1>
            <p class="subtitle">
                AI-Powered Maize Yield Prediction for Smart Farming
            </p>
            <p class="tagline">
                "See the Future of Your Harvest."
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    ### 🌍 Coverage
    Predict maize yield (kg/ha) for **Benin, Ethiopia, Ivory Coast, Kenya, and Malawi**  
    using ensemble machine learning (Best model: **CatBoost**, R² ≈ 0.84).
    """
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_file):
    faostat = pd.read_csv(uploaded_file)

    # --- exact same pipeline as the notebook (untouched logic) ---
    faostat = faostat.drop(columns=['potassium_mineral_fertilizer_kg_ha'])

    cols_missing = [
        "nitrogen_mineral_fertilizer_kg_ha",
        "phosphorus_mineral_fertilizer_kg_ha",
        "nitrogen_organic_fertilizer_kg_ha",
        "nitrogen_atmospheric_deposition_kg_ha",
        "nitrogen_biological_fixation_kg_ha",
        "nitrogen_leaching_kg_ha",
        "nitrogen_seed_kg_ha",
    ]
    faostat[cols_missing] = faostat.groupby("country")[cols_missing].transform(
        lambda s: s.fillna(s.median())
    )

    X = faostat.drop("yield_kg_ha", axis=1)
    y = faostat["yield_kg_ha"]
    X_processed = pd.get_dummies(X, columns=["country"], drop_first=True)

    return faostat, X_processed, y

@st.cache_resource
def train_model(_X_processed, _y):
    X_train, X_test, y_train, y_test = train_test_split(
        _X_processed, _y, test_size=0.2, random_state=42
    )
    model = CatBoostRegressor(random_state=42, verbose=0, iterations=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²":   r2_score(y_test, y_pred),
    }
    return model, X_train, X_test, y_train, y_test, y_pred, metrics

# ── Sidebar: dataset upload ───────────────────────────────────────────────────
st.sidebar.header("📂 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload maize_yield_data.csv",
    type="csv"
)

if uploaded_file is None:
    st.info("👈 Please upload the CSV dataset in the sidebar to get started.")
    st.stop()

# ── Load & train ──────────────────────────────────────────────────────────────
with st.spinner("Loading data and training CatBoost model…"):
    faostat, X_processed, y = load_and_preprocess(uploaded_file)
    model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(X_processed, y)

st.success("✅ Model trained successfully!")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA", "📈 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter input values to predict maize yield")

    COUNTRIES = sorted(faostat["country"].unique().tolist())
    # The reference country dropped by drop_first (alphabetically first)
    ref_country = sorted(COUNTRIES)[0]

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", COUNTRIES)
        year = st.slider("Year", int(faostat["year"].min()), int(faostat["year"].max()), 2015)
        area_harvested = st.number_input("Area Harvested (ha)", min_value=0.0,
                                         value=float(faostat["area_harvested_ha"].median()), step=100.0)
        nitrogen_mineral = st.number_input("Nitrogen Mineral Fertilizer (kg/ha)", min_value=0.0,
                                            value=float(faostat["nitrogen_mineral_fertilizer_kg_ha"].median()), step=1.0)
        phosphorus_mineral = st.number_input("Phosphorus Mineral Fertilizer (kg/ha)", min_value=0.0,
                                              value=float(faostat["phosphorus_mineral_fertilizer_kg_ha"].median()), step=1.0)

    with col2:
        nitrogen_organic = st.number_input("Nitrogen Organic Fertilizer (kg/ha)", min_value=0.0,
                                            value=float(faostat["nitrogen_organic_fertilizer_kg_ha"].median()), step=1.0)
        nitrogen_atm = st.number_input("Nitrogen Atmospheric Deposition (kg/ha)", min_value=0.0,
                                        value=float(faostat["nitrogen_atmospheric_deposition_kg_ha"].median()), step=0.5)
        nitrogen_bio = st.number_input("Nitrogen Biological Fixation (kg/ha)", min_value=0.0,
                                        value=float(faostat["nitrogen_biological_fixation_kg_ha"].median()), step=0.5)
        nitrogen_leach = st.number_input("Nitrogen Leaching (kg/ha)", min_value=0.0,
                                          value=float(faostat["nitrogen_leaching_kg_ha"].median()), step=0.5)
        nitrogen_seed = st.number_input("Nitrogen Seed (kg/ha)", min_value=0.0,
                                         value=float(faostat["nitrogen_seed_kg_ha"].median()), step=0.1)

    if st.button("🌽 Predict Yield", use_container_width=True):
        # Build input row matching X_processed column order
        input_dict = {
            "year": year,
            "area_harvested_ha": area_harvested,
            "nitrogen_mineral_fertilizer_kg_ha": nitrogen_mineral,
            "phosphorus_mineral_fertilizer_kg_ha": phosphorus_mineral,
            "nitrogen_organic_fertilizer_kg_ha": nitrogen_organic,
            "nitrogen_atmospheric_deposition_kg_ha": nitrogen_atm,
            "nitrogen_biological_fixation_kg_ha": nitrogen_bio,
            "nitrogen_leaching_kg_ha": nitrogen_leach,
            "nitrogen_seed_kg_ha": nitrogen_seed,
        }
        # Add one-hot columns (all False by default = reference country)
        for col in X_processed.columns:
            if col.startswith("country_"):
                input_dict[col] = False

        # Set the selected country's dummy to True (if not reference)
        country_col = f"country_{country}"
        if country_col in X_processed.columns:
            input_dict[country_col] = True

        input_df = pd.DataFrame([input_dict])[X_processed.columns]
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.metric("🌽 Predicted Maize Yield", f"{prediction:,.0f} kg/ha")
        st.caption(f"Reference country (baseline): **{ref_country}**")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Maize Yield by Country**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=faostat, x='country', y='yield_kg_ha', hue='country',
                    palette='Set2', legend=False, ax=ax)
        ax.set_xlabel("Country"); ax.set_ylabel("Yield (kg/ha)")
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("**Maize Yield Trends Over Time**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for country_name, group in faostat.groupby('country'):
            ax.plot(group['year'], group['yield_kg_ha'], label=country_name, marker='o', markersize=2)
        ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg/ha)")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = faostat.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("CatBoost — Test Set Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{metrics['MAE']:.2f} kg/ha")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f} kg/ha")
    m3.metric("R²",   f"{metrics['R²']:.4f}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted vs Actual**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.6)
        lims = [y.min(), y.max()]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlabel("Actual Yield (kg/ha)")
        ax.set_ylabel("Predicted Yield (kg/ha)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("**Feature Importance**")
        importance = model.get_feature_importance()
        imp_df = pd.DataFrame({
            'Feature': X_processed.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='green', edgecolor='black')
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
