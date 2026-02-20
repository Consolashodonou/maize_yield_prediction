import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yieldlens | Smart Agricultural Insights",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
    <style>
        .main-banner {
            background: linear-gradient(90deg, #2E7D32, #66BB6A);
            padding: 25px; border-radius: 15px; color: white;
        }
        .main-banner h1 { margin-bottom: 5px; }
        .subtitle { font-size: 18px; margin-top: 0px; }
        .tagline { font-style: italic; font-size: 14px; opacity: 0.9; }
        .info-box {
            background: #f0f7f0; border-left: 4px solid #2E7D32;
            padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://www.clipartmax.com/png/small/98-988750_corn-icon-maize-icon.png", width=90)
with col_title:
    st.markdown("""
        <div class="main-banner">
            <h1>🌾 Yieldlens</h1>
            <p class="subtitle">AI-Powered Maize Yield Prediction for Smart Farming</p>
            <p class="tagline">"See the Future of Your Harvest."</p>
        </div>
    """, unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL  = "yield_kg_ha"
COUNTRY_COL = "country"
DROP_COL    = "potassium_mineral_fertilizer_kg_ha"
IMPUTE_COLS = [
    "nitrogen_mineral_fertilizer_kg_ha",
    "phosphorus_mineral_fertilizer_kg_ha",
    "nitrogen_organic_fertilizer_kg_ha",
    "nitrogen_atmospheric_deposition_kg_ha",
    "nitrogen_biological_fixation_kg_ha",
    "nitrogen_leaching_kg_ha",
    "nitrogen_seed_kg_ha",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def safe_impute(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    cols = [c for c in IMPUTE_COLS if c in df.columns]
    if cols and group_col in df.columns:
        df[cols] = df.groupby(group_col)[cols].transform(lambda s: s.fillna(s.median()))
    else:
        for c in cols:
            df[c] = df[c].fillna(df[c].median())
    return df

# ── Training pipeline ─────────────────────────────────────────────────────────
@st.cache_data
def load_training_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)

    missing = [c for c in [TARGET_COL, COUNTRY_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training file is missing required column(s): {missing}\n"
            f"Detected columns: {df.columns.tolist()}"
        )

    df = df.drop(columns=[DROP_COL], errors="ignore")
    df = safe_impute(df, COUNTRY_COL)
    return df

@st.cache_resource
def train_catboost(_df: pd.DataFrame):
    X = _df.drop(columns=[TARGET_COL])
    y = _df[TARGET_COL]
    X_enc = pd.get_dummies(X, columns=[COUNTRY_COL], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42
    )
    model = CatBoostRegressor(random_state=42, verbose=0, iterations=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²":   r2_score(y_test, y_pred),
    }
    return model, X_enc, X_train, X_test, y_train, y_test, y_pred, metrics

# ── Prediction pipeline ───────────────────────────────────────────────────────
@st.cache_data
def load_prediction_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)
    df = df.drop(columns=[DROP_COL, TARGET_COL], errors="ignore")  # strip target if accidentally included
    df = safe_impute(df, COUNTRY_COL)
    return df

def encode_prediction_data(pred_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
    """One-hot encode prediction data and align columns to training schema."""
    if COUNTRY_COL in pred_df.columns:
        pred_enc = pd.get_dummies(pred_df, columns=[COUNTRY_COL], drop_first=False)
    else:
        pred_enc = pred_df.copy()
    # Align: add missing dummy cols as 0, drop any extras
    pred_enc = pred_enc.reindex(columns=train_columns, fill_value=0)
    return pred_enc

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Two uploaders, clearly separated
# ════════════════════════════════════════════════════════════════════════════════
st.sidebar.header("📂 Data Uploads")

st.sidebar.markdown("**Step 1 — Training dataset**")
st.sidebar.caption("CSV with `yield_kg_ha` column (used to train & evaluate the model).")
train_file = st.sidebar.file_uploader("Upload training CSV", type="csv", key="train")

st.sidebar.markdown("---")
st.sidebar.markdown("**Step 2 — Prediction dataset** *(optional)*")
st.sidebar.caption("CSV without `yield_kg_ha` — the app will predict yield for each row.")
pred_file = st.sidebar.file_uploader("Upload prediction CSV", type="csv", key="pred")

# ── Guard: need training file at minimum ──────────────────────────────────────
if train_file is None:
    st.markdown("""
        <div class="info-box">
        👈 <strong>Getting started:</strong><br>
        <ol>
          <li>Upload your <strong>training CSV</strong> (must include <code>yield_kg_ha</code>) in the sidebar.</li>
          <li>Optionally upload a <strong>prediction CSV</strong> (features only, no yield column) to get batch predictions.</li>
        </ol>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load training data & train model ─────────────────────────────────────────
try:
    with st.spinner("Loading training data and fitting CatBoost…"):
        train_df = load_training_data(train_file)
        model, X_enc, X_train, X_test, y_train, y_test, y_pred_test, metrics = train_catboost(train_df)
    st.success(f"✅ Model trained on **{len(train_df)} rows** | R² = {metrics['R²']:.4f} | RMSE = {metrics['RMSE']:.1f} kg/ha")
except ValueError as e:
    st.error("⚠️ Training file issue:"); st.code(str(e)); st.stop()
except Exception as e:
    st.error("⚠️ Unexpected error during training."); st.exception(e); st.stop()

# ── Load prediction data if provided ─────────────────────────────────────────
pred_df_raw = None
pred_results = None

if pred_file is not None:
    try:
        pred_df_raw = load_prediction_data(pred_file)
        pred_enc    = encode_prediction_data(pred_df_raw, X_enc.columns)
        pred_yields = model.predict(pred_enc)
        pred_results = pred_df_raw.copy()
        pred_results["predicted_yield_kg_ha"] = np.round(pred_yields, 1)
    except Exception as e:
        st.warning("⚠️ Could not process prediction file.")
        st.exception(e)

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA", "📈 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── A: Batch predictions from uploaded file ───────────────────────────────
    if pred_results is not None:
        st.subheader("📋 Batch Predictions")
        st.caption(
            f"Predictions for **{len(pred_results)} rows** from your uploaded prediction file."
        )
        st.dataframe(pred_results, use_container_width=True)

        csv_out = pred_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_out,
            file_name="maize_yield_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Quick summary
        st.markdown("**Prediction Summary**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Min",    f"{pred_results['predicted_yield_kg_ha'].min():,.0f} kg/ha")
        s2.metric("Max",    f"{pred_results['predicted_yield_kg_ha'].max():,.0f} kg/ha")
        s3.metric("Mean",   f"{pred_results['predicted_yield_kg_ha'].mean():,.0f} kg/ha")
        s4.metric("Median", f"{pred_results['predicted_yield_kg_ha'].median():,.0f} kg/ha")

        # Distribution of predicted yields
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(pred_results["predicted_yield_kg_ha"], bins=20, color="#2E7D32", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Predicted Yield (kg/ha)"); ax.set_ylabel("Count")
        ax.set_title("Distribution of Predicted Yields")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig)

        # By country if present
        if COUNTRY_COL in pred_results.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=pred_results, x=COUNTRY_COL, y="predicted_yield_kg_ha",
                        hue=COUNTRY_COL, palette="Set2", legend=False, ax=ax)
            ax.set_title("Predicted Yield by Country")
            ax.set_xlabel("Country"); ax.set_ylabel("Predicted Yield (kg/ha)")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout(); st.pyplot(fig)

    # ── B: Manual single-row prediction (always available) ────────────────────
    st.markdown("---")
    st.subheader("🖊️ Manual Single-Row Prediction")
    st.caption("Fill in feature values to get an instant prediction.")

    COUNTRIES = sorted(train_df[COUNTRY_COL].unique().tolist())

    def med(col):
        return float(train_df[col].median()) if col in train_df.columns else 0.0

    c1, c2 = st.columns(2)
    with c1:
        country_sel    = st.selectbox("Country", COUNTRIES)
        year_val       = st.slider("Year", int(train_df["year"].min()), int(train_df["year"].max()), 2015) if "year" in train_df.columns else None
        area_val       = st.number_input("Area Harvested (ha)",                     min_value=0.0, value=med("area_harvested_ha"),                    step=100.0)
        n_mineral_val  = st.number_input("Nitrogen Mineral Fertilizer (kg/ha)",     min_value=0.0, value=med("nitrogen_mineral_fertilizer_kg_ha"),     step=1.0)
        p_mineral_val  = st.number_input("Phosphorus Mineral Fertilizer (kg/ha)",   min_value=0.0, value=med("phosphorus_mineral_fertilizer_kg_ha"),   step=1.0)
    with c2:
        n_organic_val  = st.number_input("Nitrogen Organic Fertilizer (kg/ha)",     min_value=0.0, value=med("nitrogen_organic_fertilizer_kg_ha"),     step=1.0)
        n_atm_val      = st.number_input("Nitrogen Atmospheric Deposition (kg/ha)", min_value=0.0, value=med("nitrogen_atmospheric_deposition_kg_ha"), step=0.5)
        n_bio_val      = st.number_input("Nitrogen Biological Fixation (kg/ha)",    min_value=0.0, value=med("nitrogen_biological_fixation_kg_ha"),    step=0.5)
        n_leach_val    = st.number_input("Nitrogen Leaching (kg/ha)",               min_value=0.0, value=med("nitrogen_leaching_kg_ha"),               step=0.5)
        n_seed_val     = st.number_input("Nitrogen Seed (kg/ha)",                   min_value=0.0, value=med("nitrogen_seed_kg_ha"),                   step=0.1)

    if st.button("🌽 Predict Yield", use_container_width=True):
        raw = {
            "area_harvested_ha":                       area_val,
            "nitrogen_mineral_fertilizer_kg_ha":       n_mineral_val,
            "phosphorus_mineral_fertilizer_kg_ha":     p_mineral_val,
            "nitrogen_organic_fertilizer_kg_ha":       n_organic_val,
            "nitrogen_atmospheric_deposition_kg_ha":   n_atm_val,
            "nitrogen_biological_fixation_kg_ha":      n_bio_val,
            "nitrogen_leaching_kg_ha":                 n_leach_val,
            "nitrogen_seed_kg_ha":                     n_seed_val,
        }
        if year_val is not None:
            raw["year"] = year_val

        # One-hot country dummies
        for col in X_enc.columns:
            if col.startswith("country_"):
                raw[col] = False
        ck = f"country_{country_sel}"
        if ck in X_enc.columns:
            raw[ck] = True

        input_df = pd.DataFrame([raw]).reindex(columns=X_enc.columns, fill_value=0)
        single_pred = model.predict(input_df)[0]

        st.markdown("---")
        st.metric("🌽 Predicted Maize Yield", f"{single_pred:,.0f} kg/ha")
        st.caption(f"Reference (baseline) country: **{COUNTRIES[0]}**")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — DYNAMIC EDA (adapts to whatever columns exist in training data)
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    st.caption(f"Based on training dataset · {len(train_df)} rows · {len(train_df.columns)} columns")

    num_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    # ── 1. Dataset overview ──────────────────────────────────────────────────
    with st.expander("📋 Dataset Overview", expanded=False):
        st.dataframe(train_df.describe().T.style.format("{:.2f}"), use_container_width=True)
        null_counts = train_df.isnull().sum()
        null_counts = null_counts[null_counts > 0]
        if not null_counts.empty:
            st.markdown("**Missing values:**")
            st.dataframe(null_counts.rename("Missing Count").to_frame(), use_container_width=True)
        else:
            st.success("No missing values in the training dataset.")

    # ── 2. Target distribution ───────────────────────────────────────────────
    if TARGET_COL in train_df.columns:
        st.markdown("#### 🎯 Target Variable: Yield Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(train_df[TARGET_COL], bins=25, color="#2E7D32", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Yield (kg/ha)"); ax.set_ylabel("Count")
            ax.set_title("Yield Distribution"); ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)
        with c2:
            if COUNTRY_COL in train_df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.boxplot(data=train_df, x=COUNTRY_COL, y=TARGET_COL,
                            hue=COUNTRY_COL, palette="Set2", legend=False, ax=ax)
                ax.set_title("Yield by Country"); ax.set_xlabel(""); ax.set_ylabel("Yield (kg/ha)")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout(); st.pyplot(fig)

    # ── 3. Time trends (only if year column exists) ──────────────────────────
    if "year" in train_df.columns and TARGET_COL in train_df.columns:
        st.markdown("#### 📅 Trends Over Time")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            if COUNTRY_COL in train_df.columns:
                for cname, grp in train_df.groupby(COUNTRY_COL):
                    ax.plot(grp["year"], grp[TARGET_COL], label=cname, marker="o", markersize=2)
                ax.legend(fontsize=7)
            else:
                ax.plot(train_df["year"], train_df[TARGET_COL], marker="o", markersize=2, color="#2E7D32")
            ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg/ha)")
            ax.set_title("Yield Trends Over Time"); ax.grid(linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)

        with c2:
            # Pick any numeric feature that changes with year (first available)
            trend_candidates = [c for c in IMPUTE_COLS if c in train_df.columns]
            if trend_candidates and COUNTRY_COL in train_df.columns:
                feat = trend_candidates[0]
                fig, ax = plt.subplots(figsize=(6, 3.5))
                for cname, grp in train_df.groupby(COUNTRY_COL):
                    ax.plot(grp["year"], grp[feat], label=cname, marker="o", markersize=2)
                ax.set_xlabel("Year"); ax.set_ylabel(feat.replace("_", " ").title())
                ax.set_title(f"{feat.replace('_', ' ').title()} Over Time")
                ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.5)
                plt.tight_layout(); st.pyplot(fig)

    # ── 4. Numeric feature distributions (dynamic: all numeric cols) ─────────
    if len(num_cols) > 0:
        st.markdown("#### 📦 Feature Distributions by Country")
        plot_cols = [c for c in num_cols if c != TARGET_COL and c != "year"]
        if plot_cols and COUNTRY_COL in train_df.columns:
            n_cols_grid = 3
            rows = (len(plot_cols) + n_cols_grid - 1) // n_cols_grid
            fig, axes = plt.subplots(rows, n_cols_grid, figsize=(n_cols_grid * 5, rows * 3.5))
            axes = np.array(axes).flatten()
            for i, col in enumerate(plot_cols):
                sns.boxplot(data=train_df, x=COUNTRY_COL, y=col, hue=COUNTRY_COL,
                            palette="Set2", legend=False, ax=axes[i])
                axes[i].set_title(col.replace("_", " ").title(), fontsize=9)
                axes[i].set_xlabel(""); axes[i].set_ylabel("")
                axes[i].tick_params(axis="x", rotation=30, labelsize=7)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout(); st.pyplot(fig)

    # ── 5. Correlation heatmap (only numeric columns that exist) ─────────────
    if len(num_cols) >= 2:
        st.markdown("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(min(12, len(num_cols) * 1.2 + 2), max(5, len(num_cols) * 0.9)))
        corr = train_df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # show lower triangle only
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="viridis",
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Pearson Correlation — Numeric Features")
        plt.tight_layout(); st.pyplot(fig)

    # ── 6. Scatter: top correlated features vs target ─────────────────────────
    if TARGET_COL in train_df.columns and len(num_cols) > 1:
        st.markdown("#### 🔍 Top Features vs Yield")
        corr_with_target = (
            train_df[num_cols].corr()[TARGET_COL]
            .drop(TARGET_COL, errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )
        top_features = corr_with_target.head(4).index.tolist()
        if top_features:
            n_cols_grid = 2
            rows = (len(top_features) + 1) // 2
            fig, axes = plt.subplots(rows, n_cols_grid, figsize=(12, rows * 4))
            axes = np.array(axes).flatten()
            for i, feat in enumerate(top_features):
                hue_kw = dict(hue=COUNTRY_COL, palette="viridis") if COUNTRY_COL in train_df.columns else {}
                sns.scatterplot(data=train_df, x=feat, y=TARGET_COL, alpha=0.7, ax=axes[i], **hue_kw)
                axes[i].set_title(f"Yield vs {feat.replace('_', ' ').title()}", fontsize=10)
                axes[i].set_xlabel(feat.replace("_", " ")); axes[i].set_ylabel("Yield (kg/ha)")
                axes[i].grid(linestyle="--", alpha=0.4)
                if COUNTRY_COL in train_df.columns:
                    axes[i].get_legend().remove() if axes[i].get_legend() else None
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout(); st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 CatBoost — Test Set Performance")
    st.caption("Model trained on 80% of the training dataset, evaluated on the remaining 20%.")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{metrics['MAE']:.1f} kg/ha",  help="Mean Absolute Error")
    m2.metric("RMSE", f"{metrics['RMSE']:.1f} kg/ha", help="Root Mean Squared Error")
    m3.metric("R²",   f"{metrics['R²']:.4f}",          help="Variance explained by the model")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted vs Actual**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred_test, alpha=0.65, color="#2E7D32", edgecolors="white", linewidths=0.4)
        lims = [float(y_test.min()) - 50, float(y_test.max()) + 50]
        ax.plot(lims, lims, 'r--', lw=1.5, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Yield (kg/ha)"); ax.set_ylabel("Predicted Yield (kg/ha)")
        ax.set_title("Predicted vs Actual"); ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout(); st.pyplot(fig)

    with c2:
        st.markdown("**Residuals**")
        residuals = y_pred_test - np.array(y_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_pred_test, residuals, alpha=0.65, color="#1565C0", edgecolors="white", linewidths=0.4)
        ax.axhline(0, color='red', linestyle='--', lw=1.5)
        ax.set_xlabel("Predicted Yield (kg/ha)"); ax.set_ylabel("Residual (Predicted − Actual)")
        ax.set_title("Residual Plot"); ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("**Feature Importance**")
    imp_df = pd.DataFrame({
        'Feature':    X_enc.columns,
        'Importance': model.get_feature_importance()
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3, len(imp_df) * 0.35)))
    colors = ["#2E7D32" if not f.startswith("country_") else "#81C784" for f in imp_df["Feature"]]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='white')
    ax.set_xlabel("Importance Score"); ax.set_title("CatBoost Feature Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    st.caption("🟢 Dark green = numeric features · Light green = country dummies")
