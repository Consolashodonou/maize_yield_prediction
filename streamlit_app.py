import io
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

st.markdown(
    "### 🌍 Coverage\n"
    "Predict maize yield (kg/ha) for **Benin, Ethiopia, Ivory Coast, Kenya, and Malawi** "
    "using ensemble machine learning — Best model: **CatBoost** (R² ≈ 0.84)."
)

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
    """Lowercase + strip spaces from all column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute only columns that exist, grouped by country if available."""
    cols = [c for c in IMPUTE_COLS if c in df.columns]
    if not cols:
        return df
    if COUNTRY_COL in df.columns:
        df[cols] = df.groupby(COUNTRY_COL)[cols].transform(lambda s: s.fillna(s.median()))
    else:
        for c in cols:
            df[c] = df[c].fillna(df[c].median())
    return df

# ────────────────────────────────────────────────────────────────────────────
# CRITICAL FIX: all cached functions receive raw bytes (not UploadedFile).
# UploadedFile is a stream — after Streamlit hashes it for caching, the read
# pointer sits at EOF. Passing bytes instead guarantees the data is always
# available regardless of stream state.
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_training_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = normalize_columns(df)

    missing = [c for c in [TARGET_COL, COUNTRY_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training file is missing required column(s): {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    df = df.drop(columns=[DROP_COL], errors="ignore")
    df = safe_impute(df)
    return df


@st.cache_resource(show_spinner=False)
def train_catboost(file_bytes: bytes):
    """
    Accepts bytes so the cache key is deterministic and stable.
    Re-loads the df internally to keep everything self-contained.
    """
    df = load_training_data(file_bytes)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
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
    return model, X_enc.columns.tolist(), X_test, y_test, y_pred, metrics


def load_prediction_data(file_bytes: bytes) -> pd.DataFrame:
    """
    NOT cached on purpose — prediction files are typically used once and
    caching a large byte blob adds no benefit here.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = normalize_columns(df)
    # Strip target if it was accidentally included
    df = df.drop(columns=[DROP_COL, TARGET_COL], errors="ignore")
    df = safe_impute(df)
    return df


def encode_and_align(pred_df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    """
    One-hot encode the prediction dataframe and align its columns
    exactly to the training schema (same order, same dummies).
    Missing dummy columns → 0. Extra columns → dropped.
    """
    if COUNTRY_COL in pred_df.columns:
        enc = pd.get_dummies(pred_df, columns=[COUNTRY_COL], drop_first=False)
    else:
        enc = pred_df.copy()
    enc = enc.reindex(columns=train_columns, fill_value=0)
    return enc


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Data Uploads")

st.sidebar.markdown("**Step 1 — Training dataset** *(required)*")
st.sidebar.caption(f"CSV that includes `{TARGET_COL}` — used to train and evaluate the model.")
train_file = st.sidebar.file_uploader("Upload training CSV", type="csv", key="train")

st.sidebar.markdown("---")
st.sidebar.markdown("**Step 2 — Prediction dataset** *(optional)*")
st.sidebar.caption(f"CSV without `{TARGET_COL}` — the app predicts yield for every row.")
pred_file = st.sidebar.file_uploader("Upload prediction CSV", type="csv", key="pred")

# ── Guard ─────────────────────────────────────────────────────────────────────
if train_file is None:
    st.markdown("""
        <div class="info-box">
        👈 <strong>How to use Yieldlens:</strong><br><br>
        <b>Step 1</b> — Upload your <b>training CSV</b> (must contain <code>yield_kg_ha</code>)
        to train the CatBoost model.<br>
        <b>Step 2 (optional)</b> — Upload a <b>prediction CSV</b> (features only, no yield column)
        to get batch yield predictions for new observations.
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Read training bytes once — passed to all cached functions ─────────────────
train_bytes = train_file.read()

try:
    with st.spinner("Loading training data…"):
        train_df = load_training_data(train_bytes)

    with st.spinner("Training CatBoost model…"):
        model, train_columns, X_test, y_test, y_pred_test, metrics = train_catboost(train_bytes)

    st.success(
        f"✅ Model trained on **{len(train_df)} rows** — "
        f"R² = **{metrics['R²']:.4f}** | RMSE = **{metrics['RMSE']:.1f} kg/ha**"
    )

except ValueError as e:
    st.error("⚠️ **Training file issue detected.**")
    st.code(str(e))
    st.stop()
except Exception as e:
    st.error("⚠️ **Unexpected error during training.**")
    st.exception(e)
    st.stop()

# ── Process prediction file if uploaded ──────────────────────────────────────
pred_results = None

if pred_file is not None:
    try:
        # Read bytes immediately — do NOT pass the UploadedFile object directly
        pred_bytes  = pred_file.read()
        pred_df_raw = load_prediction_data(pred_bytes)

        if pred_df_raw.empty:
            st.sidebar.error("Prediction file appears to be empty after loading.")
        else:
            pred_enc    = encode_and_align(pred_df_raw, train_columns)
            pred_yields = model.predict(pred_enc)
            pred_results = pred_df_raw.copy()
            pred_results.insert(0, "predicted_yield_kg_ha", np.round(pred_yields, 1))

    except Exception as e:
        st.sidebar.error("⚠️ Could not process the prediction file.")
        st.sidebar.exception(e)

# ── Convenience helpers ───────────────────────────────────────────────────────
COUNTRIES   = sorted(train_df[COUNTRY_COL].unique().tolist())
ref_country = COUNTRIES[0]

def col_median(col):
    return float(train_df[col].median()) if col in train_df.columns else 0.0

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA", "📈 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Section A: Batch predictions from uploaded file ───────────────────────
    if pred_results is not None:
        st.subheader("📋 Batch Predictions — Uploaded File")
        st.caption(
            f"**{len(pred_results)} rows** predicted from your uploaded prediction CSV. "
            f"Predicted yield inserted as the first column."
        )

        st.dataframe(pred_results, use_container_width=True)

        csv_out = pred_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download predictions as CSV",
            data=csv_out,
            file_name="maize_yield_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Summary statistics
        st.markdown("#### Prediction Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Min",    f"{pred_results['predicted_yield_kg_ha'].min():,.0f} kg/ha")
        s2.metric("Max",    f"{pred_results['predicted_yield_kg_ha'].max():,.0f} kg/ha")
        s3.metric("Mean",   f"{pred_results['predicted_yield_kg_ha'].mean():,.0f} kg/ha")
        s4.metric("Median", f"{pred_results['predicted_yield_kg_ha'].median():,.0f} kg/ha")

        # Distribution of predictions
        pc1, pc2 = st.columns(2)
        with pc1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(pred_results["predicted_yield_kg_ha"], bins=20,
                    color="#2E7D32", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Predicted Yield (kg/ha)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Predicted Yields")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)

        with pc2:
            if COUNTRY_COL in pred_results.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.boxplot(
                    data=pred_results, x=COUNTRY_COL, y="predicted_yield_kg_ha",
                    hue=COUNTRY_COL, palette="Set2", legend=False, ax=ax
                )
                ax.set_title("Predicted Yield by Country")
                ax.set_xlabel(""); ax.set_ylabel("Predicted Yield (kg/ha)")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout(); st.pyplot(fig)
            elif "year" in pred_results.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.scatter(pred_results["year"], pred_results["predicted_yield_kg_ha"],
                           alpha=0.6, color="#2E7D32")
                ax.set_xlabel("Year"); ax.set_ylabel("Predicted Yield (kg/ha)")
                ax.set_title("Predicted Yield Over Time")
                ax.grid(linestyle="--", alpha=0.5)
                plt.tight_layout(); st.pyplot(fig)

    else:
        # Guide message if no prediction file yet
        st.info(
            "📂 No prediction file uploaded yet. "
            "Upload a CSV (without `yield_kg_ha`) in the sidebar to get batch predictions. "
            "You can also use the manual predictor below."
        )

    # ── Section B: Manual single-row prediction ───────────────────────────────
    st.markdown("---")
    st.subheader("🖊️ Manual Single-Row Prediction")
    st.caption("Fill in feature values below and click Predict to get an instant result.")

    mc1, mc2 = st.columns(2)
    with mc1:
        country_sel   = st.selectbox("Country", COUNTRIES)
        year_val      = (
            st.slider("Year", int(train_df["year"].min()), int(train_df["year"].max()), 2015)
            if "year" in train_df.columns else None
        )
        area_val      = st.number_input("Area Harvested (ha)",                     min_value=0.0, value=col_median("area_harvested_ha"),                    step=100.0)
        n_mineral_val = st.number_input("Nitrogen Mineral Fertilizer (kg/ha)",     min_value=0.0, value=col_median("nitrogen_mineral_fertilizer_kg_ha"),     step=1.0)
        p_mineral_val = st.number_input("Phosphorus Mineral Fertilizer (kg/ha)",   min_value=0.0, value=col_median("phosphorus_mineral_fertilizer_kg_ha"),   step=1.0)

    with mc2:
        n_organic_val = st.number_input("Nitrogen Organic Fertilizer (kg/ha)",     min_value=0.0, value=col_median("nitrogen_organic_fertilizer_kg_ha"),     step=1.0)
        n_atm_val     = st.number_input("Nitrogen Atmospheric Deposition (kg/ha)", min_value=0.0, value=col_median("nitrogen_atmospheric_deposition_kg_ha"), step=0.5)
        n_bio_val     = st.number_input("Nitrogen Biological Fixation (kg/ha)",    min_value=0.0, value=col_median("nitrogen_biological_fixation_kg_ha"),    step=0.5)
        n_leach_val   = st.number_input("Nitrogen Leaching (kg/ha)",               min_value=0.0, value=col_median("nitrogen_leaching_kg_ha"),               step=0.5)
        n_seed_val    = st.number_input("Nitrogen Seed (kg/ha)",                   min_value=0.0, value=col_median("nitrogen_seed_kg_ha"),                   step=0.1)

    if st.button("🌽 Predict Yield", use_container_width=True):
        raw = {
            "area_harvested_ha":                     area_val,
            "nitrogen_mineral_fertilizer_kg_ha":     n_mineral_val,
            "phosphorus_mineral_fertilizer_kg_ha":   p_mineral_val,
            "nitrogen_organic_fertilizer_kg_ha":     n_organic_val,
            "nitrogen_atmospheric_deposition_kg_ha": n_atm_val,
            "nitrogen_biological_fixation_kg_ha":    n_bio_val,
            "nitrogen_leaching_kg_ha":               n_leach_val,
            "nitrogen_seed_kg_ha":                   n_seed_val,
        }
        if year_val is not None:
            raw["year"] = year_val

        # One-hot country columns — all False = reference country
        for col in train_columns:
            if col.startswith("country_"):
                raw[col] = False
        ck = f"country_{country_sel}"
        if ck in train_columns:
            raw[ck] = True

        input_df = pd.DataFrame([raw]).reindex(columns=train_columns, fill_value=0)
        single_pred = model.predict(input_df)[0]

        st.markdown("---")
        st.metric("🌽 Predicted Maize Yield", f"{single_pred:,.0f} kg/ha")
        st.caption(f"Reference (baseline) country: **{ref_country}**")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — DYNAMIC EDA (driven entirely by training data contents)
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    st.caption(
        f"Based on the training dataset · **{len(train_df)} rows** · "
        f"**{len(train_df.columns)} columns**"
    )

    num_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # ── 1. Dataset overview ──────────────────────────────────────────────────
    with st.expander("📋 Dataset Overview (descriptive statistics)", expanded=False):
        st.dataframe(train_df.describe().T.round(2), use_container_width=True)
        nulls = train_df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            st.markdown("**Missing values after imputation:**")
            st.dataframe(nulls.rename("Count").to_frame(), use_container_width=True)
        else:
            st.success("✅ No missing values in the training dataset after imputation.")

    # ── 2. Target distribution ───────────────────────────────────────────────
    if TARGET_COL in train_df.columns:
        st.markdown("#### 🎯 Target Variable: Yield Distribution")
        tc1, tc2 = st.columns(2)
        with tc1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(train_df[TARGET_COL], bins=25, color="#2E7D32", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Yield (kg/ha)"); ax.set_ylabel("Count")
            ax.set_title("Overall Yield Distribution")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)

        with tc2:
            if COUNTRY_COL in train_df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                avg = (train_df.groupby(COUNTRY_COL)[TARGET_COL]
                       .mean().sort_values(ascending=False).reset_index())
                sns.barplot(data=avg, x=COUNTRY_COL, y=TARGET_COL,
                            palette="viridis", ax=ax)
                for i, row in avg.iterrows():
                    ax.text(i, row[TARGET_COL] + 15, f"{row[TARGET_COL]:.0f}",
                            ha="center", fontsize=8)
                ax.set_title("Average Yield by Country")
                ax.set_xlabel(""); ax.set_ylabel("Avg Yield (kg/ha)")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout(); st.pyplot(fig)

    # ── 3. Yield by country — boxplot ────────────────────────────────────────
    if COUNTRY_COL in train_df.columns and TARGET_COL in train_df.columns:
        st.markdown("#### 📦 Yield Distribution by Country")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=train_df, x=COUNTRY_COL, y=TARGET_COL,
                    hue=COUNTRY_COL, palette="Set2", legend=False, ax=ax)
        ax.set_xlabel("Country"); ax.set_ylabel("Yield (kg/ha)")
        ax.set_title("Maize Yield by Country")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig)

    # ── 4. Time trends ───────────────────────────────────────────────────────
    if "year" in train_df.columns and TARGET_COL in train_df.columns:
        st.markdown("#### 📅 Trends Over Time")
        tc1, tc2 = st.columns(2)
        with tc1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            if COUNTRY_COL in train_df.columns:
                for cname, grp in train_df.groupby(COUNTRY_COL):
                    ax.plot(grp["year"], grp[TARGET_COL], label=cname, marker="o", markersize=2)
                ax.legend(fontsize=7)
            else:
                ax.plot(train_df["year"], train_df[TARGET_COL], color="#2E7D32", marker="o", markersize=2)
            ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg/ha)")
            ax.set_title("Yield Trends Over Time")
            ax.grid(linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)

        with tc2:
            # Pick the nutrient feature most correlated with yield
            candidates = [c for c in IMPUTE_COLS if c in train_df.columns]
            if candidates and COUNTRY_COL in train_df.columns:
                corr_vals = {c: abs(train_df[c].corr(train_df[TARGET_COL]))
                             for c in candidates if TARGET_COL in train_df.columns}
                best_feat = max(corr_vals, key=corr_vals.get) if corr_vals else candidates[0]
                fig, ax = plt.subplots(figsize=(6, 3.5))
                for cname, grp in train_df.groupby(COUNTRY_COL):
                    ax.plot(grp["year"], grp[best_feat], label=cname, marker="o", markersize=2)
                ax.set_xlabel("Year")
                ax.set_ylabel(best_feat.replace("_", " ").title())
                ax.set_title(f"{best_feat.replace('_', ' ').title()} Over Time")
                ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.5)
                plt.tight_layout(); st.pyplot(fig)

    # ── 5. Numeric feature boxplots by country ───────────────────────────────
    feature_cols = [c for c in num_cols if c not in [TARGET_COL, "year"]]
    if feature_cols and COUNTRY_COL in train_df.columns:
        st.markdown("#### 📦 Feature Distributions by Country")
        n_grid = 3
        rows = (len(feature_cols) + n_grid - 1) // n_grid
        fig, axes = plt.subplots(rows, n_grid, figsize=(n_grid * 5, rows * 3.5))
        axes = np.array(axes).flatten()
        for i, col in enumerate(feature_cols):
            sns.boxplot(data=train_df, x=COUNTRY_COL, y=col, hue=COUNTRY_COL,
                        palette="Set2", legend=False, ax=axes[i])
            axes[i].set_title(col.replace("_", " ").title(), fontsize=9)
            axes[i].set_xlabel(""); axes[i].set_ylabel("")
            axes[i].tick_params(axis="x", rotation=30, labelsize=7)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(); st.pyplot(fig)

    # ── 6. Correlation heatmap ────────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(
            figsize=(max(8, len(num_cols) * 1.1), max(5, len(num_cols) * 0.9))
        )
        corr = train_df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="viridis",
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Pearson Correlation — Numeric Features")
        plt.tight_layout(); st.pyplot(fig)

    # ── 7. Top features vs yield (scatter) ────────────────────────────────────
    if TARGET_COL in train_df.columns and len(num_cols) > 1:
        st.markdown("#### 🔍 Top 4 Features vs Yield")
        corr_target = (
            train_df[num_cols].corr()[TARGET_COL]
            .drop(TARGET_COL, errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )
        top4 = corr_target.head(4).index.tolist()
        if top4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            hue_kw = (
                {"hue": COUNTRY_COL, "palette": "viridis"}
                if COUNTRY_COL in train_df.columns else {}
            )
            for i, feat in enumerate(top4):
                sns.scatterplot(data=train_df, x=feat, y=TARGET_COL,
                                alpha=0.7, ax=axes[i], **hue_kw)
                axes[i].set_title(f"Yield vs {feat.replace('_', ' ').title()}", fontsize=10)
                axes[i].set_xlabel(feat.replace("_", " "))
                axes[i].set_ylabel("Yield (kg/ha)")
                axes[i].grid(linestyle="--", alpha=0.4)
                if hue_kw and axes[i].get_legend():
                    axes[i].get_legend().remove()
            plt.tight_layout(); st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 CatBoost — Test Set Performance")
    st.caption("Trained on 80% of the training data · Evaluated on the remaining 20%.")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{metrics['MAE']:.1f} kg/ha",  help="Mean Absolute Error")
    m2.metric("RMSE", f"{metrics['RMSE']:.1f} kg/ha", help="Root Mean Squared Error")
    m3.metric("R²",   f"{metrics['R²']:.4f}",          help="Share of variance explained")

    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown("**Predicted vs Actual**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred_test, alpha=0.65, color="#2E7D32",
                   edgecolors="white", linewidths=0.4)
        lo = float(min(y_test.min(), y_pred_test.min())) - 50
        hi = float(max(y_test.max(), y_pred_test.max())) + 50
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual Yield (kg/ha)"); ax.set_ylabel("Predicted Yield (kg/ha)")
        ax.set_title("Predicted vs Actual"); ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig)

    with pc2:
        st.markdown("**Residual Plot**")
        residuals = y_pred_test - np.array(y_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_pred_test, residuals, alpha=0.65, color="#1565C0",
                   edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Predicted Yield (kg/ha)")
        ax.set_ylabel("Residual (Predicted − Actual)")
        ax.set_title("Residuals")
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("**Feature Importance**")
    imp_df = pd.DataFrame({
        "Feature":    train_columns,
        "Importance": model.get_feature_importance(),
    }).sort_values("Importance", ascending=True)
    colors = [
        "#2E7D32" if not f.startswith("country_") else "#81C784"
        for f in imp_df["Feature"]
    ]
    fig, ax = plt.subplots(figsize=(9, max(3, len(imp_df) * 0.38)))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("CatBoost Feature Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    st.caption("🟢 Dark green = numeric features · Light green = country dummy variables")
