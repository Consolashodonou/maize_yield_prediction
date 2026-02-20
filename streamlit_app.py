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
    page_title="Yieldlens | Maize Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
.main-banner {
    background: linear-gradient(90deg, #2E7D32, #66BB6A);
    padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;
}
.main-banner h1 { margin-bottom: 5px; }
.subtitle { font-size: 18px; margin-top: 0px; }
.tagline { font-style: italic; font-size: 14px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 6])
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

# ── Utility functions ─────────────────────────────────────────────────────────
def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def impute(df):
    cols = [c for c in IMPUTE_COLS if c in df.columns]
    if not cols:
        return df
    if COUNTRY_COL in df.columns:
        df[cols] = df.groupby(COUNTRY_COL)[cols].transform(
            lambda s: s.fillna(s.median())
        )
    else:
        for c in cols:
            df[c] = df[c].fillna(df[c].median())
    return df

def to_float(df):
    """Cast boolean columns (pandas ≥ 2.0 get_dummies output) to float64."""
    bools = df.select_dtypes(include="bool").columns
    if len(bools):
        df[bools] = df[bools].astype(np.float64)
    return df

# ── Training functions (cached by file bytes) ─────────────────────────────────
@st.cache_data(show_spinner=False)
def load_train(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = normalize_cols(df)
    missing = [c for c in [TARGET_COL, COUNTRY_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from training file: {missing}\n"
            f"Found: {df.columns.tolist()}"
        )
    df = df.drop(columns=[DROP_COL], errors="ignore")
    df = impute(df)
    return df

@st.cache_resource(show_spinner=False)
def fit_model(file_bytes):
    df     = load_train(file_bytes)
    X      = df.drop(columns=[TARGET_COL])
    y      = df[TARGET_COL]
    X_enc  = pd.get_dummies(X, columns=[COUNTRY_COL], drop_first=True)
    X_enc  = to_float(X_enc)
    col_names = X_enc.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.2, random_state=42)
    mdl = CatBoostRegressor(random_state=42, verbose=0, iterations=200)
    mdl.fit(X_tr, y_tr)
    preds = mdl.predict(X_te)

    metrics = {
        "MAE" : mean_absolute_error(y_te, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, preds))),
        "R²"  : r2_score(y_te, preds),
    }
    return mdl, col_names, X_te, y_te, preds, metrics

# ── Sidebar: training file only ───────────────────────────────────────────────
st.sidebar.header("📂 Training Dataset")
st.sidebar.caption(f"Upload a CSV that includes `{TARGET_COL}`.")
train_file = st.sidebar.file_uploader("Upload training CSV", type="csv", key="train")

if train_file is None:
    st.info("👈 Upload your **training CSV** in the sidebar to get started.")
    st.stop()

# ── Load & train ──────────────────────────────────────────────────────────────
train_bytes = train_file.read()

try:
    with st.spinner("Loading and training…"):
        train_df = load_train(train_bytes)
        model, train_cols, X_te, y_te, preds_te, metrics = fit_model(train_bytes)
except Exception as e:
    st.error("❌ Training failed. Details:")
    st.exception(e)
    st.stop()

st.success(
    f"✅ Model ready — trained on **{len(train_df)} rows** | "
    f"R² = **{metrics['R²']:.4f}** | RMSE = **{metrics['RMSE']:.1f} kg/ha**"
)

COUNTRIES   = sorted(train_df[COUNTRY_COL].unique().tolist())
ref_country = COUNTRIES[0]
col_med     = lambda c: float(train_df[c].median()) if c in train_df.columns else 0.0

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA", "📈 Model Performance"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── SECTION A: Batch prediction from uploaded CSV ─────────────────────────
    st.subheader("📋 Batch Prediction — Upload a Test File")
    st.markdown(
        "Upload a CSV **without** `yield_kg_ha`. "
        "It must have the same feature columns as your training file."
    )

    # ── File uploader lives here in Tab 1 — NOT in sidebar ────────────────────
    pred_file = st.file_uploader(
        "Upload prediction CSV (no yield column)",
        type="csv",
        key="pred"
    )

    if pred_file is not None:
        # ── Step 1: read bytes immediately ────────────────────────────────────
        pred_bytes = pred_file.read()

        # ── Step 2: show a debug expander so the user always sees what happened
        with st.expander("🔍 Debug info (click to expand if something looks wrong)", expanded=False):
            try:
                raw_preview = pd.read_csv(io.BytesIO(pred_bytes))
                st.write(f"**Rows read:** {len(raw_preview)} | **Columns read:** {list(raw_preview.columns)}")
                st.write(f"**Training model columns:** {train_cols}")
            except Exception as debug_e:
                st.error(f"Could not even read the CSV: {debug_e}")

        # ── Step 3: full pipeline inline — error shown right here ─────────────
        try:
            # Load and clean
            pred_df = pd.read_csv(io.BytesIO(pred_bytes))
            pred_df = normalize_cols(pred_df)

            # Strip target and potassium if accidentally present
            pred_df = pred_df.drop(columns=[TARGET_COL, DROP_COL], errors="ignore")

            # Impute missing values
            pred_df = impute(pred_df)

            # One-hot encode country — drop_first=False then reindex to training schema
            if COUNTRY_COL in pred_df.columns:
                pred_enc = pd.get_dummies(pred_df, columns=[COUNTRY_COL], drop_first=False)
            else:
                pred_enc = pred_df.copy()

            # Align to exact training column order; fill any missing dummy with 0
            pred_enc = pred_enc.reindex(columns=train_cols, fill_value=0)

            # Cast booleans to float (pandas ≥ 2.0 fix)
            pred_enc = to_float(pred_enc)

            # Ensure all columns are numeric — catch stray text columns
            non_num = pred_enc.select_dtypes(exclude="number").columns.tolist()
            if non_num:
                raise ValueError(
                    f"Non-numeric columns found after encoding: {non_num}. "
                    "Check your CSV for unexpected text values."
                )

            # Predict
            predictions = model.predict(pred_enc)

            # Build results dataframe
            results = pred_df.copy()
            results.insert(0, "predicted_yield_kg_ha", np.round(predictions, 1))

            # ── Show results ──────────────────────────────────────────────────
            st.success(f"✅ Predictions generated for **{len(results)} rows**.")
            st.dataframe(results, use_container_width=True)

            # Download button
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=csv_out,
                file_name="maize_yield_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Summary metrics
            st.markdown("#### 📊 Prediction Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Min",    f"{results['predicted_yield_kg_ha'].min():,.0f} kg/ha")
            m2.metric("Max",    f"{results['predicted_yield_kg_ha'].max():,.0f} kg/ha")
            m3.metric("Mean",   f"{results['predicted_yield_kg_ha'].mean():,.0f} kg/ha")
            m4.metric("Median", f"{results['predicted_yield_kg_ha'].median():,.0f} kg/ha")

            # Charts
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(results["predicted_yield_kg_ha"], bins=min(20, len(results)),
                        color="#2E7D32", edgecolor="white", alpha=0.85)
                ax.set_xlabel("Predicted Yield (kg/ha)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Predicted Yields")
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                plt.tight_layout(); st.pyplot(fig)

            with c2:
                if COUNTRY_COL in results.columns:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    sns.boxplot(data=results, x=COUNTRY_COL,
                                y="predicted_yield_kg_ha", hue=COUNTRY_COL,
                                palette="Set2", legend=False, ax=ax)
                    ax.set_title("Predicted Yield by Country")
                    ax.set_xlabel(""); ax.set_ylabel("Predicted Yield (kg/ha)")
                    ax.tick_params(axis="x", rotation=30)
                    plt.tight_layout(); st.pyplot(fig)
                elif "year" in results.columns:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.scatter(results["year"], results["predicted_yield_kg_ha"],
                               color="#2E7D32", alpha=0.7)
                    ax.set_xlabel("Year"); ax.set_ylabel("Predicted Yield (kg/ha)")
                    ax.set_title("Predicted Yield Over Time")
                    ax.grid(linestyle="--", alpha=0.5)
                    plt.tight_layout(); st.pyplot(fig)

        except Exception as e:
            # ── Show the error HERE in Tab 1 — never hidden ───────────────────
            st.error("❌ Prediction failed. Full error below:")
            st.exception(e)
            st.markdown("---")
            st.markdown("**Expected columns (from your training file):**")
            st.code("\n".join(train_cols))

    # ── SECTION B: Manual single-row prediction ───────────────────────────────
    st.markdown("---")
    st.subheader("🖊️ Manual Single-Row Prediction")
    st.caption("Fill in values and click Predict.")

    c1, c2 = st.columns(2)
    with c1:
        sel_country = st.selectbox("Country", COUNTRIES)
        sel_year    = (
            st.slider("Year", int(train_df["year"].min()), int(train_df["year"].max()), 2015)
            if "year" in train_df.columns else None
        )
        v_area  = st.number_input("Area Harvested (ha)",                     min_value=0.0, value=col_med("area_harvested_ha"),                    step=100.0)
        v_nmin  = st.number_input("Nitrogen Mineral Fertilizer (kg/ha)",     min_value=0.0, value=col_med("nitrogen_mineral_fertilizer_kg_ha"),     step=1.0)
        v_pmin  = st.number_input("Phosphorus Mineral Fertilizer (kg/ha)",   min_value=0.0, value=col_med("phosphorus_mineral_fertilizer_kg_ha"),   step=1.0)

    with c2:
        v_norg  = st.number_input("Nitrogen Organic Fertilizer (kg/ha)",     min_value=0.0, value=col_med("nitrogen_organic_fertilizer_kg_ha"),     step=1.0)
        v_natm  = st.number_input("Nitrogen Atmospheric Deposition (kg/ha)", min_value=0.0, value=col_med("nitrogen_atmospheric_deposition_kg_ha"), step=0.5)
        v_nbio  = st.number_input("Nitrogen Biological Fixation (kg/ha)",    min_value=0.0, value=col_med("nitrogen_biological_fixation_kg_ha"),    step=0.5)
        v_nlea  = st.number_input("Nitrogen Leaching (kg/ha)",               min_value=0.0, value=col_med("nitrogen_leaching_kg_ha"),               step=0.5)
        v_nsee  = st.number_input("Nitrogen Seed (kg/ha)",                   min_value=0.0, value=col_med("nitrogen_seed_kg_ha"),                   step=0.1)

    if st.button("🌽 Predict Yield", use_container_width=True):
        try:
            row = {
                "area_harvested_ha":                     v_area,
                "nitrogen_mineral_fertilizer_kg_ha":     v_nmin,
                "phosphorus_mineral_fertilizer_kg_ha":   v_pmin,
                "nitrogen_organic_fertilizer_kg_ha":     v_norg,
                "nitrogen_atmospheric_deposition_kg_ha": v_natm,
                "nitrogen_biological_fixation_kg_ha":    v_nbio,
                "nitrogen_leaching_kg_ha":               v_nlea,
                "nitrogen_seed_kg_ha":                   v_nsee,
            }
            if sel_year is not None:
                row["year"] = sel_year

            # Country dummies — use floats, not booleans
            for col in train_cols:
                if col.startswith("country_"):
                    row[col] = 0.0
            ck = f"country_{sel_country}"
            if ck in train_cols:
                row[ck] = 1.0

            inp = pd.DataFrame([row]).reindex(columns=train_cols, fill_value=0.0)
            result = model.predict(inp)[0]

            st.markdown("---")
            st.metric("🌽 Predicted Maize Yield", f"{result:,.0f} kg/ha")
            st.caption(f"Reference (baseline) country: **{ref_country}**")

        except Exception as e:
            st.error("❌ Manual prediction failed:")
            st.exception(e)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    st.caption(f"Training dataset — {len(train_df)} rows · {len(train_df.columns)} columns")

    num_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Overview
    with st.expander("📋 Descriptive Statistics", expanded=False):
        st.dataframe(train_df.describe().T.round(2), use_container_width=True)
        nulls = train_df.isnull().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            st.success("✅ No missing values after imputation.")
        else:
            st.dataframe(nulls.rename("Missing").to_frame(), use_container_width=True)

    # Target distribution
    if TARGET_COL in train_df.columns:
        st.markdown("#### 🎯 Yield Distribution")
        tc1, tc2 = st.columns(2)
        with tc1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(train_df[TARGET_COL], bins=25, color="#2E7D32", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Yield (kg/ha)"); ax.set_ylabel("Count")
            ax.set_title("Overall Distribution")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)
        with tc2:
            if COUNTRY_COL in train_df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                avg = (train_df.groupby(COUNTRY_COL)[TARGET_COL]
                       .mean().sort_values(ascending=False).reset_index())
                sns.barplot(data=avg, x=COUNTRY_COL, y=TARGET_COL, palette="viridis", ax=ax)
                for i, row in avg.iterrows():
                    ax.text(i, row[TARGET_COL] + 15, f"{row[TARGET_COL]:.0f}",
                            ha="center", fontsize=8)
                ax.set_title("Average Yield by Country")
                ax.set_xlabel(""); ax.set_ylabel("Avg Yield (kg/ha)")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout(); st.pyplot(fig)

    # Yield boxplot by country
    if COUNTRY_COL in train_df.columns and TARGET_COL in train_df.columns:
        st.markdown("#### 📦 Yield by Country")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=train_df, x=COUNTRY_COL, y=TARGET_COL,
                    hue=COUNTRY_COL, palette="Set2", legend=False, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel("Yield (kg/ha)")
        ax.set_title("Maize Yield Distribution by Country")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig)

    # Time trends
    if "year" in train_df.columns and TARGET_COL in train_df.columns:
        st.markdown("#### 📅 Trends Over Time")
        tc1, tc2 = st.columns(2)
        with tc1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for cname, grp in train_df.groupby(COUNTRY_COL):
                ax.plot(grp["year"], grp[TARGET_COL], label=cname, marker="o", markersize=2)
            ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg/ha)")
            ax.set_title("Yield Over Time"); ax.legend(fontsize=7)
            ax.grid(linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig)
        with tc2:
            candidates = [c for c in IMPUTE_COLS if c in train_df.columns]
            if candidates:
                corr_vals = {c: abs(train_df[c].corr(train_df[TARGET_COL])) for c in candidates}
                best = max(corr_vals, key=corr_vals.get)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                for cname, grp in train_df.groupby(COUNTRY_COL):
                    ax.plot(grp["year"], grp[best], label=cname, marker="o", markersize=2)
                ax.set_xlabel("Year"); ax.set_ylabel(best.replace("_", " ").title())
                ax.set_title(f"{best.replace('_', ' ').title()} Over Time")
                ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.5)
                plt.tight_layout(); st.pyplot(fig)

    # Feature distributions
    feat_cols = [c for c in num_cols if c not in [TARGET_COL, "year"]]
    if feat_cols and COUNTRY_COL in train_df.columns:
        st.markdown("#### 📦 Feature Distributions by Country")
        n_grid = 3
        rows = (len(feat_cols) + n_grid - 1) // n_grid
        fig, axes = plt.subplots(rows, n_grid, figsize=(n_grid * 5, rows * 3.5))
        axes = np.array(axes).flatten()
        for i, col in enumerate(feat_cols):
            sns.boxplot(data=train_df, x=COUNTRY_COL, y=col,
                        hue=COUNTRY_COL, palette="Set2", legend=False, ax=axes[i])
            axes[i].set_title(col.replace("_", " ").title(), fontsize=9)
            axes[i].set_xlabel(""); axes[i].set_ylabel("")
            axes[i].tick_params(axis="x", rotation=30, labelsize=7)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(); st.pyplot(fig)

    # Correlation heatmap
    if len(num_cols) >= 2:
        st.markdown("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 1.1), max(5, len(num_cols) * 0.9)))
        corr = train_df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="viridis",
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Pearson Correlation — Numeric Features")
        plt.tight_layout(); st.pyplot(fig)

    # Top 4 scatter
    if TARGET_COL in train_df.columns and len(num_cols) > 1:
        st.markdown("#### 🔍 Top 4 Features vs Yield")
        top4 = (
            train_df[num_cols].corr()[TARGET_COL]
            .drop(TARGET_COL, errors="ignore").abs()
            .sort_values(ascending=False).head(4).index.tolist()
        )
        if top4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            hkw = {"hue": COUNTRY_COL, "palette": "viridis"} if COUNTRY_COL in train_df.columns else {}
            for i, feat in enumerate(top4):
                sns.scatterplot(data=train_df, x=feat, y=TARGET_COL,
                                alpha=0.7, ax=axes[i], **hkw)
                axes[i].set_title(f"Yield vs {feat.replace('_', ' ').title()}", fontsize=10)
                axes[i].set_xlabel(feat.replace("_", " "))
                axes[i].set_ylabel("Yield (kg/ha)")
                axes[i].grid(linestyle="--", alpha=0.4)
                if hkw and axes[i].get_legend():
                    axes[i].get_legend().remove()
            plt.tight_layout(); st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 CatBoost — Test Set Performance")
    st.caption("80% train / 20% test split · random_state = 42")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{metrics['MAE']:.1f} kg/ha",  help="Mean Absolute Error")
    m2.metric("RMSE", f"{metrics['RMSE']:.1f} kg/ha", help="Root Mean Squared Error")
    m3.metric("R²",   f"{metrics['R²']:.4f}",          help="Variance explained")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Predicted vs Actual**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_te, preds_te, alpha=0.65, color="#2E7D32", edgecolors="white", linewidths=0.4)
        lo = float(min(y_te.min(), preds_te.min())) - 50
        hi = float(max(y_te.max(), preds_te.max())) + 50
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual (kg/ha)"); ax.set_ylabel("Predicted (kg/ha)")
        ax.set_title("Predicted vs Actual"); ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig)

    with pc2:
        st.markdown("**Residuals**")
        resid = preds_te - np.array(y_te)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(preds_te, resid, alpha=0.65, color="#1565C0",
                   edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Predicted (kg/ha)"); ax.set_ylabel("Residual")
        ax.set_title("Residuals"); ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("**Feature Importance**")
    imp = pd.DataFrame({
        "Feature":    train_cols,
        "Importance": model.get_feature_importance()
    }).sort_values("Importance", ascending=True)
    colors = ["#2E7D32" if not f.startswith("country_") else "#81C784" for f in imp["Feature"]]
    fig, ax = plt.subplots(figsize=(9, max(3, len(imp) * 0.38)))
    ax.barh(imp["Feature"], imp["Importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("CatBoost Feature Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig)
    st.caption("🟢 Dark green = numeric features · Light green = country dummies")
