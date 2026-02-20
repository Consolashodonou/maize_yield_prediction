# 🌾 Yieldlens

**AI-Powered Maize Yield Prediction for Smart Farming**
**Problem**: Maize is a critical staple crop. Inaccurate predictions hinder effective agricultural planning and resource management for millions. This project is created in the context of Women in Data Science Fellowship. program by AWARD.

Yieldlens is a **Streamlit web application** that predicts maize yield (kg/ha) using ensemble machine learning.
The best-performing model is **CatBoost Regressor (R² ≈ 0.84)**.

It supports both **model training** and **batch/single prediction** workflows.

---

## 🌍 Coverage

Currently supports maize yield prediction for:

* Benin
* Ethiopia
* Ivory Coast
* Kenya
* Malawi

---

## 🚀 Features

* 📂 Upload training dataset (with `yield_kg_ha`)
* 🤖 Automatic CatBoost model training
* 📈 Model performance evaluation (MAE, RMSE, R²)
* 🔮 Batch prediction from new datasets
* 🖊 Manual single-row prediction
* 📊 Interactive EDA dashboard
* 🔥 Feature importance visualization
* 💾 Optional loading of pre-trained model (`.pkl`)

---

## 🧠 Model

* **Algorithm:** CatBoost Regressor
* **Train/Test split:** 80/20
* **Encoding:** One-hot encoding for country
* **Missing values:** Country-wise median imputation

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/yieldlens.git
cd yieldlens
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 📂 Required Columns (Training File)

Your training CSV must include:

* `yield_kg_ha` (target)
* `country`
* Fertilizer and nitrogen-related features
* Optional: `year`

Prediction files should **NOT include `yield_kg_ha`**.

---

## 📊 Example Workflow

1. Upload training dataset
2. Review EDA and model performance
3. Upload prediction dataset (optional)
4. Download predicted yields

---

## 🛠 Tech Stack

* Python
* Streamlit
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* CatBoost

---

## 📌 Project Structure

```
yieldlens/
│
├── app.py
├── catboost_yield_model.pkl (optional)
├── requirements.txt
└── README.md
```

---

## 🌱 Vision

Yieldlens aims to support **data-driven agriculture in Africa**, enabling farmers, researchers, and policymakers to better anticipate crop productivity.

> “See the Future of Your Harvest.”
