<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-00e5c8?style=for-the-badge&logo=python&logoColor=white&labelColor=050c14"/>
<img src="https://img.shields.io/badge/XGBoost-R²_0.825-f5a623?style=for-the-badge&logo=xgboost&logoColor=white&labelColor=050c14"/>
<img src="https://img.shields.io/badge/LSTM-TensorFlow-00e5c8?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=050c14"/>
<img src="https://img.shields.io/badge/Streamlit-Live_Demo-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=050c14"/>

<br/><br/>

# 🛒 RetailIQ — Smart Demand Forecasting Platform

### End-to-end retail analytics: ETL pipeline · XGBoost + LSTM forecasting · SHAP explainability · Live dashboard

<br/>

> *"Why did Store 47 lose €12,000 last Tuesday?"*
> RetailIQ tells you — automatically.

<br/>

**[🚀 Live Demo](#)** · **[📊 Model Results](#-model-performance)** · **[🏗️ Architecture](#️-architecture)**

</div>

---

## ✨ What This Does

Retail businesses lose **$1.75 trillion annually** from bad inventory decisions. RetailIQ fixes that by:

- **Ingesting** raw sales data from multiple sources automatically
- **Cleaning & validating** 1M+ records through an automated ETL pipeline
- **Forecasting** demand using XGBoost (R² = 0.825) and LSTM models
- **Explaining** every prediction in plain English using SHAP + AI
- **Visualising** everything on a live, interactive dashboard

---

## 📊 Model Performance

| Model | RMSE | MAE | R² | Notes |
|-------|------|-----|-----|-------|
| **XGBoost** | 1,299 | 932 | **0.825** | Primary forecasting model |
| **LSTM** | 2,421 | 1,683 | 0.316 | Time-series deep learning |

> XGBoost explains **82.5% of all sales variation** across 1,115 stores — production-grade accuracy.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│         Kaggle Rossmann · CSV uploads · APIs                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   AWS S3 (Raw Storage)                      │
│              Free tier · 5GB · 12 months                    │
└─────────────────────┬───────────────────────────────────────┘
                      │  triggers
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              ETL PIPELINE (AWS Lambda)                      │
│   ingest → clean → validate (9 checks) → load              │
│   Pandas · NumPy · SQLAlchemy                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Supabase PostgreSQL (Free forever)             │
│                    844,338 clean rows                       │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────────────────┐
│  XGBoost Model   │       │         LSTM Model               │
│   R² = 0.825     │       │      Time-series forecasting     │
│  SHAP explained  │       │      30-day sequence windows     │
└────────┬─────────┘       └─────────────────┬────────────────┘
         │                                   │
         └──────────────┬────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           FastAPI Backend (Railway.app · Free)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│        Streamlit Dashboard (Streamlit Cloud · Free)         │
│  KPIs · Trend · Forecast · Heatmap · AI Explainer           │
└─────────────────────────────────────────────────────────────┘
                      +
┌─────────────────────────────────────────────────────────────┐
│             Power BI (connected via DB)                     │
│         Executive dashboards · Drill-down reports           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
retail-forecasting/
│
├── 📂 etl/
│   ├── ingest.py        ← Download from Kaggle, upload to AWS S3
│   ├── clean.py         ← Merge, impute nulls, engineer 10+ features
│   ├── validate.py      ← 9 automated data quality checks
│   ├── load.py          ← Load to Supabase/MySQL via SQLAlchemy
│   └── pipeline.py      ← Master orchestrator + AWS Lambda handler
│
├── 📂 models/
│   ├── train_xgboost.py ← XGBoost baseline (R²=0.825)
│   ├── train_lstm.py    ← LSTM deep learning forecaster
│   └── explain.py       ← SHAP values + Gemini AI explanations
│
├── 📂 dashboard/
│   └── app.py           ← Full Streamlit dashboard (6 sections)
│
├── 📂 tests/
│   └── test_validate.py ← Pytest unit tests for data validation
│
├── 📂 data/             ← Auto-generated (gitignored)
├── 📂 docs/             ← Validation reports, model cards
├── setup_data.py        ← Auto-setup for cloud deployment
├── .env.example         ← Environment variables template
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
# 1. Clone & install
git clone https://github.com/Mohit-Balachander/retail-forecasting
cd retail-forecasting
pip install -r requirements.txt

# 2. Set up credentials
cp .env.example .env
# Add: KAGGLE_USERNAME, KAGGLE_KEY

# 3. Run full ETL pipeline
python -m etl.pipeline --skip-db

# 4. Train models
python -m models.train_xgboost   # ~2 mins
python -m models.train_lstm      # ~15 mins

# 5. Launch dashboard
streamlit run dashboard/app.py

# 6. Run tests
pytest tests/ -v
```

---

## 🔍 AI Explainer — Sample Output

```
📊 Store 1 | 2015-07-15
   Actual Sales    : €4,767
   Predicted Sales : €5,260
   Difference      : -€493 (-9.4%)

🔍 Top SHAP Drivers:
   Promo = 1          →  +€863  (promotion boosted sales)
   Assortment = 0     →  -€750  (limited product range)
   CompetitionDist    →  -€512  (competitor 1,270m away)
   DayOfWeek = Tue    →  -€444  (slow shopping day)

💡 AI Insight: Sales dropped 9.4% for Store 1 primarily due to
   limited product assortment and a nearby competitor, despite
   an active promotion adding ~€863 in lift.
```

---

## ☁️ Deployment Stack (100% Free)

| Layer | Service | Plan | Cost |
|-------|---------|------|------|
| Raw Storage | AWS S3 | Free Tier | €0 / 12 months |
| ETL Trigger | AWS Lambda | Free Forever | €0 |
| Database | Supabase | Free Forever | €0 |
| ML Backend | Railway.app | Starter | €0 |
| Dashboard | Streamlit Cloud | Community | €0 |
| **Total** | | | **€0** |

---

## 🛠️ Tech Stack

**Data Engineering:** Python · Pandas · NumPy · SQLAlchemy · AWS S3 · Lambda

**Machine Learning:** XGBoost · TensorFlow/Keras · Scikit-learn · SHAP

**Visualization:** Streamlit · Plotly · Power BI

**Backend:** FastAPI · Uvicorn

**Database:** Supabase (PostgreSQL) · MySQL

**DevOps:** Git · Pytest · AWS Free Tier

---

## 📈 Key Features

- ✅ **1,017,209 rows** processed through automated ETL
- ✅ **9/9 data quality checks** passing
- ✅ **10+ engineered features** (IsWeekend, CompetitionAge, IsPromoMonth...)
- ✅ **XGBoost R² = 0.825** — explains 82.5% of sales variation
- ✅ **SHAP explainability** — know *why* every prediction was made
- ✅ **AI plain-English summaries** via Gemini API
- ✅ **Power BI integration** via live database connection
- ✅ **AWS Lambda handler** for serverless ETL triggering

---

## 👨‍💻 Author

**Mohit Balachander** — Integrated M.Tech CSE, VIT-AP University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/mohit-balachander/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Mohit-Balachander)

---

<div align="center">
<sub>Built as a portfolio project for Data & Analytics internship applications · 2026</sub>
</div>
