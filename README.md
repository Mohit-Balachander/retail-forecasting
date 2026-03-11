# 🛒 Smart Retail Analytics & Demand Forecasting Platform

> End-to-end data pipeline + ML forecasting + live dashboard — deployed on AWS Free Tier

---

## 📁 Project Structure

```
retail-forecasting/
├── data/
│   ├── raw/            ← Raw CSVs from Kaggle (Rossmann dataset)
│   ├── processed/      ← Cleaned & feature-engineered data
│   └── validated/      ← Quality-checked data ready for DB load
│
├── etl/
│   ├── ingest.py       ← Download from Kaggle, upload to S3
│   ├── clean.py        ← Clean, merge, feature engineer
│   ├── validate.py     ← Data quality checks + validation report
│   ├── load.py         ← Load to Supabase/MySQL
│   └── pipeline.py     ← Master orchestrator (+ AWS Lambda handler)
│
├── models/             ← XGBoost + LSTM forecasting models (Week 3-4)
├── api/                ← FastAPI backend (Week 5)
├── dashboard/          ← Streamlit frontend (Week 5-6)
├── tests/              ← Pytest unit tests
├── docs/               ← Validation reports, model cards, data dictionary
├── notebooks/          ← EDA Jupyter notebooks
├── .env.example        ← Environment variables template
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
# 1. Clone & install
git clone https://github.com/yourname/retail-forecasting
cd retail-forecasting
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Fill in: KAGGLE_USERNAME, KAGGLE_KEY, DB_URL, GEMINI_API_KEY

# 3. Run ETL pipeline
python -m etl.pipeline --skip-db        # local mode (no DB needed)
python -m etl.pipeline                  # full pipeline with DB load

# 4. Run tests
pytest tests/ -v
```

---

## 🏗️ Architecture

```
Kaggle Dataset (Rossmann)
        ↓
   AWS S3 (raw storage)
        ↓
 AWS Lambda (ETL trigger)
        ↓
 Python ETL Pipeline
  ingest → clean → validate → load
        ↓
  Supabase PostgreSQL
        ↓
  ML Models (XGBoost + LSTM)
        ↓
  FastAPI → Streamlit Dashboard
        ↓
  Power BI (connected via DB)
```

---

## ☁️ Deployment (Free Tier)

| Layer | Service | Cost |
|-------|---------|------|
| Raw Storage | AWS S3 | Free 12 months |
| ETL Trigger | AWS Lambda | Free forever |
| Database | Supabase | Free forever |
| Backend | Railway.app | Free |
| Frontend | Streamlit Cloud | Free forever |

---

## 📊 Dataset

**Rossmann Store Sales** — [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)
- 1,017,209 sales records across 1,115 stores
- Features: date, promotions, holidays, competition distance

---

## 🗓️ Build Plan

| Week | Focus |
|------|-------|
| 1–2 | ETL Pipeline (this folder) |
| 3–4 | EDA + ML Models (XGBoost + LSTM + SHAP) |
| 5–6 | FastAPI + Streamlit Dashboard + Power BI |
| 7–8 | AWS Deployment + Documentation |
