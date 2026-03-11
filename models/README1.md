# Models — Week 3-4

## Files
- `train_xgboost.py` — XGBoost baseline forecasting model
- `train_lstm.py`    — LSTM deep learning time-series model  
- `explain.py`       — SHAP + Gemini AI explainability

## Run Order
```bash
# Step 1 — Train XGBoost (fast, ~2 mins)
python -m models.train_xgboost

# Step 2 — Train LSTM (slower, ~10 mins)
python -m models.train_lstm

# Step 3 — Run explainer on any store+date
python -m models.explain --store 1 --date 2015-07-15
```

## Saved Files (auto-created)
```
models/saved/
├── xgboost_model.json       ← trained XGBoost
├── xgboost_metadata.json    ← metrics + feature importance
├── lstm_model.keras         ← trained LSTM
└── lstm_metadata.json       ← metrics
```
