"""
models/explain.py
-----------------
SHAP Explainability + Gemini AI "Why did sales drop?" module
- Computes SHAP values for any prediction
- Calls Gemini API to generate plain English explanation
- Example: "Sales dropped 23% due to no promotion + school holiday"
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SAVED_DIR  = Path("models/saved")
REPORT_DIR = Path("docs")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "Promo2", "Year", "Month", "Day", "WeekOfYear",
    "IsWeekend", "Quarter", "CompetitionOpenMonths", "IsPromoMonth"
]


# ─────────────────────────────────────────────
# 1. Load trained XGBoost model
# ─────────────────────────────────────────────
def load_xgboost_model():
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(SAVED_DIR / "xgboost_model.json"))
    print("✅ XGBoost model loaded.")
    return model


# ─────────────────────────────────────────────
# 2. Compute SHAP values
# ─────────────────────────────────────────────
def compute_shap(model, X: pd.DataFrame) -> dict:
    """
    Computes SHAP values for a single prediction row.
    Returns top factors driving the prediction (positive = increases sales).
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame({
            "feature": X.columns,
            "value":   X.iloc[0].values,
            "shap":    shap_values[0]
        }).sort_values("shap", key=abs, ascending=False)

        print("\n🔍 SHAP — Top drivers of this prediction:")
        print(shap_df.head(8).to_string(index=False))

        return shap_df.head(8).to_dict(orient="records")

    except ImportError:
        print("⚠️  SHAP not installed. Run: pip install shap")
        return []


# ─────────────────────────────────────────────
# 3. Gemini AI — plain English explanation
# ─────────────────────────────────────────────
def explain_with_gemini(
    store_id: int,
    date: str,
    actual_sales: float,
    predicted_sales: float,
    shap_factors: list
) -> str:
    """
    Sends SHAP factors to Gemini and gets a plain English business explanation.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key":
        # Fallback: rule-based explanation without API
        return generate_rule_based_explanation(
            store_id, date, actual_sales, predicted_sales, shap_factors
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")

        pct_change = ((actual_sales - predicted_sales) / predicted_sales) * 100
        direction  = "dropped" if pct_change < 0 else "increased"

        top_factors = "\n".join([
            f"- {f['feature']}: value={f['value']}, impact={f['shap']:+.0f} sales units"
            for f in shap_factors[:5]
        ])

        prompt = f"""
You are a retail business analyst. Explain in 2 clear sentences why sales {direction} for this store.

Store: {store_id}
Date: {date}
Actual Sales: {actual_sales:,.0f}
Predicted Sales: {predicted_sales:,.0f}
Change: {pct_change:+.1f}%

Top SHAP factors (what drove this prediction):
{top_factors}

Write a business-friendly explanation. Be specific. Mention the top 2 factors.
Example format: "Sales dropped 23% in Week 12 due to [reason 1] and [reason 2]."
"""
        response = model.generate_content(prompt)
        explanation = response.text.strip()
        print(f"\n🤖 Gemini Explanation:\n{explanation}")
        return explanation

    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return generate_rule_based_explanation(
            store_id, date, actual_sales, predicted_sales, shap_factors
        )


def generate_rule_based_explanation(
    store_id, date, actual_sales, predicted_sales, shap_factors
) -> str:
    """Fallback: generates explanation from SHAP without API call."""
    pct_change = ((actual_sales - predicted_sales) / predicted_sales) * 100
    direction  = "dropped" if pct_change < 0 else "increased"

    top = shap_factors[:2] if len(shap_factors) >= 2 else shap_factors

    reasons = []
    for f in top:
        fname = f["feature"]
        fval  = f["value"]
        impact = f["shap"]

        if fname == "Promo" and fval == 0:
            reasons.append("no active promotion")
        elif fname == "Promo" and fval == 1:
            reasons.append("active promotion boosting traffic")
        elif fname == "IsWeekend" and fval == 1:
            reasons.append("weekend shopping pattern")
        elif fname == "SchoolHoliday" and fval == 1:
            reasons.append("school holiday reducing foot traffic")
        elif fname == "CompetitionDistance":
            reasons.append(f"nearby competitor at {fval:.0f}m")
        elif fname == "Month":
            months = {1:"January",2:"February",3:"March",4:"April",5:"May",
                     6:"June",7:"July",8:"August",9:"September",10:"October",
                     11:"November",12:"December"}
            reasons.append(f"seasonal pattern in {months.get(int(fval), 'this month')}")
        else:
            reasons.append(f"{fname} = {fval}")

    reason_str = " and ".join(reasons) if reasons else "multiple contributing factors"
    explanation = (
        f"Sales {direction} {abs(pct_change):.1f}% for Store {store_id} on {date} "
        f"due to {reason_str}. "
        f"Actual: {actual_sales:,.0f} vs predicted: {predicted_sales:,.0f} units."
    )
    return explanation


# ─────────────────────────────────────────────
# 4. Full explain pipeline
# ─────────────────────────────────────────────
def run_explanation(store_id: int = 1, date: str = "2015-07-15"):
    """
    Runs full explanation for a specific store + date.
    """
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings("ignore")

    # Load data
    df = pd.read_csv("data/validated/validated_train.csv")
    row = df[(df["Store"] == store_id) & (df["Date"] == date)]

    if row.empty:
        print(f"⚠️  No data for Store {store_id} on {date}. Using sample row.")
        row = df[df["Store"] == store_id].iloc[[0]]

    # Prepare features
    le = LabelEncoder()
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            if col in row.columns:
                row = row.copy()
                row[col] = le.transform(row[col].astype(str))

    available = [c for c in FEATURE_COLS if c in row.columns]
    X = row[available]
    actual_sales = float(row["Sales"].values[0])

    # Load model + predict
    model        = load_xgboost_model()
    predicted    = float(model.predict(X)[0])

    print(f"\n📊 Store {store_id} | {date}")
    print(f"   Actual Sales   : {actual_sales:,.0f}")
    print(f"   Predicted Sales: {predicted:,.0f}")
    print(f"   Difference     : {actual_sales - predicted:+,.0f}")

    # SHAP
    shap_factors = compute_shap(model, X)

    # Explanation
    explanation = explain_with_gemini(
        store_id, date, actual_sales, predicted, shap_factors
    )

    # Save report
    report = {
        "store_id":        store_id,
        "date":            date,
        "actual_sales":    actual_sales,
        "predicted_sales": round(predicted, 2),
        "explanation":     explanation,
        "shap_factors":    shap_factors,
    }
    out = REPORT_DIR / f"explanation_store{store_id}_{date}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Explanation saved → {out}")
    print(f"\n💡 Summary:\n{explanation}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--date",  type=str, default="2015-07-15")
    args = parser.parse_args()
    run_explanation(store_id=args.store, date=args.date)
