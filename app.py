"""
app.py – Churn Enterprise Prediction Backend
Steps 27 + 29 + 30 + 31
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
import shap

# Import preprocessing function directly (avoids pickle __main__ issues)
from preprocessing import preprocess, RAW_FEATURES, TRANSFORMED_NAMES as TRANS_NAMES

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(path):
    with open(os.path.join(BASE_DIR, path), "rb") as f:
        return pickle.load(f)

print("Loading models...")
rf_model    = load_pickle("models/rf_model.pkl")
customer_df = pd.read_csv(
    os.path.join(BASE_DIR, "data/processed/customer_df.csv"),
    index_col=0
)

# SHAP explainer (tree-based, fast)
print("Initialising SHAP explainer...")

# Build transformed training matrix for SHAP background
_X_raw_all = customer_df[RAW_FEATURES].copy()
_X_trans   = preprocess(_X_raw_all)
_explainer = shap.TreeExplainer(rf_model, _X_trans)

print("[OK] All artifacts loaded. Server ready.")

# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS LOGIC (mirrors notebook)
# ─────────────────────────────────────────────────────────────────────────────
def risk_level(probability):
    if   probability >= 0.75: return "Critical"
    elif probability >= 0.50: return "High"
    elif probability >= 0.30: return "Medium"
    else:                     return "Low"

def retention_priority_label(score):
    if   score >= 70: return "P1 - Immediate Action"
    elif score >= 50: return "P2 - High Priority"
    elif score >= 30: return "P3 - Monitor"
    else:             return "P4 - Low Priority"

def retention_action(segment, risk):
    if   risk == "Critical" and segment == "VIP / High-Value Loyal":
        return "Personalized VIP retention offer"
    elif risk == "Critical":
        return "Immediate retention campaign"
    elif risk == "High" and segment == "At-Risk":
        return "Targeted discount / loyalty offer"
    elif risk == "High":
        return "Personalized re-engagement campaign"
    elif risk == "Medium":
        return "Email recommendation / promotional campaign"
    else:
        return "Regular loyalty engagement"

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/customers", methods=["GET"])
def get_customers():
    """Return list of all customer IDs for the search dropdown."""
    ids = customer_df["CustomerID"].dropna().astype(int).sort_values().tolist()
    return jsonify({"customers": ids})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "customer_id": 15361 }
    Returns full prediction + SHAP explanations.
    """
    try:
        data        = request.get_json(force=True)
        customer_id = float(data.get("customer_id", -1))

        row = customer_df[customer_df["CustomerID"] == customer_id]
        if len(row) == 0:
            return jsonify({"error": f"Customer ID {int(customer_id)} not found."}), 404

        row = row.iloc[0]

        # ── Raw features for model ──
        X_raw = pd.DataFrame([row[RAW_FEATURES].values], columns=RAW_FEATURES)
        X_t   = preprocess(X_raw)

        # ── Prediction ──
        churn_proba = float(rf_model.predict_proba(X_t)[0, 1])
        churn_pred  = int(rf_model.predict(X_t)[0])

        # ── Business metrics ──
        total_revenue    = float(row["TotalRevenue"])
        revenue_at_risk  = total_revenue * churn_proba
        risk             = risk_level(churn_proba)
        segment          = str(row["CustomerSegment"])

        # Retention priority score (relative to dataset max)
        max_rev          = float(customer_df["TotalRevenue"].max())
        priority_score   = (churn_proba * 0.6 + (revenue_at_risk / (max_rev * churn_proba + 1e-9)) * 0.4) * 100
        priority_score   = min(round(priority_score, 2), 100.0)
        priority_label   = retention_priority_label(priority_score)
        action           = retention_action(segment, risk)

        # ── SHAP explanation ──
        shap_vals = _explainer.shap_values(X_t)
        # shap_vals shape: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if isinstance(shap_vals, list):
            sv_churn = shap_vals[1][0]       # class 1 (churn), first sample
        elif shap_vals.ndim == 3:
            sv_churn = shap_vals[0, :, 1]    # first sample, class 1
        else:
            sv_churn = shap_vals[0]

        shap_data = sorted(
            [
                {
                    "feature": TRANS_NAMES[i],
                    "value":   round(float(X_t.iloc[0, i]), 4),
                    "shap":    round(float(sv_churn[i]),     4),
                }
                for i in range(len(TRANS_NAMES))
            ],
            key=lambda x: abs(x["shap"]),
            reverse=True,
        )[:6]   # top-6 drivers

        return jsonify({
            "customer_id":          int(customer_id),
            "customer_segment":     segment,
            "country":              str(row.get("Country", "N/A")),
            "churn_probability":    round(churn_proba * 100, 2),
            "churn_prediction":     churn_pred,
            "total_revenue":        round(total_revenue, 2),
            "revenue_at_risk":      round(revenue_at_risk, 2),
            "risk_level":           risk,
            "priority_score":       priority_score,
            "priority_label":       priority_label,
            "recommended_action":   action,
            # Customer profile stats
            "total_orders":         int(row["TotalOrders"]),
            "total_products":       int(row["TotalProducts"]),
            "recency_days":         int(row["RecencyDays"]),
            "tenure_days":          int(row["CustomerTenureDays"]),
            "avg_order_value":      round(float(row["AverageOrderValue"]), 2),
            "return_quantity_rate": round(float(row["ReturnQuantityRate"]) * 100, 2),
            # SHAP
            "shap_explanations":    shap_data,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Dashboard-level statistics."""
    df = customer_df
    total = len(df)

    critical = int((df["RiskLevel"] == "Critical").sum())  if "RiskLevel" in df.columns else 0
    high     = int((df["RiskLevel"] == "High").sum())      if "RiskLevel" in df.columns else 0
    medium   = int((df["RiskLevel"] == "Medium").sum())    if "RiskLevel" in df.columns else 0
    low      = int((df["RiskLevel"] == "Low").sum())       if "RiskLevel" in df.columns else 0

    total_rev     = round(float(df["TotalRevenue"].sum()),     2)
    rev_at_risk   = round(float(df["RevenueAtRisk"].sum()),    2) if "RevenueAtRisk" in df.columns else 0
    avg_churn     = round(float(df["ChurnProbability"].mean()) * 100, 2) if "ChurnProbability" in df.columns else 0

    return jsonify({
        "total_customers":   total,
        "critical":          critical,
        "high":              high,
        "medium":            medium,
        "low":               low,
        "total_revenue":     total_rev,
        "revenue_at_risk":   rev_at_risk,
        "avg_churn_prob":    avg_churn,
        "churn_rate":        round(float(df["Churn"].mean()) * 100, 2),
    })


@app.route("/api/top_risk", methods=["GET"])
def top_risk():
    """Top 10 highest risk customers."""
    if "ChurnProbability" not in customer_df.columns:
        return jsonify({"customers": []})

    top = (
        customer_df.sort_values("RevenueAtRisk", ascending=False)
        .head(10)[[
            "CustomerID", "CustomerSegment", "TotalRevenue",
            "ChurnProbability", "RevenueAtRisk", "RiskLevel",
            "RetentionPriority", "RecommendedAction"
        ]]
    )
    result = []
    for _, r in top.iterrows():
        result.append({
            "customer_id":       int(r["CustomerID"]),
            "segment":           str(r["CustomerSegment"]),
            "total_revenue":     round(float(r["TotalRevenue"]),    2),
            "churn_probability": round(float(r["ChurnProbability"]) * 100, 2),
            "revenue_at_risk":   round(float(r["RevenueAtRisk"]),   2),
            "risk_level":        str(r["RiskLevel"]),
            "priority":          str(r["RetentionPriority"]),
            "action":            str(r["RecommendedAction"]),
        })
    return jsonify({"customers": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
