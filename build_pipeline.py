"""
Step 25 + 26: Build & Save Final Preprocessing Pipeline
Reproduces EXACTLY the notebook's feature engineering and saves:
  - models/preprocessing.pkl   (FunctionTransformer)
  - models/rf_model.pkl        (Random Forest)
  - models/feature_names.pkl   (list of 10 feature names)
  - data/processed/customer_df.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel("data/raw/online_retail.xlsx", sheet_name=None)
df_retail = pd.concat([df["Year 2009-2010"], df["Year 2010-2011"]], ignore_index=True)

retail = df_retail.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"})
retail = retail.dropna(subset=["CustomerID"])

# Keep only valid purchases (positive qty & price)
purchases = retail[(retail["Quantity"] > 0) & (retail["UnitPrice"] > 0)].copy()
purchases["Revenue"] = purchases["Quantity"] * purchases["UnitPrice"]

# Returns (negative qty)
returns = retail[retail["Quantity"] < 0].copy()
returns["ReturnedQuantity"] = returns["Quantity"].abs()

print(f"Purchases: {purchases.shape[0]:,} rows | Returns: {returns.shape[0]:,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL SPLIT  (exact same as notebook)
# ─────────────────────────────────────────────────────────────────────────────
cutoff_date = pd.Timestamp("2011-09-10")

past_data   = purchases[purchases["InvoiceDate"] <= cutoff_date].copy()
future_data = purchases[purchases["InvoiceDate"] >  cutoff_date].copy()

print(f"Past: {past_data.shape[0]:,} | Future: {future_data.shape[0]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CHURN LABEL
# ─────────────────────────────────────────────────────────────────────────────
future_customers = set(future_data["CustomerID"].unique())
past_customers   = past_data["CustomerID"].unique()

churn_target = pd.DataFrame({"CustomerID": past_customers})
churn_target["Churn"] = (~churn_target["CustomerID"].isin(future_customers)).astype(int)

print(f"Churn rate: {churn_target['Churn'].mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAST FEATURES  (only from past_data)
# ─────────────────────────────────────────────────────────────────────────────
agg = past_data.groupby("CustomerID").agg(
    TotalRevenue    = ("Revenue",       "sum"),
    TotalOrders     = ("Invoice",       "nunique"),
    TotalQuantity   = ("Quantity",      "sum"),
    TotalProducts   = ("StockCode",     "nunique"),
    FirstPurchaseDate = ("InvoiceDate", "min"),
    LastPurchaseDate  = ("InvoiceDate", "max"),
).reset_index()

agg["AverageOrderValue"]  = agg["TotalRevenue"]  / agg["TotalOrders"]
agg["RecencyDays"]        = (cutoff_date - agg["LastPurchaseDate"]).dt.days
agg["CustomerTenureDays"] = (cutoff_date - agg["FirstPurchaseDate"]).dt.days

# Returns per customer
cust_returns = (
    returns.groupby("CustomerID")["ReturnedQuantity"].sum().reset_index()
          .rename(columns={"ReturnedQuantity": "TotalReturnedQuantity"})
)
return_invoices = (
    returns.groupby("CustomerID")["Invoice"].nunique().reset_index()
           .rename(columns={"Invoice": "ReturnOrders"})
)

past_features = agg.merge(cust_returns,   on="CustomerID", how="left")
past_features = past_features.merge(return_invoices, on="CustomerID", how="left")
past_features["TotalReturnedQuantity"] = past_features["TotalReturnedQuantity"].fillna(0)
past_features["ReturnOrders"]          = past_features["ReturnOrders"].fillna(0)

# ReturnOrderRate (for compatibility, not used in final model features)
past_features["ReturnOrderRate"] = (
    past_features["ReturnOrders"] / past_features["TotalOrders"]
)

# Country (first recorded)
cust_country = (
    past_data.groupby("CustomerID")["Country"].first().reset_index()
)
past_features = past_features.merge(cust_country, on="CustomerID", how="left")

print(f"past_features shape: {past_features.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MERGE → customer_ml
# ─────────────────────────────────────────────────────────────────────────────
customer_ml = past_features.merge(churn_target, on="CustomerID", how="inner")

# ReturnQuantityRate  (fix NaN/inf)
customer_ml["ReturnQuantityRate"] = (
    customer_ml["TotalReturnedQuantity"] / customer_ml["TotalQuantity"]
).fillna(0).clip(0, 10)   # clip prevents inf from very small denominators

print(f"customer_ml shape: {customer_ml.shape}")
print(f"Churn distribution:\n{customer_ml['Churn'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CUSTOMER SEGMENTATION (simple RFM-based, matches notebook)
# ─────────────────────────────────────────────────────────────────────────────
def assign_segment(row):
    rev  = row["TotalRevenue"]
    ord_ = row["TotalOrders"]
    rec  = row["RecencyDays"]
    if rev >= 5000 and ord_ >= 5:
        return "VIP / High-Value Loyal"
    elif rec <= 90 and ord_ >= 3:
        return "Active Regular"
    elif rec > 270 or ord_ == 1:
        return "At-Risk"
    elif ord_ <= 2:
        return "New / Occasional"
    else:
        return "Moderate Regular"

customer_ml["CustomerSegment"] = customer_ml.apply(assign_segment, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD X, y  (exact 10 features from notebook)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "RecencyDays",
    "TotalOrders",
    "TotalRevenue",
    "TotalQuantity",
    "TotalProducts",
    "AverageOrderValue",
    "CustomerTenureDays",
    "TotalReturnedQuantity",
    "ReturnOrders",
    "ReturnQuantityRate",
]

LOG_FEATURES = [
    "TotalRevenue",
    "TotalOrders",
    "TotalQuantity",
    "TotalProducts",
    "AverageOrderValue",
    "TotalReturnedQuantity",
    "ReturnOrders",
]

TRANSFORMED_NAMES = [
    "RecencyDays",
    "LogTotalOrders",
    "LogTotalRevenue",
    "LogTotalQuantity",
    "LogTotalProducts",
    "LogAverageOrderValue",
    "CustomerTenureDays",
    "LogTotalReturnedQuantity",
    "LogReturnOrders",
    "ReturnQuantityRate",
]

X_raw = customer_ml[MODEL_FEATURES].copy()
y     = customer_ml["Churn"].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 8. PREPROCESSING FUNCTION  (matches notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(X):
    """
    Input:  DataFrame with MODEL_FEATURES columns (raw values)
    Output: DataFrame with TRANSFORMED_NAMES columns (log-transformed)
    """
    df = X.copy()
    out = pd.DataFrame(index=df.index)
    out["RecencyDays"]              = df["RecencyDays"]
    out["LogTotalOrders"]           = np.log1p(df["TotalOrders"])
    out["LogTotalRevenue"]          = np.log1p(df["TotalRevenue"])
    out["LogTotalQuantity"]         = np.log1p(df["TotalQuantity"])
    out["LogTotalProducts"]         = np.log1p(df["TotalProducts"])
    out["LogAverageOrderValue"]     = np.log1p(df["AverageOrderValue"])
    out["CustomerTenureDays"]       = df["CustomerTenureDays"]
    out["LogTotalReturnedQuantity"] = np.log1p(df["TotalReturnedQuantity"])
    out["LogReturnOrders"]          = np.log1p(df["ReturnOrders"])
    out["ReturnQuantityRate"]       = df["ReturnQuantityRate"]
    return out

X_transformed = preprocess(X_raw)
print(f"X_transformed shape: {X_transformed.shape}")
print(f"Feature names: {TRANSFORMED_NAMES}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN / TEST SPLIT  (same as notebook)
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.20, random_state=42, stratify=y
)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAIN RANDOM FOREST  (params from notebook rf_model)
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc  = rf_model.score(X_test,  y_test)
print(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SKLEARN PIPELINE  (preprocessing → RF)
# ─────────────────────────────────────────────────────────────────────────────
preprocessing = FunctionTransformer(preprocess, validate=False)

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model",         rf_model),
])

print("Pipeline created successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# 12. BUSINESS LOGIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def risk_level(probability):
    if   probability >= 0.75: return "Critical"
    elif probability >= 0.50: return "High"
    elif probability >= 0.30: return "Medium"
    else:                     return "Low"

def retention_priority(score):
    if   score >= 70: return "P1 - Immediate Action"
    elif score >= 50: return "P2 - High Priority"
    elif score >= 30: return "P3 - Monitor"
    else:             return "P4 - Low Priority"

def retention_action(row):
    segment = row["CustomerSegment"]
    risk    = row["RiskLevel"]
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
# 13. BUILD FULL customer_df  (all customers, all predictions)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding full customer_df with predictions...")

churn_proba = rf_model.predict_proba(X_transformed)[:, 1]
churn_pred  = rf_model.predict(X_transformed)

customer_df = customer_ml[[
    "CustomerID", "TotalRevenue", "TotalOrders", "TotalQuantity",
    "TotalProducts", "AverageOrderValue", "RecencyDays",
    "CustomerTenureDays", "TotalReturnedQuantity", "ReturnOrders",
    "ReturnQuantityRate", "Country", "CustomerSegment", "Churn",
    "FirstPurchaseDate", "LastPurchaseDate"
]].copy()

customer_df["ChurnProbability"] = churn_proba
customer_df["ChurnPrediction"]  = churn_pred
customer_df["RevenueAtRisk"]    = (
    customer_df["TotalRevenue"] * customer_df["ChurnProbability"]
)
customer_df["RiskLevel"] = customer_df["ChurnProbability"].apply(risk_level)

max_rev_at_risk = customer_df["RevenueAtRisk"].max()
customer_df["RetentionPriorityScore"] = (
    customer_df["ChurnProbability"] * 0.6
    + (customer_df["RevenueAtRisk"] / max_rev_at_risk) * 0.4
) * 100
customer_df["RetentionPriorityScore"] = customer_df["RetentionPriorityScore"].round(2)
customer_df["RetentionPriority"]  = customer_df["RetentionPriorityScore"].apply(retention_priority)
customer_df["RecommendedAction"]  = customer_df.apply(retention_action, axis=1)

print(f"customer_df shape: {customer_df.shape}")
print(customer_df["RiskLevel"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Save preprocessing function
with open("models/preprocessing.pkl", "wb") as f:
    pickle.dump(preprocess, f)

# Save RF model
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save full pipeline
with open("models/full_pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump({
        "raw_features":         MODEL_FEATURES,
        "transformed_features": TRANSFORMED_NAMES,
        "log_features":         LOG_FEATURES,
    }, f)

# Save customer_df
customer_df.to_csv("data/processed/customer_df.csv", index=True)

print("\n✅ All artifacts saved:")
print("  models/preprocessing.pkl")
print("  models/rf_model.pkl")
print("  models/full_pipeline.pkl")
print("  models/feature_names.pkl")
print("  data/processed/customer_df.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 15. QUICK VERIFICATION  (match notebook: CustomerID 15361 → ~76.4%)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Verification (CustomerID 15361) ──")
cust = customer_df[customer_df["CustomerID"] == 15361.0]
if len(cust) > 0:
    row = cust.iloc[0]
    print(f"  ChurnProbability : {row['ChurnProbability']:.4f}")
    print(f"  TotalRevenue     : £{row['TotalRevenue']:.2f}")
    print(f"  RevenueAtRisk    : £{row['RevenueAtRisk']:.2f}")
    print(f"  RiskLevel        : {row['RiskLevel']}")
    print(f"  RetentionPriority: {row['RetentionPriority']}")
    print(f"  RecommendedAction: {row['RecommendedAction']}")
else:
    print("  CustomerID 15361 not found in past_data customers.")

print("\nDone! ✅")
