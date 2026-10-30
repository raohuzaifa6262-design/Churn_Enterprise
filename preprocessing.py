"""
preprocessing.py — Shared preprocessing function for Churn Enterprise
Used by both build_pipeline.py (training) and app.py (inference)
"""

import pandas as pd
import numpy as np


# Exact 10 raw feature columns expected as input
RAW_FEATURES = [
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

# Transformed names after log1p
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

# Which raw features get log1p
LOG_FEATURES = [
    "TotalRevenue",
    "TotalOrders",
    "TotalQuantity",
    "TotalProducts",
    "AverageOrderValue",
    "TotalReturnedQuantity",
    "ReturnOrders",
]


def preprocess(X):
    """
    Input:  DataFrame with RAW_FEATURES columns (raw values)
    Output: DataFrame with TRANSFORMED_NAMES columns (log-transformed)
    Matches the notebook's X_transformed construction exactly.
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
