import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# =========================================
# LOAD DATA
# =========================================

DATA_PATH = os.path.join(
    "data",
    "city_hour.csv"
)

df = pd.read_csv(DATA_PATH)
# MULTI-CITY DATA
target_cities = [
    "Bengaluru",
    "Delhi",
    "Mumbai",
    "Chennai",
    "Hyderabad"
]

df = df[df["City"].isin(target_cities)]

# DATETIME
df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values(["City", "Datetime"])

# =========================================
# CLEANING
# =========================================

def clean_city_data(group):

    numeric_cols = group.select_dtypes(include=[np.number]).columns

    group[numeric_cols] = (
        group[numeric_cols]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    return group

df = df.groupby(
    "City",
    group_keys=False
).apply(clean_city_data)

df = df.dropna(subset=["AQI"])

# =========================================
# FEATURES
# =========================================

features = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3"
]

# LAG FEATURE
df["AQI_Lag24"] = df.groupby("City")["AQI"].shift(24)

df = df.dropna(subset=["AQI_Lag24"])

X = df[features + ["AQI_Lag24"]]
y = df["AQI"]

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =========================================
# SCALING
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# XGBOOST MODEL
# =========================================

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

# =========================================
# TRAINING TIME
# =========================================

train_start = time.time()

model.fit(
    X_train_scaled,
    y_train
)

train_end = time.time()

training_time = train_end - train_start

# =========================================
# INFERENCE TIME
# =========================================

infer_start = time.time()

y_pred = model.predict(
    X_test_scaled
)

infer_end = time.time()

inference_time = infer_end - infer_start

# =========================================
# EVALUATION
# =========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

# =========================================
# RESULTS
# =========================================

print("\n===== XGBOOST RESULTS =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

print("\n===== COMPUTATIONAL PERFORMANCE =====")

print(f"Training Time  : {training_time:.4f} seconds")
print(f"Inference Time : {inference_time:.4f} seconds")