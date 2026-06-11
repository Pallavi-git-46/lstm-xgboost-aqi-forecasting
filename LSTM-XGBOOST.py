import pandas as pd
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
import time
import os
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor
import random

np.random.seed(42)

tf.random.set_seed(42)

random.seed(42)

# =========================================
# 1. LOAD DATA
# =========================================

print("Loading and cleaning data...")

DATA_PATH = os.path.join(
    "data",
    "city_hour.csv"
)

df = pd.read_csv(DATA_PATH)
# =========================================
# 2. MULTI-CITY FILTER
# =========================================

target_cities = [
    "Bengaluru",
    "Delhi",
    "Mumbai",
    "Chennai",
    "Hyderabad"
]

df = df[df["City"].isin(target_cities)]

# =========================================
# 3. DATETIME PROCESSING
# =========================================

df["Datetime"] = pd.to_datetime(df["Datetime"])
print(df['Datetime'].min())
print(df['Datetime'].max())
df = df.sort_values(
    ["City", "Datetime"]
)
print("\nCity-wise Record Distribution")

city_counts = df["City"].value_counts()

print(city_counts)
# =========================================
# 4. CLEANING FUNCTION
# =========================================

def clean_city_data(group):

    group = group.infer_objects(copy=False)

    numeric_cols = group.select_dtypes(
        include=[np.number]
    ).columns

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

# =========================================
# 5. FEATURE ENGINEERING
# =========================================

print("Creating temporal features...")

# -----------------------------------------
# 24-Hour Lag Feature
# -----------------------------------------

df["AQI_Lag24"] = (
    df.groupby("City")["AQI"]
    .shift(24)
)

# -----------------------------------------
# 12-Hour Rolling Mean
# Uses ONLY past values
# -----------------------------------------

df["AQI_Roll12"] = (
    df.groupby("City")["AQI"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(window=12)
        .mean()
    )
)

# -----------------------------------------
# 6-Hour Rolling Standard Deviation
# -----------------------------------------

df["AQI_Std6"] = (
    df.groupby("City")["AQI"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(window=6)
        .std()
    )
)
# =========================================
# 6. REMOVE MISSING VALUES
# =========================================

df = df.dropna(
    subset=[
        "AQI",
        "AQI_Lag24",
        "AQI_Roll12",
        "AQI_Std6"
    ]
)
# =========================================
# 7. FEATURES & TARGET
# =========================================

features = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI_Lag24",
    "AQI_Roll12",
    "AQI_Std6"
]

X = df[features]

y = df["AQI"]

print("Final Dataset Shape:", X.shape)

# =========================================
# 8. CITY-WISE TRAIN TEST SPLIT
# =========================================

train_parts = []

test_parts = []

for city in target_cities:

    city_df = df[
        df["City"] == city
    ]

    # Preserve temporal order
    split_index = int(
        len(city_df) * 0.8
    )

    # First 80% → Train
    train_parts.append(
        city_df.iloc[:split_index]
    )

    # Last 20% → Test
    test_parts.append(
        city_df.iloc[split_index:]
    )

# Combine all city datasets
train_df = pd.concat(train_parts)

test_df = pd.concat(test_parts)

# -----------------------------------------
# Features & Target
# -----------------------------------------

X_train = train_df[features]

y_train = train_df["AQI"]

X_test = test_df[features]

y_test = test_df["AQI"]

print("Training Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)

# =========================================
# 9. FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Full scaled dataset for cross-validation      
X_scaled = scaler.transform(X)
# =========================================
# 10. STAGE I - LSTM BASE MODEL
# =========================================
overall_start_time = time.time()        
print("\n[Stage 1] Training LSTM Base Model...")

# Reshape for LSTM
X_train_lstm = X_train_scaled.reshape(
    (
        X_train_scaled.shape[0],
        1,
        X_train_scaled.shape[1]
    )
)

X_test_lstm = X_test_scaled.reshape(
    (
        X_test_scaled.shape[0],
        1,
        X_test_scaled.shape[1]
    )
)

# Build Model
lstm_base = Sequential([

    LSTM(
        64,
        return_sequences=False,
        input_shape=(
            1,
            X_train_scaled.shape[1]
        )
    ),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])

# Compile
lstm_base.compile(
    optimizer="adam",
    loss="mse"
)

# Early Stopping
early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

# Train
lstm_start = time.time()

lstm_base.fit(
    X_train_lstm,
    y_train,
    epochs=20,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop]
)

lstm_training_time = (
    time.time() - lstm_start
)

print(
    f"LSTM Training Time: {lstm_training_time:.2f} seconds"
)

# =========================================
# 11. BASE PREDICTIONS
# =========================================

lstm_train_preds = (
    lstm_base.predict(
        X_train_lstm,
        verbose=0
    )
    .flatten()
)

lstm_test_preds = (
    lstm_base.predict(
        X_test_lstm,
        verbose=0
    )
    .flatten()
)

# =========================================
# 11b. BENCHMARK LAYER - ATTENTION-LSTM
# =========================================

print("\n[Benchmark] Training Attention-LSTM Model...")

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# =========================================
# CUSTOM TEMPORAL ATTENTION LAYER
# =========================================

class TemporalAttention(Layer):

    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )

        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(TemporalAttention, self).build(input_shape)

    def call(self, x):

        # Attention score
        et = K.squeeze(
            K.tanh(K.dot(x, self.W) + self.b),
            axis=-1
        )

        # Attention weights
        at = K.softmax(et)

        at = K.expand_dims(at, axis=-1)

        # Weighted sequence
        output = x * at

        return K.sum(output, axis=1)

# =========================================
# RESHAPE DATA FOR ATTENTION MODEL
# =========================================

X_train_att = X_train_scaled.reshape(
    (
        X_train_scaled.shape[0],
        1,
        X_train_scaled.shape[1]
    )
)

X_test_att = X_test_scaled.reshape(
    (
        X_test_scaled.shape[0],
        1,
        X_test_scaled.shape[1]
    )
)

# =========================================
# BUILD ATTENTION-LSTM MODEL
# =========================================

attention_lstm = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(1, X_train_scaled.shape[1])
    ),

    TemporalAttention(),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)

])

# =========================================
# COMPILE MODEL
# =========================================

attention_lstm.compile(
    optimizer="adam",
    loss="mse"
)

# =========================================
# TRAIN MODEL
# =========================================

attention_lstm.fit(

    X_train_att,
    y_train,

    epochs=15,

    batch_size=64,

    verbose=1,

    callbacks=[early_stop]

)

# =========================================
# GENERATE PREDICTIONS
# =========================================

att_test_preds = attention_lstm.predict(
    X_test_att,
    verbose=0
).flatten()

# =========================================
# EVALUATION
# =========================================

att_r2 = r2_score(
    y_test,
    att_test_preds
)

att_mae = mean_absolute_error(
    y_test,
    att_test_preds
)

att_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        att_test_preds
    )
)

print("\n" + "=" * 50)
print("ATTENTION-LSTM RESULTS")
print("=" * 50)

print(f"R2 Score : {att_r2:.4f}")
print(f"MAE      : {att_mae:.4f}")
print(f"RMSE     : {att_rmse:.4f}")

# =========================================
# 12. RESIDUAL CALCULATION
# =========================================

print("\nCalculating residuals...")

train_residuals = (
    y_train - lstm_train_preds
)

print(
    "Mean Residual:",
    np.mean(train_residuals)
)

# =========================================
# 13. XGBOOST RESIDUAL CORRECTOR
# =========================================

print(
    "\n[Stage 2] Training XGBoost Residual Corrector..."
)

xgb_corrector = XGBRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)

# Train on residuals
xgb_start = time.time()

xgb_corrector.fit(
    X_train_scaled,
    train_residuals
)

xgb_training_time = (
    time.time() - xgb_start
)

print(
    f"XGBoost Training Time: {xgb_training_time:.2f} seconds"
)

# =========================================
# 14. FINAL RESIDUAL CORRECTION
# =========================================

print(
    "\n[Stage 3] Performing Residual Correction..."
)

predicted_residuals = (
    xgb_corrector.predict(
        X_test_scaled
    )
)
inference_start = time.time()
# Final Prediction
final_preds = (
    lstm_test_preds +
    predicted_residuals
)
inference_time = (
    time.time() - inference_start
)

print(
    f"Inference Time: {inference_time:.4f} seconds"
)
# =========================================
# 15. EVALUATION
# =========================================

r2 = r2_score(
    y_test,
    final_preds
)

mae = mean_absolute_error(
    y_test,
    final_preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_preds
    )
)

print("\n" + "=" * 45)
print("HYBRID LSTM-XGBOOST RESULTS")
print("=" * 45)

print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
# =========================================
# TEMPORAL CROSS-VALIDATION
# =========================================

from sklearn.model_selection import TimeSeriesSplit

print("\nPerforming Time-Series Cross-Validation...")

tscv = TimeSeriesSplit(n_splits=5)

cv_r2 = []
cv_mae = []
cv_rmse = []

fold = 1

for train_index, test_index in tscv.split(X_scaled):

    print(f"\nProcessing Fold {fold}...")

    # Split data
    X_train_cv = X_scaled[train_index]
    X_test_cv = X_scaled[test_index]

    y_train_cv = y.iloc[train_index]
    y_test_cv = y.iloc[test_index]

    # Train XGBoost only for fast validation
    xgb_cv = XGBRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    xgb_cv.fit(
        X_train_cv,
        y_train_cv
    )

    preds_cv = xgb_cv.predict(
        X_test_cv
    )

    # Metrics
    r2_cv = r2_score(
        y_test_cv,
        preds_cv
    )

    mae_cv = mean_absolute_error(
        y_test_cv,
        preds_cv
    )

    rmse_cv = np.sqrt(
        mean_squared_error(
            y_test_cv,
            preds_cv
        )
    )

    cv_r2.append(r2_cv)
    cv_mae.append(mae_cv)
    cv_rmse.append(rmse_cv)

    print(f"Fold {fold} R2   : {r2_cv:.4f}")
    print(f"Fold {fold} MAE  : {mae_cv:.4f}")
    print(f"Fold {fold} RMSE : {rmse_cv:.4f}")

    fold += 1

# =========================================
# FINAL CV RESULTS
# =========================================

print("\n" + "=" * 50)
print("TEMPORAL CROSS-VALIDATION SUMMARY")
print("=" * 50)

print(f"Mean R2   : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
print(f"Mean MAE  : {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")
print(f"Mean RMSE : {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")

# =========================================
# 18. CITY-WISE PERFORMANCE ANALYSIS
# =========================================

print("\n" + "=" * 45)
print("CITY-WISE PERFORMANCE ANALYSIS")
print("=" * 45)

# -----------------------------------------
# Create test dataframe
# -----------------------------------------

test_data = X_test.copy()

# Actual AQI
test_data["Actual_AQI"] = y_test.values

# Predicted AQI
test_data["Predicted_AQI"] = final_preds

# Correct city labels
test_data["City"] = test_df["City"].values

# -----------------------------------------
# Calculate city-wise metrics
# -----------------------------------------

cities = test_data["City"].unique()

city_results = []

for city in cities:

    city_data = test_data[
        test_data["City"] == city
    ]

    actual = city_data["Actual_AQI"]

    predicted = city_data["Predicted_AQI"]

    city_r2 = r2_score(
        actual,
        predicted
    )

    city_mae = mean_absolute_error(
        actual,
        predicted
    )

    city_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    city_results.append([
        city,
        city_r2,
        city_mae,
        city_rmse
    ])

    print(f"\nCity: {city}")

    print(f"R2   : {city_r2:.4f}")

    print(f"MAE  : {city_mae:.4f}")

    print(f"RMSE : {city_rmse:.4f}")

# -----------------------------------------
# Save Results
# -----------------------------------------

results_df = pd.DataFrame(
    city_results,
    columns=[
        "City",
        "R2 Score",
        "MAE",
        "RMSE"
    ]
)

results_df.to_csv(
    "citywise_results.csv",
    index=False
)

print("\nExecution Completed Successfully!")
# -----------------------------------------
# Save Results Table
# -----------------------------------------

results_df = pd.DataFrame(
    city_results,
    columns=[
        "City",
        "R2 Score",
        "MAE",
        "RMSE"
    ]
)

results_df.to_csv(
    "citywise_results.csv",
    index=False
)
"""
PATCH FOR ORIGINAL SCRIPT
==========================
1. Delete EVERY plt.savefig() and plt.show() block from your original script
   (sections 16 through the residual histogram at the end).
2. Keep all the training, prediction, and evaluation code unchanged.
3. Paste THIS block at the very end of your script, after all models are trained.
"""

# ---- paste below the last print("Execution Completed Successfully!") ----

from plot_publication_figures import generate_all_figures

# Fill in the actual metric values your run produced
results_dict = {
    "LSTM":            {"R2": 0.7481, "MAE": 13.1500, "RMSE": 17.1800},
    "XGBoost":         {"R2": 0.8309, "MAE":  8.3923, "RMSE": 14.0745},
    "Attention-LSTM":  {"R2": 0.9746, "MAE":  8.0936, "RMSE": 12.3032},
    "Proposed Hybrid": {"R2": r2,     "MAE":  mae,    "RMSE": rmse    },
}

generate_all_figures(
    xgb_corrector  = xgb_corrector,
    X_test_scaled  = X_test_scaled,
    features       = features,
    y_test         = y_test,
    final_preds    = final_preds,
    r2             = r2,
    results_dict   = results_dict,
)

# =========================================
# CROSS-CITY GENERALIZATION STUDY
# =========================================
print("\n" + "="*60)
print("CROSS-CITY GENERALIZATION ANALYSIS")
print("="*60)

cross_city_results = []

for test_city in target_cities:

    print(f"\nTesting on unseen city: {test_city}")

    # Train on remaining cities
    train_city_df = df[df["City"] != test_city]

    # Test on unseen city
    test_city_df = df[df["City"] == test_city]

    X_train_cc = train_city_df[features]
    y_train_cc = train_city_df["AQI"]

    X_test_cc = test_city_df[features]
    y_test_cc = test_city_df["AQI"]

    scaler_cc = StandardScaler()

    X_train_cc_scaled = scaler_cc.fit_transform(X_train_cc)
    X_test_cc_scaled = scaler_cc.transform(X_test_cc)

    # Train XGBoost
    model_cc = XGBRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    model_cc.fit(
        X_train_cc_scaled,
        y_train_cc
    )

    preds_cc = model_cc.predict(
        X_test_cc_scaled
    )

    r2_cc = r2_score(
        y_test_cc,
        preds_cc
    )

    mae_cc = mean_absolute_error(
        y_test_cc,
        preds_cc
    )

    rmse_cc = np.sqrt(
        mean_squared_error(
            y_test_cc,
            preds_cc
        )
    )

    cross_city_results.append([
        test_city,
        r2_cc,
        mae_cc,
        rmse_cc
    ])

    print(
        f"R2={r2_cc:.4f} "
        f"MAE={mae_cc:.4f} "
        f"RMSE={rmse_cc:.4f}"
    )
    cross_city_df = pd.DataFrame(
    cross_city_results,
    columns=[
        "Unseen City",
        "R2",
        "MAE",
        "RMSE"
    ]
)

cross_city_df.to_csv(
    "cross_city_results.csv",
    index=False
)

print(cross_city_df)


# =========================================
# COMPUTATIONAL EFFICIENCY ANALYSIS
# =========================================
# =========================================
# TRAIN MODEL
# =========================================

att_start_train = time.time()

attention_lstm.fit(
    X_train_att,
    y_train,
    epochs=20,          # ← match LSTM's 20 epochs
    batch_size=64,
    verbose=1,
    # callbacks=[early_stop]   ← remove this line
)

attention_training_time = time.time() - att_start_train

print("\n" + "="*60)
print("COMPUTATIONAL PERFORMANCE COMPARISON")
print("="*60)

# LSTM Training Time
print(
    f"LSTM Training Time           : "
    f"{lstm_training_time:.4f} seconds"
)

# Attention-LSTM Training Time
print(
    f"Attention-LSTM Training Time : "
    f"{attention_training_time:.4f} seconds"
)

# XGBoost Training Time
print(
    f"XGBoost Training Time        : "
    f"{xgb_training_time:.4f} seconds"
)

# Hybrid Training Time
hybrid_training_time = (
    lstm_training_time +
    xgb_training_time
)

print(
    f"Hybrid Training Time         : "
    f"{hybrid_training_time:.4f} seconds"
)

# =========================================
# HYBRID INFERENCE TIME
# =========================================

hybrid_start = time.time()

lstm_preds = lstm_base.predict(
    X_test_lstm,
    verbose=0
).flatten()

xgb_preds = xgb_corrector.predict(
    X_test_scaled
)

hybrid_preds = (
    lstm_preds +
    xgb_preds
)

hybrid_inference_time = (
    time.time() - hybrid_start
)

print(
    f"Hybrid Inference Time        : "
    f"{hybrid_inference_time:.4f} seconds"
)