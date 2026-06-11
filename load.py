import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. LOAD DATA & FILTER CITIES
df = pd.read_csv("C:\\Users\\palla\\OneDrive\\Desktop\\AQIproject2\\archive\\city_hour.csv")
target_cities = ['Bengaluru', 'Delhi', 'Mumbai', 'Chennai', 'Hyderabad']
df = df[df['City'].isin(target_cities)]

# 2. DATA CLEANING (Fixed Index and Interpolation)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values(['City', 'Datetime'])

# Use a safer way to interpolate that avoids index ambiguity
def clean_city_data(group):
    # Convert object types to best possible numeric types before interpolating
    group = group.infer_objects(copy=False)
    # Interpolate only numeric columns
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    group[numeric_cols] = group[numeric_cols].interpolate(method='linear').ffill().bfill()
    return group

# group_keys=False prevents 'City' from becoming an ambiguous index level
df = df.groupby('City', group_keys=False).apply(clean_city_data)

# Drop rows where AQI is still missing (essential for supervised learning)
df = df.dropna(subset=['AQI']) 

# 3. TEMPORAL FEATURE ENGINEERING
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']

# Create 24-hour lag. This works now because 'City' is a clean column.
df['AQI_Lag24'] = df.groupby('City')['AQI'].shift(24)
df = df.dropna(subset=['AQI_Lag24']) # Remove rows without enough history

X = df[features + ['AQI_Lag24']]
y = df['AQI']

# 4. SCALING & SEQUENCING
# Use shuffle=False for time-series to keep 'future' data out of the 'past' training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM Stream (Samples, Time Steps, Features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 5. HYBRID ARCHITECTURE (Stream A: LSTM)
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, verbose=1)

# Stream B: Ensembles
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
xgb = XGBRegressor(n_estimators=50, learning_rate=0.05, n_jobs=-1)

rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

# 6. META-FUSION (Final Prediction)
p1 = lstm_model.predict(X_test_lstm).flatten()
p2 = rf.predict(X_test_scaled)
p3 = xgb.predict(X_test_scaled)

meta_X = np.column_stack((p1, p2, p3))
meta_learner = LinearRegression()
meta_learner.fit(meta_X, y_test)

final_preds = meta_learner.predict(meta_X)

# 7. EXPORT RESULTS
print(f"\n--- Final Hybrid Model Results ---")
print(f"R2 Score: {r2_score(y_test, final_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_test, final_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, final_preds)):.4f}")