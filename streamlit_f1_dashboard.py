import os
import fastf1

os.makedirs('f1_cache', exist_ok = True)
fastf1.Cache.enable_cache('f1_cache')

from fastf1 import get_session

# Load session data
session = get_session(2024, 'Monaco', 'Q')
session.load()

# Get all laps
laps = session.laps

# Filter to "quick" laps (exclude in-laps, out-laps, slow laps)
quick_laps = laps.pick_quicklaps()
weather = session.weather_data

import pandas as pd

columns_to_keep = ['Driver', 'Team', 'Compound', 'LapTime', 'LapNumber', 'TyreLife', 'Sector1Time', 'Sector2Time', 'Sector3Time']

df = quick_laps[columns_to_keep].copy()

# Convert time deltas to seconds
for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
    df[col] = df[col].dt.total_seconds()

# Drop rows with any missing values
df.dropna(inplace=True)

# Encode categorical columns like Driver, Team, Compound
df['Driver'] = df['Driver'].astype('category').cat.codes
df['Team'] = df['Team'].astype('category').cat.codes
df['Compound'] = df['Compound'].astype('category').cat.codes

weather = session.weather_data.copy()
weather['Time'] = pd.to_timedelta(weather['Time'])

laps = session.laps.pick_quicklaps().copy()
laps['LapStartTime'] = pd.to_timedelta(laps['LapStartTime'])

# For each lap, find the nearest weather reading
def find_closest_weather(lap_time):
    return weather.iloc[(weather['Time'] - lap_time).abs().argsort()[:1]]

# Build weather rows aligned by timestamp
aligned_weather = pd.concat([find_closest_weather(t) for t in laps['LapStartTime']], ignore_index=True)

# Reset and align
laps = laps.reset_index(drop=True)
aligned_weather = aligned_weather[['AirTemp', 'Humidity', 'TrackTemp', 'WindSpeed']].reset_index(drop=True)
df = pd.concat([df.reset_index(drop=True), aligned_weather], axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib  # to save models and scalers

features_to_scale = ['LapNumber', 'TyreLife', 'Sector1Time', 'Sector2Time',
                     'Sector3Time', 'AirTemp', 'Humidity', 'TrackTemp', 'WindSpeed']

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Predicting lap time
# Split data 
X_lap = df.drop(columns=['LapTime'])
y_lap = df['LapTime']

X_train, X_test, y_train, y_test = train_test_split(X_lap, y_lap, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_lap_model = XGBRegressor(
    n_estimators=200,      # more trees = better performance up to a point
    max_depth=6,           # controls overfitting
    learning_rate=0.05,    # smaller = more accurate, slower
    subsample=0.8,         # prevents overfitting
    colsample_bytree=0.8,  # use 80% features per tree
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

xgb_lap_model.fit(X_train, y_train)

# Save the model
joblib.dump(xgb_lap_model, 'xgb_lap_time_model.pkl')

# Predicting finishing position
position_data = session.results[['DriverNumber', 'Position']].copy()
position_data['DriverNumber'] = position_data['DriverNumber'].astype(int)

df['DriverNumber'] = laps['DriverNumber'].astype(int).values
df_pos = df.merge(position_data, on='DriverNumber')

# Train
X_pos = df_pos.drop(columns=['Position', 'LapTime'])
y_pos = df_pos['Position']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)

xgb_pos_model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_pos_model.fit(X_train_p, y_train_p)

# Save model
joblib.dump(xgb_pos_model, 'xgb_finishing_position_model.pkl')

# Save model parameters
import json

with open("xgb_lap_params.json", "w") as f:
    json.dump(xgb_lap_model.get_params(), f)

with open("xgb_pos_params.json", "w") as f:
    json.dump(xgb_pos_model.get_params(), f)

import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide")
st.title("🏎️ F1 Lap Time Prediction Dashboard")

# Load saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("xgb_lap_time_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Upload test CSV file
uploaded_file = st.file_uploader("Upload test CSV (must include LapTime column)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'LapTime' not in df.columns:
        st.error("Missing required column: LapTime")
        st.stop()

    # Separate features and labels
    X = df.drop(columns=['LapTime'])
    y_true = df['LapTime']

    # Scale numeric features only
    X[scaler.feature_names_in_] = scaler.transform(X[scaler.feature_names_in_])

    # Make predictions
    y_pred = model.predict(X)

    # Evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.3f} sec")
    col2.metric("RMSE", f"{rmse:.3f} sec")
    col3.metric("R²", f"{r2:.3f}")

    # Plot Actual vs Predicted
    st.subheader("Actual vs Predicted Lap Times")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    ax.set_xlabel("Actual Lap Time (s)")
    ax.set_ylabel("Predicted Lap Time (s)")
    ax.set_title("XGBoost Lap Time Prediction")
    ax.grid(True)
    st.pyplot(fig)

    # Show results table
    st.subheader("Results Table")
    result_df = df.copy()
    result_df['PredictedLapTime'] = y_pred
    st.dataframe(result_df[['LapTime', 'PredictedLapTime']].round(3))

    # Optional: download predictions
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, file_name='predictions.csv')



