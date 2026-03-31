from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "crop_production.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
STATE_ENCODER_PATH = BASE_DIR / "le_state.pkl"
CROP_ENCODER_PATH = BASE_DIR / "le_crop.pkl"
SEASON_ENCODER_PATH = BASE_DIR / "le_season.pkl"

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")


def train_and_save_model() -> None:
    st.write("Training model... Please wait")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    df = df[["State_Name", "Crop", "Season", "Area", "Production"]]
    df["Yield"] = df["Production"] / df["Area"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    le_state = LabelEncoder()
    le_crop = LabelEncoder()
    le_season = LabelEncoder()

    df["State_Name"] = le_state.fit_transform(df["State_Name"])
    df["Crop"] = le_crop.fit_transform(df["Crop"])
    df["Season"] = le_season.fit_transform(df["Season"])

    X = df[["State_Name", "Crop", "Season", "Area"]]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_state, STATE_ENCODER_PATH)
    joblib.dump(le_crop, CROP_ENCODER_PATH)
    joblib.dump(le_season, SEASON_ENCODER_PATH)

    st.success("Model trained and saved.")


if not MODEL_PATH.exists():
    train_and_save_model()

model = joblib.load(MODEL_PATH)
le_state = joblib.load(STATE_ENCODER_PATH)
le_crop = joblib.load(CROP_ENCODER_PATH)
le_season = joblib.load(SEASON_ENCODER_PATH)

st.title("Crop Yield Prediction System")
st.write("Enter details to predict crop yield (kg/hectare).")

state = st.selectbox("Select State", le_state.classes_)
crop = st.selectbox("Select Crop", le_crop.classes_)
season = st.selectbox("Select Season", le_season.classes_)
area = st.number_input("Enter Area (in hectares)", min_value=0.1)

state_encoded = le_state.transform([state])[0]
crop_encoded = le_crop.transform([crop])[0]
season_encoded = le_season.transform([season])[0]

if st.button("Predict Yield"):
    input_data = pd.DataFrame(
        [[state_encoded, crop_encoded, season_encoded, area]],
        columns=["State_Name", "Crop", "Season", "Area"],
    )
    prediction = model.predict(input_data)
    st.success(f"Predicted Yield: {prediction[0]:.2f} kg/hectare")

if st.checkbox("Show Dataset Insights"):
    df = pd.read_csv(DATA_PATH).dropna()
    df["Yield"] = df["Production"] / df["Area"]

    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Yield Distribution")
    st.bar_chart(df["Yield"].head(50))
